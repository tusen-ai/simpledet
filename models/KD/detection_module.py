# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-instance-attributes, too-many-arguments, protected-access, too-many-branches
# pylint: disable=too-many-public-methods
"""A `Module` implement the `BaseModule` API by wrapping a `Symbol` and one or
more `Executor` for data parallelization.
"""
import time
import logging
import warnings

import mxnet as mx

from mxnet import metric
from mxnet import context as ctx
from mxnet import optimizer as opt
from mxnet import ndarray as nd

from mxnet.base import _as_list
from mxnet.module.executor_group import DataParallelExecutorGroup
from mxnet.model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from mxnet.model import load_checkpoint
from mxnet.model import BatchEndParam
from mxnet.initializer import Uniform, InitDesc
from mxnet.io import DataDesc, DataBatch
from mxnet.ndarray import zeros

from mxnet.module.base_module import BaseModule, _check_input_names, _parse_data_desc
from mxnet.module.module import Module
from core.detection_module import DetModule


class KDDetModule(DetModule):
    """Module is a basic module that wrap a `Symbol`. It is functionally the same
    as the `FeedForward` model, except under the module API.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
        Defaults to `('data')` for a typical model used in image classification.
    label_names : list of str
        Defaults to `('softmax_label')` for a typical model used in image
        classification.
    logger : Logger
        Defaults to `logging`.
    context : Context or list of Context
        Defaults to ``mx.cpu()``.
    work_load_list : list of number
        Default ``None``, indicating uniform workload.
    fixed_param_names: list of str
        Default ``None``, indicating no network parameters are fixed.
    state_names : list of str
        states are similar to data and label, but not provided by data iterator.
        Instead they are initialized to 0 and can be set by `set_states()`.
    group2ctxs : dict of str to context or list of context,
                 or list of dict of str to context
        Default is `None`. Mapping the `ctx_group` attribute to the context assignment.
    compression_params : dict
        Specifies type of gradient compression and additional arguments depending
        on the type of compression being used. For example, 2bit compression requires a threshold.
        Arguments would then be {'type':'2bit', 'threshold':0.5}
        See mxnet.KVStore.set_gradient_compression method for more details on gradient compression.
    """
    def __init__(self, symbol, teacher_module=None, teacher_label_names=None, teacher_label_shapes=None,
                 data_names=None, label_names=None, logger=logging, context=ctx.cpu(),
                 fixed_param=None, excluded_param=None):
        super().__init__(symbol=symbol, data_names=data_names, 
                         label_names=label_names + teacher_label_names, logger=logger, context=context,
                         fixed_param=fixed_param, excluded_param=excluded_param)
        
        assert isinstance(teacher_module, DetModule)
        self.teacher_module = teacher_module
        self.teacher_label_shapes = teacher_label_shapes
        self.t_output = None

    def forward(self, data_batch, is_train=None):
        """Forward computation. It supports data batches with different shapes, such as
        different batch sizes or different image sizes.
        If reshaping of data batch relates to modification of symbol or module, such as
        changing image layout ordering or switching from training to predicting, module
        rebinding is required.

        See Also
        ----------
        :meth:`BaseModule.forward`.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is ``None``, which means ``is_train`` takes the value of ``self.for_training``.
        """
        assert self.binded and self.params_initialized

        curr_data_shapes = tuple(i.shape for i in self._data_shapes)

        if isinstance(data_batch, list):
            assert data_batch is not None, "Encountered empty data batch"
            new_data_shapes = []
            for i in range(len(data_batch[0].data)):
                shape = data_batch[0].data[i].shape
                for db in data_batch:
                    assert shape == db.data[i].shape, \
                        "All data batches in a list need to have the same shape"
                new_batch_size = len(data_batch) * shape[0]
                new_data_shapes.append((new_batch_size,) + shape[1:])
            new_data_shapes = tuple(new_data_shapes)
        else:
            new_data_shapes = tuple(i.shape for i in data_batch.data)

        if curr_data_shapes != new_data_shapes:
            if hasattr(data_batch, "provide_data") and data_batch.provide_data:
                new_dshape = data_batch.provide_data
            else:
                new_dshape = [DataDesc(i.name, shape, i.dtype, i.layout) \
                              for i, shape in zip(self._data_shapes, new_data_shapes)]

            if hasattr(data_batch, "provide_label") and data_batch.provide_label:
                new_lshape = data_batch.provide_label
            elif hasattr(data_batch, "label") and data_batch.label:
                new_lshape = [DataDesc(i.name, j.shape, i.dtype, i.layout) \
                              for i, j in zip(self._label_shapes, data_batch.label)]
            else:
                new_lshape = None

            # TODO: hard code
            self.teacher_module.reshape(new_dshape[:1], None)
            t_data_batch = DataBatch(data=data_batch.data[:1], 
                                     provide_data=data_batch.provide_data[:1])
            self.teacher_module.forward(data_batch=t_data_batch, is_train=True)
            # TODO: should handle multi teacher label
            self.t_output = self.teacher_module.get_outputs()
            t_shape = self.t_output[0].shape
            new_lshape += [('teacher_label', t_shape)]
            self.reshape(new_dshape, new_lshape)
        if self.t_output is None:
            t_data_batch = DataBatch(data=data_batch.data[:1], 
                                    provide_data=data_batch.provide_data[:1])
            self.teacher_module.forward(data_batch=t_data_batch, is_train=True)
            self.t_output = self.teacher_module.get_outputs()
        for data in self.t_output:
            data.wait_to_read()
        data_batch.label += self.t_output
        self.t_output = None

        self._exec_group.forward(data_batch, is_train)

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, sparse_row_id_fn=None, profile=False):

        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, 
                  label_shapes=train_data.provide_label + self.teacher_label_shapes,
                  for_training=True, force_rebind=force_rebind)
        super().fit(force_rebind=False, train_data=train_data, eval_data=eval_data, eval_metric=eval_metric,
                    epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback,
                    kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params,
                    eval_end_callback=eval_end_callback,
                    eval_batch_end_callback=eval_batch_end_callback, initializer=initializer,
                    arg_params=arg_params, aux_params=aux_params, allow_missing=allow_missing,
                    force_init=force_init, begin_epoch=begin_epoch,
                    num_epoch=num_epoch, validation_metric=validation_metric, monitor=monitor,
                    sparse_row_id_fn=sparse_row_id_fn, profile=profile)