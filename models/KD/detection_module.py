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

        """Trains the module parameters.
        Checkout `Module Tutorial <http://mxnet.io/tutorials/basic/module.html>`_ to see
        a end-to-end use-case.
        Parameters
        ----------
        train_data : DataIter
            Train DataIter.
        eval_data : DataIter
            If not ``None``, will be used as validation set and the performance
            after each epoch will be evaluated.
        eval_metric : str or EvalMetric
            Defaults to 'accuracy'. The performance measure used to display during training.
            Other possible predefined metrics are:
            'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
        epoch_end_callback : function or list of functions
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Defaults to 'local'.
        optimizer : str or Optimizer
            Defaults to 'sgd'.
        optimizer_params : dict
            Defaults to ``(('learning_rate', 0.01),)``. The parameters for
            the optimizer constructor.
            The default value is not a dict, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each mini-batch during evaluation.
        initializer : Initializer
            The initializer is called to initialize the module parameters when they are
            not already initialized.
        arg_params : dict
            Defaults to ``None``, if not ``None``, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has a higher priority than `initializer`.
        aux_params : dict
            Defaults to ``None``. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Defaults to ``False``. Indicates whether to allow missing parameters when `arg_params`
            and `aux_params` are not ``None``. If this is ``True``, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Defaults to ``False``. Whether to force rebinding the executors if already bound.
        force_init : bool
            Defaults to ``False``. Indicates whether to force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Defaults to 0. Indicates the starting epoch. Usually, if resumed from a
            checkpoint saved at a previous training phase at epoch N, then this value should be
            N+1.
        num_epoch : int
            Number of epochs for training.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        """
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, 
                  label_shapes=train_data.provide_label + self.teacher_label_shapes,
                  for_training=True, force_rebind=force_rebind)

        self.logger.info("MEM usage: {} MiB".
            format(int(self._exec_group.execs[0].debug_str().split('\n')[-3].split()[1])))

        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        total_iter = 0
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                if profile is True and epoch == begin_epoch and nbatch == 1:
                    self.logger.info("Profiling begins")
                    import mxnet as mx
                    mx.profiler.set_state("run")

                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()

                if isinstance(data_batch, list):
                    self.update_metric(eval_metric,
                                       [db.label for db in data_batch],
                                       pre_sliced=True)
                else:
                    self.update_metric(eval_metric, data_batch.label)

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
                except StopIteration:
                    end_of_batch = True

                if monitor is not None:
                    monitor.toc_print()

                if end_of_batch:
                    eval_name_vals = eval_metric.get_name_value()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1
                total_iter += 1

                if profile is True and epoch == begin_epoch and nbatch == 10:
                    self.logger.info("Profiling ends")
                    mx.profiler.set_state("stop")
                    mx.profiler.dump()

            # one epoch of training is finished
            for name, val in eval_name_vals:
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None and self._kvstore.rank == 0:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()
