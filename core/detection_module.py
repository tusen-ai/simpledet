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

import mxnet

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
from mxnet.io import DataDesc
from mxnet.ndarray import zeros

from mxnet.module.base_module import BaseModule, _check_input_names, _parse_data_desc
from mxnet.module.module import Module


class DetModule(BaseModule):
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
    def __init__(self, symbol, data_names=None, label_names=None, logger=logging, context=ctx.cpu(),
                 fixed_param=None, excluded_param=None):
        super().__init__(logger=logger)

        if isinstance(context, ctx.Context):
            context = [context]
        self._context = context
        work_load_list = [1] * len(self._context)
        assert len(work_load_list) == len(self._context)
        self._work_load_list = work_load_list

        self._group2ctxs = None

        self._symbol = symbol

        data_names = list(data_names) if data_names is not None else []
        label_names = list(label_names) if label_names is not None else []
        state_names = []

        fixed_param_names = list()
        if fixed_param is not None:
            for name in self._symbol.list_arguments():
                for fixed_partial_name in fixed_param:
                    if fixed_partial_name in name:
                        fixed_param_names.append(name)
        if excluded_param is not None:
            for name in fixed_param_names.copy():
                for excluded_partial_name in excluded_param:
                    if excluded_partial_name in name:
                        fixed_param_names.remove(name)
        logger.info("fixed parameters: {}".format(fixed_param_names))

        _check_input_names(symbol, data_names, "data", True)
        _check_input_names(symbol, label_names, "label", False)
        _check_input_names(symbol, state_names, "state", True)
        _check_input_names(symbol, fixed_param_names, "fixed_param", True)

        arg_names = symbol.list_arguments()
        input_names = data_names + label_names + state_names
        self._param_names = [x for x in arg_names if x not in input_names]
        self._fixed_param_names = fixed_param_names
        self._aux_names = symbol.list_auxiliary_states()
        self._data_names = data_names
        self._label_names = label_names
        self._state_names = state_names
        self._output_names = symbol.list_outputs()

        self._arg_params = None
        self._aux_params = None
        self._params_dirty = False

        self._compression_params = None
        self._optimizer = None
        self._kvstore = None
        self._update_on_kvstore = None
        self._updater = None
        self._preload_opt_states = None
        self._grad_req = None

        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, **kwargs):
        """Creates a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : list of str
            Default is `('data')` for a typical model used in image classification.
        label_names : list of str
            Default is `('softmax_label')` for a typical model used in image
            classification.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is ``cpu()``.
        work_load_list : list of number
            Default ``None``, indicating uniform workload.
        fixed_param_names: list of str
            Default ``None``, indicating no network parameters are fixed.
        """
        sym, args, auxs = load_checkpoint(prefix, epoch)
        mod = DetModule(symbol=sym, **kwargs)
        mod._arg_params = args
        mod._aux_params = auxs
        mod.params_initialized = True
        if load_optimizer_states:
            mod._preload_opt_states = '%s-%04d.states'%(prefix, epoch)
        return mod

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Saves current progress to checkpoint.
        Use `mx.callback.module_checkpoint` as `epoch_end_callback` to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to.
        epoch : int
            The current epoch number.
        save_optimizer_states : bool
            Whether to save optimizer states to continue training.
        """
        self._symbol.save('%s-symbol.json'%prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        logging.info('Saved checkpoint to \"%s\"', param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self.save_optimizer_states(state_name)
            logging.info('Saved optimizer state to \"%s\"', state_name)

    def _reset_bind(self):
        """Internal function to reset binded state."""
        self.binded = False
        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        return self._data_names

    @property
    def label_names(self):
        """A list of names for labels required by this module."""
        return self._label_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        return self._output_names

    @property
    def data_shapes(self):
        """Gets data shapes.

        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._data_shapes

    @property
    def label_shapes(self):
        """Gets label shapes.

        Returns
        -------
        A list of `(name, shape)` pairs.
            The return value could be ``None`` if
            the module does not need labels, or if the module is not bound for
            training (in this case, label information is not available).
        """
        assert self.binded
        return self._label_shapes

    @property
    def output_shapes(self):
        """Gets output shapes.

        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._exec_group.get_output_shapes()

    def get_params(self):
        """Gets current parameters.

        Returns
        -------
        `(arg_params, aux_params)`
            A pair of dictionaries each mapping parameter names to NDArray values.
        """
        assert self.binded and self.params_initialized

        if self._params_dirty:
            self._sync_params_from_devices()
        return (self._arg_params, self._aux_params)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        """Initializes the parameters and auxiliary states.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not ``None``, should be a dictionary of existing arg_params. Initialization
            will be copied from that.
        aux_params : dict
            If not ``None``, should be a dictionary of existing aux_params. Initialization
            will be copied from that.
        allow_missing : bool
            If ``True``, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If ``True``, will force re-initialize even if already initialized.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        """
        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "init_params call ignored.", stacklevel=2)
            return
        assert self.binded, 'call bind before initializing the parameters'

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        cache_arr.copyto(arr)
                else:
                    if not allow_missing:
                        raise RuntimeError("%s is not presented" % name)
                    if initializer is not None:
                        initializer(name, arr)
            else:
                initializer(name, arr)

        attrs = self._symbol.attr_dict()
        for name, arr in sorted(self._arg_params.items()):
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, arg_params)

        for name, arr in sorted(self._aux_params.items()):
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, aux_params)

        self.params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec_group.set_params(self._arg_params, self._aux_params,
                                    allow_extra=allow_extra)

    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True,
                   allow_extra=False):
        """Assigns parameter and aux state values.

        Parameters
        ----------
        arg_params : dict
            Dictionary of name to `NDArray`.
        aux_params : dict
            Dictionary of name to `NDArray`.
        allow_missing : bool
            If ``True``, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If ``True``, will force re-initialize even if already initialized.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        Examples
        --------
        >>> # An example of setting module parameters.
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, n_epoch_load)
        >>> mod.set_params(arg_params=arg_params, aux_params=aux_params)
        """
        if not allow_missing:
            self.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params,
                             allow_missing=allow_missing, force_init=force_init,
                             allow_extra=allow_extra)
            return

        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "set_params call ignored.", stacklevel=2)
            return

        self._exec_group.set_params(arg_params, aux_params, allow_extra=allow_extra)

        # because we didn't update self._arg_params, they are dirty now.
        self._params_dirty = True
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        """Binds the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_data``.
        label_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_label``.
        for_training : bool
            Default is ``True``. Whether the executors should be bound for training.
        inputs_need_grad : bool
            Default is ``False``. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is ``False``. This function does nothing if the executors are already
            bound. But with this ``True``, the executors will be forced to rebind.
        shared_module : Module
            Default is ``None``. This is used in bucketing. When not ``None``, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already bound, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        self._grad_req = grad_req

        if not for_training:
            assert not inputs_need_grad
        else:
            pass
            # this is not True, as some module might not contains a loss function
            # that consumes the labels
            # assert label_shapes is not None

        self._data_shapes, self._label_shapes = _parse_data_desc(
            self.data_names, self.label_names, data_shapes, label_shapes)

        if shared_module is not None:
            assert isinstance(shared_module, Module) and \
                    shared_module.binded and shared_module.params_initialized
            shared_group = shared_module._exec_group
            assert len(shared_group.execs) >= len(self._context)
        else:
            shared_group = None

        self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
                                                     self._work_load_list, self._data_shapes,
                                                     self._label_shapes, self._param_names,
                                                     for_training, inputs_need_grad,
                                                     shared_group, logger=self.logger,
                                                     fixed_param_names=self._fixed_param_names,
                                                     grad_req=grad_req, group2ctxs=self._group2ctxs,
                                                     state_names=self._state_names)
        self._total_exec_bytes = self._exec_group._total_exec_bytes
        if shared_module is not None:
            self.params_initialized = True
            self._arg_params = shared_module._arg_params
            self._aux_params = shared_module._aux_params
        elif self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self._exec_group.set_params(self._arg_params, self._aux_params)
        else:
            assert self._arg_params is None and self._aux_params is None
            param_arrays = [
                zeros(shape=x[0].shape, dtype=x[0].dtype, stype=x[0].stype)
                for x in self._exec_group.param_arrays
            ]
            self._arg_params = {name:arr for name, arr in zip(self._param_names, param_arrays)}

            aux_arrays = [
                zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.aux_arrays
            ]
            self._aux_params = {name:arr for name, arr in zip(self._aux_names, aux_arrays)}

        if shared_module is not None and shared_module.optimizer_initialized:
            self.borrow_optimizer(shared_module)

    def reshape(self, data_shapes, label_shapes=None):
        """Reshapes the module for new input shapes.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_data``.
        label_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_label``.
        """
        assert self.binded
        self._data_shapes, self._label_shapes = _parse_data_desc(
            self.data_names, self.label_names, data_shapes, label_shapes)

        self._exec_group.reshape(self._data_shapes, self._label_shapes)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Installs and initializes optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        if self._params_dirty:
            self._sync_params_from_devices()

        (kvstore, update_on_kvstore) = \
                _create_kvstore(kvstore, len(self._context), self._arg_params)

        batch_size = self._exec_group.batch_size
        if kvstore and 'dist' in kvstore.type and '_sync' in kvstore.type:
            batch_size *= kvstore.num_workers
        rescale_grad = 1.0/batch_size

        if isinstance(optimizer, str):
            idx2name = {}
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i*len(self._context)+k: n
                                     for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = rescale_grad
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)
            if optimizer.rescale_grad != rescale_grad:
                #pylint: disable=no-member
                warnings.warn(
                    "Optimizer created manually outside Module but rescale_grad " +
                    "is not normalized to 1.0/batch_size/num_workers (%s vs. %s). "%(
                        optimizer.rescale_grad, rescale_grad) +
                    "Is this intended?", stacklevel=2)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if kvstore:
            if self._compression_params:
                kvstore.set_gradient_compression(self._compression_params)
            if update_on_kvstore:
                kvstore.set_optimizer(self._optimizer)
            # copy initialized local parameters to kvstore
            _initialize_kvstore(kvstore=kvstore,
                                param_arrays=self._exec_group.param_arrays,
                                arg_params=self._arg_params,
                                param_names=self._param_names,
                                update_on_kvstore=update_on_kvstore)

        if not update_on_kvstore:
            self._updater = opt.get_updater(optimizer)

        self.optimizer_initialized = True

        if self._preload_opt_states is not None:
            self.load_optimizer_states(self._preload_opt_states)
            self._preload_opt_states = None

    def borrow_optimizer(self, shared_module):
        """Borrows optimizer from a shared module. Used in bucketing, where exactly the same
        optimizer (esp. kvstore) is used.

        Parameters
        ----------
        shared_module : Module
        """
        assert shared_module.optimizer_initialized
        self._optimizer = shared_module._optimizer
        self._kvstore = shared_module._kvstore
        self._update_on_kvstore = shared_module._update_on_kvstore
        self._updater = shared_module._updater
        self.optimizer_initialized = True

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

            self.reshape(new_dshape, new_lshape)

        self._exec_group.forward(data_batch, is_train)

    def backward(self, out_grads=None):
        """Backward computation.

        See Also
        ----------
        :meth:`BaseModule.backward`.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert self.binded and self.params_initialized
        self._exec_group.backward(out_grads=out_grads)

    def update(self):
        """Updates parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.

        When KVStore is used to update parameters for multi-device or multi-machine training,
        a copy of the parameters are stored in KVStore. Note that for `row_sparse` parameters,
        this function does update the copy of parameters in KVStore, but doesn't broadcast the
        updated parameters to all devices / machines. Please call `prepare` to broadcast
        `row_sparse` parameters with the next batch of data.

        See Also
        ----------
        :meth:`BaseModule.update`.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        self._params_dirty = True
        if self._update_on_kvstore:
            _update_params_on_kvstore(self._exec_group.param_arrays,
                                      self._exec_group.grad_arrays,
                                      self._kvstore, self._exec_group.param_names)
        else:
            _update_params(self._exec_group.param_arrays,
                           self._exec_group.grad_arrays,
                           updater=self._updater,
                           num_device=len(self._context),
                           kvstore=self._kvstore,
                           param_names=self._exec_group.param_names)

    def get_outputs(self, merge_multi_context=True):
        """Gets outputs of the previous forward computation.

        If ``merge_multi_context`` is ``True``, it is like ``[out1, out2]``. Otherwise, it
        is like ``[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]``. All the output
        elements are `NDArray`. When `merge_multi_context` is `False`, those `NDArray`
        might live on different devices.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArray or list of list of NDArray
            Output.
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        """Gets the gradients with respect to the inputs of the module.

        If ``merge_multi_context`` is ``True``, it is like ``[grad1, grad2]``. Otherwise, it
        is like ``[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]``. All the output
        elements are `NDArray`.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArray or list of list of NDArray
              Input gradients
        """
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._exec_group.get_input_grads(merge_multi_context=merge_multi_context)

    def get_states(self, merge_multi_context=True):
        """Gets states from all devices.

        If `merge_multi_context` is ``True``, it is like ``[out1, out2]``. Otherwise, it
        is like ``[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]``. All the output
        elements are `NDArray`.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the states
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArray or list of list of NDArray
            States
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_states(merge_multi_context=merge_multi_context)

    def set_states(self, states=None, value=None):
        """Sets value for states. Only one of the states & value can be specified.

        Parameters
        ----------
        states : list of list of NDArrays
            source states arrays formatted like ``[[state1_dev1, state1_dev2],
            [state2_dev1, state2_dev2]]``.
        value : number
            a single scalar value for all state arrays.
        """
        assert self.binded and self.params_initialized
        self._exec_group.set_states(states, value)

    def update_metric(self, eval_metric, labels, pre_sliced=False):
        """Evaluates and accumulates evaluation metric on outputs of the last forward computation.

        See Also
        ----------
        :meth:`BaseModule.update_metric`.

        Parameters
        ----------
        eval_metric : EvalMetric
            Evaluation metric to use.
        labels : list of NDArray if `pre_sliced` parameter is set to `False`,
            list of lists of NDArray otherwise. Typically `data_batch.label`.
        pre_sliced: bool
            Whether the labels are already sliced per device (default: False).
        """
        if mxnet.__version__ >= "1.3.0":
            self._exec_group.update_metric(eval_metric, labels, pre_sliced)
        else:
            self._exec_group.update_metric(eval_metric, labels)

    def _sync_params_from_devices(self):
        """Synchronizes parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read the
        latest parameters from ``self._arg_params`` and ``self._aux_params``.

        For row_sparse parameters on devices, ther are pulled from KVStore with all row ids.

        """
        self._exec_group.get_params(self._arg_params, self._aux_params)
        if self._kvstore and self._update_on_kvstore:
            for param_name, param_val in sorted(self._arg_params.items()):
                if param_val.stype == 'row_sparse':
                    row_ids = nd.arange(0, param_val.shape[0], dtype='int64')
                    self._kvstore.row_sparse_pull(param_name, param_val, row_ids=row_ids)
        self._params_dirty = False

    def save_optimizer_states(self, fname):
        """Saves optimizer (updater) state to a file.

        Parameters
        ----------
        fname : str
            Path to output states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.save_optimizer_states(fname)
        else:
            with open(fname, 'wb') as fout:
                fout.write(self._updater.get_states())

    def load_optimizer_states(self, fname):
        """Loads optimizer (updater) state from a file.

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.load_optimizer_states(fname)
        else:
            self._updater.set_states(open(fname, 'rb').read())

    def install_monitor(self, mon):
        """Installs monitor on all executors. """
        assert self.binded
        self._exec_group.install_monitor(mon)

    def prepare(self, data_batch, sparse_row_id_fn=None):
        '''Prepares the module for processing a data batch.

        Usually involves switching bucket and reshaping.
        For modules that contain `row_sparse` parameters in KVStore,
        it prepares the `row_sparse` parameters based on the sparse_row_id_fn.

        When KVStore is used to update parameters for multi-device or multi-machine training,
        a copy of the parameters are stored in KVStore. Note that for `row_sparse` parameters,
        the `update()` updates the copy of parameters in KVStore, but doesn't broadcast
        the updated parameters to all devices / machines. The `prepare` function is used to
        broadcast `row_sparse` parameters with the next batch of data.

        Parameters
        ----------
        data_batch : DataBatch
            The current batch of data for forward computation.

        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        '''
        assert self.binded
        if sparse_row_id_fn is not None:
            if not self._kvstore or not self._update_on_kvstore:
                warnings.warn(UserWarning("Parameters are not updated in the KVStore. "
                                          "No need to call sparse_row_id_fn."))
            else:
                row_ids = sparse_row_id_fn(data_batch)
                assert(isinstance(row_ids, dict)), "Expected dict output from sparse_row_id_fn"
                for param_name, row_id in row_ids.items():
                    param_idx = self._exec_group.param_names.index(param_name)
                    param_val = self._exec_group.param_arrays[param_idx]
                    assert(isinstance(param_val, (tuple, list)))
                    if param_val[0].stype != 'row_sparse':
                        warnings.warn(UserWarning("%s.stype is not 'row_sparse'. No need to "
                                                  "perform row_sparse_pull." % param_name))
                    else:
                        self._kvstore.row_sparse_pull(param_name, param_val, row_ids=row_id,
                                                      priority=-param_idx)

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

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)

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
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
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

                if profile is True and nbatch == 10:
                    self.logger.info("Profiling ends")
                    import mxnet as mx
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
