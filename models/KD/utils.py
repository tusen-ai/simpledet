import pprint
import mxnet as mx

from core.detection_module import DetModule
from utils.load_model import load_checkpoint


def create_teacher_module(pTeacherModel, worker_data_shape, input_batch_size, ctx, rank, logger):
    t_prefix = pTeacherModel.prefix
    t_epoch = pTeacherModel.epoch
    t_endpoint = pTeacherModel.endpoint
    t_data_name = pTeacherModel.data_name
    t_label_name = pTeacherModel.label_name
    if rank == 0:
        logger.info('Building teacher module with endpoint: {}'.format(t_endpoint))
    t_sym = pTeacherModel.prefix + '-symbol.json'
    t_sym = mx.sym.load(t_sym)
    t_sym = mx.sym.Group([t_sym.get_internals()[out] for out in t_endpoint])
    t_worker_data_shape = {key: worker_data_shape[key] for key in t_data_name}
    _, t_out_shape, _ = t_sym.infer_shape(**t_worker_data_shape)
    t_terminal_out_shape_dict = zip(t_sym.list_outputs(), t_out_shape)
    t_data_shape = []
    for idx, data_name in enumerate(t_data_name):
        data_shape = t_worker_data_shape[data_name]
        data_shape = (input_batch_size,) + data_shape[1:]
        t_data_shape.append((data_name, data_shape))
    t_label_shape = []
    for idx, label_name in enumerate(t_label_name):
        label_shape = t_out_shape[idx]
        label_shape = (input_batch_size,) + label_shape[1:]
        t_label_shape.append((label_name, label_shape))
    if rank == 0:
        logger.info('Teacher data_name: {}'.format(t_data_name))
        logger.info('Teacher data_shape: {}'.format(t_data_shape))
        logger.info('Teacher label_name: {}'.format(t_label_name))
        logger.info('Teacher label_shape: {}'.format(t_label_shape))

    if rank == 0:
        logger.info('Teacher terminal output shape')
        logger.info(pprint.pformat([i for i in t_terminal_out_shape_dict]))
    t_arg_params, t_aux_params = load_checkpoint(t_prefix, t_epoch)
    t_mod = DetModule(t_sym, data_names=t_data_name, label_names=None,
                      logger=logger, context=ctx)
    t_mod.bind(data_shapes=t_data_shape, for_training=False, grad_req='null')
    t_mod.set_params(t_arg_params, t_aux_params)
    if rank == 0:
        logger.info('Finish teacher module build')
    return t_mod, t_label_name, t_label_shape