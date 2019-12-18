import argparse
import importlib
import time
from utils.patch_config import patch_config_as_nothrow
from core.detection_module import DetModule

import mxnet as mx


def parse_args():
    parser = argparse.ArgumentParser(description='Test detector inference speed')
    # general
    parser.add_argument('--config', help='config file path', type=str, required=True)
    parser.add_argument('--shape', help='specify input 2d image shape', metavar=('SHORT', 'LONG'), type=int, nargs=2, required=True)
    parser.add_argument('--gpu', help='GPU index', type=int, default=0)
    parser.add_argument('--count', help='number of runs, final result will be averaged', type=int, default=100)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config, args.gpu, args.shape, args.count


if __name__ == "__main__":
    config, gpu, shape, count = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = config.get_config(is_train=False)
    sym = pModel.test_symbol

    # create dummy data batch
    data = mx.nd.ones(shape=[1, 3] + shape)
    im_info = mx.nd.array([x / 2.0 for x in shape] + [2.0]).reshape(1, 3)
    im_id = mx.nd.array([1])
    rec_id = mx.nd.array([1])
    data_names = ["data", "im_info", "im_id", "rec_id"]
    data_shape = [[1, 3] + shape, [1, 3], [1], [1]]
    data_shape = [(name, shape) for name, shape in zip(data_names, data_shape)]
    data_batch = mx.io.DataBatch(data=[data, im_info, im_id, rec_id])

    '''
    there are some conflicts between `mergebn` and `attach_quantized_node` in graph_optimize.py
    when mergebn ahead of attach_quantized_node
    such as `Symbol.ComposeKeyword`
    '''
    pModel = patch_config_as_nothrow(pModel)
    if pModel.QuantizeTrainingParam is not None and pModel.QuantizeTrainingParam.quantize_flag:
        pQuant = pModel.QuantizeTrainingParam
        assert pGen.fp16 == False, "current quantize training only support fp32 mode."
        from utils.graph_optimize import attach_quantize_node
        worker_data_shape = dict([(name, tuple(shape)) for name, shape in data_shape])
        # print(worker_data_shape)
        # raise NotImplementedError
        _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
        out_shape_dictoinary = dict(zip(sym.get_internals().list_outputs(), out_shape))
        sym = attach_quantize_node(sym, out_shape_dictoinary, pQuant.WeightQuantizeParam,
                                   pQuant.ActQuantizeParam, pQuant.quantized_op)
    sym.save(pTest.model.prefix + "_infer_speed.json")


    ctx = mx.gpu(gpu)
    mod = DetModule(sym, data_names=data_names, context=ctx)
    mod.bind(data_shapes=data_shape, for_training=False)
    mod.set_params({}, {}, True)

    # let AUTOTUNE run for once
    mod.forward(data_batch, is_train=False)
    for output in mod.get_outputs():
        output.wait_to_read()

    tic = time.time()
    for _ in range(count):
        mod.forward(data_batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
    toc = time.time()

    print((toc - tic) / count * 1000)

