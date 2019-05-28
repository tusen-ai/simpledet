import time

from core.detection_module import DetModule

import argparse
import importlib
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

