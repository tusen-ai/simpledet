import os
import math
import pprint

from core.detection_module import DetModule
from core.detection_input import Loader
from utils.load_model import load_checkpoint
from utils.patch_config import patch_config_as_nothrow

from functools import reduce
from queue import Queue
from threading import Thread
import argparse
import importlib
import mxnet as mx
import numpy as np
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    # general
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--epoch', help='override test epoch specified by config', type=int, default=None)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config, args


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = os.environ.get("MXNET_CUDNN_AUTOTUNE_DEFAULT", "0")

    config, args = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    transform, data_name, label_name, metric_list = config.get_config(is_train=False)
    pGen = patch_config_as_nothrow(pGen)
    pKv = patch_config_as_nothrow(pKv)
    pRpn = patch_config_as_nothrow(pRpn)
    pRoi = patch_config_as_nothrow(pRoi)
    pBbox = patch_config_as_nothrow(pBbox)
    pDataset = patch_config_as_nothrow(pDataset)
    pModel = patch_config_as_nothrow(pModel)
    pOpt = patch_config_as_nothrow(pOpt)
    pTest = patch_config_as_nothrow(pTest)

    sym = pModel.test_symbol

    image_sets = pDataset.image_set
    roidbs_all = [pkl.load(open("data/cache/{}.roidb".format(i), "rb"), encoding="latin1") for i in image_sets]
    roidbs_all = reduce(lambda x, y: x + y, roidbs_all)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from utils.roidb_to_coco import roidb_to_coco
    if pTest.coco.annotation is not None:
        coco = COCO(pTest.coco.annotation)
    else:
        coco = roidb_to_coco(roidbs_all)

    data_queue = Queue(100)
    result_queue = Queue()

    execs = []
    workers = []
    coco_result = []
    split_size = 1000

    for index_split in range(int(math.ceil(len(roidbs_all) / split_size))):
        print("evaluating [%d, %d)" % (index_split * split_size, (index_split + 1) * split_size))
        roidb = roidbs_all[index_split * split_size:(index_split + 1) * split_size]
        roidb = pTest.process_roidb(roidb)
        for i, x in enumerate(roidb):
            x["rec_id"] = np.array(i, dtype=np.float32)
            x["im_id"] = np.array(x["im_id"], dtype=np.float32)

        loader = Loader(roidb=roidb,
                        transform=transform,
                        data_name=data_name,
                        label_name=label_name,
                        batch_size=1,
                        shuffle=False,
                        num_worker=4,
                        num_collector=2,
                        worker_queue_depth=2,
                        collector_queue_depth=2)

        print("total number of images: {}".format(loader.total_batch))

        data_names = [k[0] for k in loader.provide_data]

        if index_split == 0:
            arg_params, aux_params = load_checkpoint(pTest.model.prefix, args.epoch or pTest.model.epoch)
            if pModel.process_weight is not None:
                pModel.process_weight(sym, arg_params, aux_params)

            # infer shape
            worker_data_shape = dict(loader.provide_data + loader.provide_label)
            for key in worker_data_shape:
                worker_data_shape[key] = (pKv.batch_image,) + worker_data_shape[key][1:]
            arg_shape, _, aux_shape = sym.infer_shape(**worker_data_shape)
            _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
            out_shape_dict = list(zip(sym.get_internals().list_outputs(), out_shape))
            _, out_shape, _ = sym.infer_shape(**worker_data_shape)
            terminal_out_shape_dict = zip(sym.list_outputs(), out_shape)
            print('parameter shape')
            print(pprint.pformat([i for i in out_shape_dict if not i[0].endswith('output')]))
            print('intermediate output shape')
            print(pprint.pformat([i for i in out_shape_dict if i[0].endswith('output')]))
            print('terminal output shape')
            print(pprint.pformat([i for i in terminal_out_shape_dict]))

            '''
            there are some conflicts between `mergebn` and `attach_quantized_node` in graph_optimize.py
            when mergebn ahead of attach_quantized_node
            such as `Symbol.ComposeKeyword`
            '''
            if pModel.QuantizeTrainingParam is not None and pModel.QuantizeTrainingParam.quantize_flag:
                pQuant = pModel.QuantizeTrainingParam
                assert pGen.fp16 == False, "current quantize training only support fp32 mode."
                from utils.graph_optimize import attach_quantize_node
                _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
                out_shape_dictoinary = dict(zip(sym.get_internals().list_outputs(), out_shape))
                sym = attach_quantize_node(sym, out_shape_dictoinary, pQuant.WeightQuantizeParam,
                                        pQuant.ActQuantizeParam, pQuant.quantized_op)

            # merge batch normalization to speedup test
            from utils.graph_optimize import merge_bn
            sym, arg_params, aux_params = merge_bn(sym, arg_params, aux_params)
            sym.save(pTest.model.prefix + "_test.json")

            for i in pKv.gpus:
                ctx = mx.gpu(i)
                mod = DetModule(sym, data_names=data_names, context=ctx)
                mod.bind(data_shapes=loader.provide_data, for_training=False)
                mod.set_params(arg_params, aux_params, allow_extra=False)
                execs.append(mod)

        all_outputs = []

        if index_split == 0:
            def eval_worker(exe, data_queue, result_queue):
                while True:
                    batch = data_queue.get()
                    exe.forward(batch, is_train=False)
                    out = [x.asnumpy() for x in exe.get_outputs()]
                    result_queue.put(out)
            for exe in execs:
                workers.append(Thread(target=eval_worker, args=(exe, data_queue, result_queue)))
            for w in workers:
                w.daemon = True
                w.start()

        import time
        t1_s = time.time()

        def data_enqueue(loader, data_queue):
            for batch in loader:
                data_queue.put(batch)
        enqueue_worker = Thread(target=data_enqueue, args=(loader, data_queue))
        enqueue_worker.daemon = True
        enqueue_worker.start()

        for _ in range(loader.total_batch):
            r = result_queue.get()

            rid, id, info, cls, box = r
            rid, id, info, cls, box = rid.squeeze(), id.squeeze(), info.squeeze(), cls.squeeze(), box.squeeze()
            # TODO: POTENTIAL BUG, id or rid overflows float32(int23, 16.7M)
            id = np.asscalar(id)
            rid = np.asscalar(rid)

            scale = info[2]  # h_raw, w_raw, scale
            box = box / scale  # scale to original image scale
            cls = cls[:, 1:]   # remove background
            # TODO: the output shape of class_agnostic box is [n, 4], while class_aware box is [n, 4 * (1 + class)]
            box = box[:, 4:] if box.shape[1] != 4 else box

            output_record = dict(
                rec_id=rid,
                im_id=id,
                im_info=info,
                bbox_xyxy=box,  # ndarray (n, class * 4) or (n, 4)
                cls_score=cls   # ndarray (n, class)
            )

            all_outputs.append(output_record)

        t2_s = time.time()
        print("network uses: %.1f" % (t2_s - t1_s))

        # let user process all_outputs
        all_outputs = pTest.process_output(all_outputs, roidb)

        # aggregate results for ensemble and multi-scale test
        output_dict = {}
        for rec in all_outputs:
            im_id = rec["im_id"]
            if im_id not in output_dict:
                output_dict[im_id] = dict(
                    bbox_xyxy=[rec["bbox_xyxy"]],
                    cls_score=[rec["cls_score"]]
                )
            else:
                output_dict[im_id]["bbox_xyxy"].append(rec["bbox_xyxy"])
                output_dict[im_id]["cls_score"].append(rec["cls_score"])

        for k in output_dict:
            if len(output_dict[k]["bbox_xyxy"]) > 1:
                output_dict[k]["bbox_xyxy"] = np.concatenate(output_dict[k]["bbox_xyxy"])
            else:
                output_dict[k]["bbox_xyxy"] = output_dict[k]["bbox_xyxy"][0]

            if len(output_dict[k]["cls_score"]) > 1:
                output_dict[k]["cls_score"] = np.concatenate(output_dict[k]["cls_score"])
            else:
                output_dict[k]["cls_score"] = output_dict[k]["cls_score"][0]

        t3_s = time.time()
        print("aggregate uses: %.1f" % (t3_s - t2_s))


        if callable(pTest.nms.type):
            nms = pTest.nms.type(pTest.nms.thr)
        else:
            from operator_py.nms import py_nms_wrapper
            nms = py_nms_wrapper(pTest.nms.thr)

        def do_nms(k):
            bbox_xyxy = output_dict[k]["bbox_xyxy"]
            cls_score = output_dict[k]["cls_score"]
            final_dets = {}

            for cid in range(cls_score.shape[1]):
                score = cls_score[:, cid]
                if bbox_xyxy.shape[1] != 4:
                    cls_box = bbox_xyxy[:, cid * 4:(cid + 1) * 4]
                else:
                    cls_box = bbox_xyxy
                valid_inds = np.where(score > pTest.min_det_score)[0]
                box = cls_box[valid_inds]
                score = score[valid_inds]
                det = np.concatenate((box, score.reshape(-1, 1)), axis=1).astype(np.float32)
                det = nms(det)
                dataset_cid = coco.getCatIds()[cid]
                final_dets[dataset_cid] = det
            output_dict[k]["det_xyxys"] = final_dets
            del output_dict[k]["bbox_xyxy"]
            del output_dict[k]["cls_score"]
            return (k, output_dict[k])

        from multiprocessing import cpu_count
        from multiprocessing.pool import Pool
        pool = Pool(cpu_count() // 2)
        output_dict = pool.map(do_nms, output_dict.keys())
        output_dict = dict(output_dict)
        pool.close()

        t4_s = time.time()
        print("nms uses: %.1f" % (t4_s - t3_s))

        for iid in output_dict:
            result = []
            for cid in output_dict[iid]["det_xyxys"]:
                det = output_dict[iid]["det_xyxys"][cid]
                if det.shape[0] == 0:
                    continue
                scores = det[:, -1]
                xs = det[:, 0]
                ys = det[:, 1]
                ws = det[:, 2] - xs + 1
                hs = det[:, 3] - ys + 1
                result += [
                    {'image_id': int(iid),
                     'category_id': int(cid),
                     'bbox': [float(xs[k]), float(ys[k]), float(ws[k]), float(hs[k])],
                     'score': float(scores[k])}
                    for k in range(det.shape[0])
                ]
            result = sorted(result, key=lambda x: x['score'])[-pTest.max_det_per_image:]
            coco_result += result

        t5_s = time.time()
        print("convert to coco format uses: %.1f" % (t5_s - t4_s))

    import json
    json.dump(coco_result,
              open("experiments/{}/{}_result.json".format(pGen.name, pDataset.image_set[0]), "w"),
              sort_keys=True, indent=2)

    coco_dt = coco.loadRes(coco_result)
    coco_eval = COCOeval(coco, coco_dt)
    coco_eval.params.iouType = "bbox"
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    t6_s = time.time()
    print("coco eval uses: %.1f" % (t6_s - t5_s))
