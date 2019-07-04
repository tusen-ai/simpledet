import argparse
import importlib
import logging
import os
import pprint
import pickle as pkl
from functools import reduce

from core.detection_module import DetModule
from utils import callback
from utils.memonger_v2 import search_plan_to_layer
from utils.lr_scheduler import WarmupMultiFactorScheduler, LRSequential, AdvancedLRScheduler
from utils.load_model import load_checkpoint
from utils.patch_config import patch_config_as_nothrow

import mxnet as mx
import numpy as np

def train_net(config):
    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    transform, data_name, label_name, metric_list = config.get_config(is_train=True)
    pGen = patch_config_as_nothrow(pGen)
    pKv = patch_config_as_nothrow(pKv)
    pRpn = patch_config_as_nothrow(pRpn)
    pRoi = patch_config_as_nothrow(pRoi)
    pBbox = patch_config_as_nothrow(pBbox)
    pDataset = patch_config_as_nothrow(pDataset)
    pModel = patch_config_as_nothrow(pModel)
    pOpt = patch_config_as_nothrow(pOpt)
    pTest = patch_config_as_nothrow(pTest)

    ctx = [mx.gpu(int(i)) for i in pKv.gpus]
    pretrain_prefix = pModel.pretrain.prefix
    pretrain_epoch = pModel.pretrain.epoch
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    begin_epoch = pOpt.schedule.begin_epoch
    end_epoch = pOpt.schedule.end_epoch
    lr_iter = pOpt.schedule.lr_iter

    # only rank==0 print all debug infos
    kvstore_type = "dist_sync" if os.environ.get("DMLC_ROLE") == "worker" else pKv.kvstore
    kv = mx.kvstore.create(kvstore_type)
    rank = kv.rank

    # for distributed training using shared file system
    os.makedirs(save_path, exist_ok=True)

    from utils.logger import config_logger
    config_logger(os.path.join(save_path, "log.txt"))

    model_prefix = os.path.join(save_path, "checkpoint")

    # set up logger
    logger = logging.getLogger()

    sym = pModel.train_symbol

    # setup multi-gpu
    input_batch_size = pKv.batch_image * len(ctx)

    # print config
    # if rank == 0:
    #     logger.info(pprint.pformat(config))

    # load dataset and prepare imdb for training
    image_sets = pDataset.image_set
    roidbs = [pkl.load(open("data/cache/{}.roidb".format(i), "rb"), encoding="latin1") for i in image_sets]
    roidb = reduce(lambda x, y: x + y, roidbs)
    # filter empty image
    roidb = [rec for rec in roidb if rec["gt_bbox"].shape[0] > 0]
    # add flip roi record
    flipped_roidb = []
    for rec in roidb:
        new_rec = rec.copy()
        new_rec["flipped"] = True
        flipped_roidb.append(new_rec)
    roidb = roidb + flipped_roidb

    from core.detection_input import AnchorLoader
    train_data = AnchorLoader(
        roidb=roidb,
        transform=transform,
        data_name=data_name,
        label_name=label_name,
        batch_size=input_batch_size,
        shuffle=True,
        kv=kv,
        num_worker=8,
        num_collector=4,
        worker_queue_depth=2,
        collector_queue_depth=2
    )

    # infer shape
    worker_data_shape = dict(train_data.provide_data + train_data.provide_label)
    for key in worker_data_shape:
        worker_data_shape[key] = (pKv.batch_image,) + worker_data_shape[key][1:]
    arg_shape, _, aux_shape = sym.infer_shape(**worker_data_shape)

    _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
    out_shape_dict = list(zip(sym.get_internals().list_outputs(), out_shape))

    _, out_shape, _ = sym.infer_shape(**worker_data_shape)
    terminal_out_shape_dict = zip(sym.list_outputs(), out_shape)

    if rank == 0:
        logger.info('parameter shape')
        logger.info(pprint.pformat([i for i in out_shape_dict if not i[0].endswith('output')]))

        logger.info('intermediate output shape')
        logger.info(pprint.pformat([i for i in out_shape_dict if i[0].endswith('output')]))

        logger.info('terminal output shape')
        logger.info(pprint.pformat([i for i in terminal_out_shape_dict]))

    # memonger
    if pModel.memonger:
        last_block = pModel.memonger_until or ""
        if rank == 0:
            logger.info("do memonger up to {}".format(last_block))

        type_dict = {k: np.float32 for k in worker_data_shape}
        sym = search_plan_to_layer(sym, last_block, 1000, type_dict=type_dict, **worker_data_shape)

    # load and initialize params
    if pOpt.schedule.begin_epoch != 0:
        arg_params, aux_params = load_checkpoint(model_prefix, begin_epoch)
    elif pModel.from_scratch:
        arg_params, aux_params = dict(), dict()
    else:
        arg_params, aux_params = load_checkpoint(pretrain_prefix, pretrain_epoch)

    if pModel.process_weight is not None:
        pModel.process_weight(sym, arg_params, aux_params)

    # merge batch normalization to save memory in fix bn training
    from utils.graph_optimize import merge_bn
    sym, arg_params, aux_params = merge_bn(sym, arg_params, aux_params)

    if pModel.random:
        import time
        mx.random.seed(int(time.time()))
        np.random.seed(int(time.time()))

    init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
    init.set_verbosity(verbose=True)

    # create solver
    fixed_param = pModel.pretrain.fixed_param
    excluded_param = pModel.pretrain.excluded_param
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]

    mod = DetModule(sym, data_names=data_names, label_names=label_names,
                    logger=logger, context=ctx, fixed_param=fixed_param, excluded_param=excluded_param)

    eval_metrics = mx.metric.CompositeEvalMetric(metric_list)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=pGen.log_frequency)
    epoch_end_callback = callback.do_checkpoint(model_prefix)
    sym.save(model_prefix + ".json")

    # decide learning rate
    lr_mode = pOpt.optimizer.lr_mode or 'step'
    base_lr = pOpt.optimizer.lr * kv.num_workers
    lr_factor = 0.1

    iter_per_epoch = len(train_data) // input_batch_size
    lr_iter = [it // kv.num_workers for it in lr_iter]
    lr_iter = [it - iter_per_epoch * begin_epoch for it in lr_iter]
    lr_iter_discount = [it for it in lr_iter if it > 0]
    current_lr = base_lr * (lr_factor ** (len(lr_iter) - len(lr_iter_discount)))
    if rank == 0:
        logging.info('total iter {}'.format(iter_per_epoch * (end_epoch - begin_epoch)))
        logging.info('lr {}, lr_iters {}'.format(current_lr, lr_iter_discount))
        logging.info('lr mode: {}'.format(lr_mode))

    if pOpt.warmup is not None and pOpt.schedule.begin_epoch == 0:
        if rank == 0:
            logging.info(
                'warmup lr {}, warmup step {}'.format(
                    pOpt.warmup.lr,
                    pOpt.warmup.iter // kv.num_workers)
                )
        if lr_mode == 'step':
            lr_scheduler = WarmupMultiFactorScheduler(
                step=lr_iter_discount,
                factor=lr_factor,
                warmup=True,
                warmup_type=pOpt.warmup.type,
                warmup_lr=pOpt.warmup.lr,
                warmup_step=pOpt.warmup.iter // kv.num_workers
            )
        elif lr_mode == 'cosine':
            warmup_lr_scheduler = AdvancedLRScheduler(
                mode='linear', 
                base_lr=pOpt.warmup.lr,
                target_lr=base_lr, 
                niters=pOpt.warmup.iter // kv.num_workers
            )
            cosine_lr_scheduler = AdvancedLRScheduler(
                mode='cosine', 
                base_lr=base_lr, 
                target_lr=0,
                niters=(iter_per_epoch * (end_epoch - begin_epoch) - pOpt.warmup.iter) // kv.num_workers
            )
            lr_scheduler = LRSequential([warmup_lr_scheduler, cosine_lr_scheduler])
        else:
            raise NotImplementedError
    else:
        if lr_mode == 'step':
            lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iter_discount, lr_factor)
        elif lr_mode == 'cosine':
            lr_scheduler = AdvancedLRScheduler(
                mode='cosine', 
                base_lr=base_lr, 
                target_lr=0,
                niters=iter_per_epoch * (end_epoch - begin_epoch) // kv.num_workers
            )
        else:
            lr_scheduler = None

    # optimizer
    optimizer_params = dict(
        momentum=pOpt.optimizer.momentum,
        wd=pOpt.optimizer.wd,
        learning_rate=current_lr,
        lr_scheduler=lr_scheduler,
        rescale_grad=1.0 / (len(pKv.gpus) * kv.num_workers),
        clip_gradient=pOpt.optimizer.clip_gradient
    )

    if pKv.fp16:
        optimizer_params['multi_precision'] = True
        optimizer_params['rescale_grad'] /= 128.0

    profile = pGen.profile or False
    if profile:
        mx.profiler.set_config(profile_all=True, filename=os.path.join(save_path, "profile.json"))

    # train
    mod.fit(
        train_data=train_data,
        eval_metric=eval_metrics,
        epoch_end_callback=epoch_end_callback,
        batch_end_callback=batch_end_callback,
        kvstore=kv,
        optimizer=pOpt.optimizer.type,
        optimizer_params=optimizer_params,
        initializer=init,
        allow_missing=True,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=begin_epoch,
        num_epoch=end_epoch,
        profile=profile
    )

    logging.info("Training has done")
    time.sleep(10)
    logging.info("Exiting")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Detection')
    parser.add_argument('--config', help='config file path', type=str)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config


if __name__ == '__main__':
    train_net(parse_args())
