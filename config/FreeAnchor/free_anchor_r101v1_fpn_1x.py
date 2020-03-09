from symbol.builder import add_anchor_to_arg
from models.FreeAnchor.builder import FreeAnchorRetinaNet as Detector
from models.retinanet.builder import MSRAResNet101V1FPN as Backbone
from models.retinanet.builder import RetinaNetNeck as Neck
from models.FreeAnchor.builder import FreeAnchorRetinaNetHead as RpnHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 2 if is_train else 1
        fp16 = False

    class KvstoreParam:
        kvstore = "nccl"
        batch_image = General.batch_image
        gpus = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16 = General.fp16

    class NormalizeParam:
        normalizer = normalizer_factory(type="fixbn")

    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer

    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer

    class RpnParam:
        num_class = 1 + 80
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image

        class anchor_generate:
            scale = (4 * 2 ** 0, 4 * 2 ** (1.0 / 3.0), 4 * 2 ** (2.0 / 3.0))
            ratio = (0.5, 1.0, 2.0)
            stride = (8, 16, 32, 64, 128)
            max_side = 1440

        class anchor_assign:
            allowed_border = 9999
            bbox_thr = 0.6
            pre_anchor_top_n = 50

        class head:
            conv_channel = 256
            mean = (.0, .0, .0, .0)
            std = (0.1, 0.1, 0.2, 0.2)

        class proposal:
            pre_nms_top_n = 1000
            post_nms_top_n = None
            nms_thr = None
            min_bbox_side = None

        class subsample_proposal:
            proposal_wo_gt = None
            image_roi = None
            fg_fraction = None
            fg_thr = None
            bg_thr_hi = None
            bg_thr_lo = None

        class bbox_target:
            num_reg_class = None
            class_agnostic = None
            weight = None
            mean = None
            std = None

        class focal_loss:
            alpha = 0.5
            gamma = 2.0

    class BboxParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        num_class = None
        image_roi = None
        batch_image = None

        class regress_target:
            class_agnostic = None
            mean = None
            std = None

    class RoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        out_size = None
        stride = None

    class DatasetParam:
        if is_train:
            image_set = ("coco_train2017", )
        else:
            image_set = ("coco_val2017", )

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head)
        test_sym = None
    else:
        train_sym = None
        test_sym = detector.get_test_symbol(backbone, neck, rpn_head)

    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet-v1-101"
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]

        def process_weight(sym, arg, aux):
            for stride in RpnParam.anchor_generate.stride:
                add_anchor_to_arg(
                    sym, arg, aux, RpnParam.anchor_generate.max_side,
                    stride, RpnParam.anchor_generate.scale,
                    RpnParam.anchor_generate.ratio)

    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = 35

        class schedule:
            begin_epoch = 0
            end_epoch = 6
            lr_iter = [60000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3
            iter = 1000

    class TestParam:
        min_det_score = 0.05  # filter appended boxes
        max_det_per_image = 100

        def process_roidb(x): return x  # noqa: E704

        def process_output(x, y): return x  # noqa: E704

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            type = "nms"
            thr = 0.5

        class coco:
            annotation = "data/coco/annotations/instances_minival2014.json"

    # data processing
    class NormParam:
        mean = (122.7717, 115.9465, 102.9801)  # RGB order
        std = (1.0, 1.0, 1.0)

    class ResizeParam:
        short = 800
        long = 1333

    class PadParam:
        short = 800
        long = 1333
        max_num_gt = 100

    class RenameParam:
        mapping = dict(image="data")

    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord
    from models.retinanet.input import Norm2DImage

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data"]
        label_name = ["gt_bbox", "im_info"]
    else:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "im_id", "rec_id"]
        label_name = []

    import core.detection_metric as metric
    pos_loss = metric.ScalarLoss(
        "PosLoss",
        ["positive_loss_output"],
        []
    )
    neg_loss = metric.ScalarLoss(
        "NegLoss",
        ["negative_loss_output"],
        []
    )
    metric_list = [pos_loss, neg_loss]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
        ModelParam, OptimizeParam, TestParam, \
        transform, data_name, label_name, metric_list
