from symbol.builder import add_anchor_to_arg
from symbol.builder import ResNetV1bFPN as Backbone
from models.FPN.builder import FPNNeck as Neck
from models.FPN.builder import FPNRoiAlign as RoiExtractor
from models.FPN.builder import FPNBbox2fcHead as BboxHead
from mxnext.complicate import normalizer_factory

from models.maskrcnn.builder import MaskFasterRcnn as Detector
from models.maskrcnn.builder import MaskFPNRpnHead as RpnHead
from models.maskrcnn.builder import MaskFasterRcnn4ConvHead as MaskHead
from models.maskrcnn.builder import BboxPostProcessor
from models.maskrcnn.process_output import process_output


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 2 if is_train else 1
        fp16 = False
        loader_worker = 8


    class KvstoreParam:
        kvstore     = "nccl"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16


    class NormalizeParam:
        normalizer = normalizer_factory(type="fixbn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        depth = 101


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image
        nnvm_proposal = True
        nnvm_rpn_target = False

        class anchor_generate:
            scale = (8,)
            ratio = (0.5, 1.0, 2.0)
            stride = (4, 8, 16, 32, 64)
            image_anchor = 256
            max_side = 1400

        class anchor_assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0
            image_anchor = 256
            pos_fraction = 0.5

        class head:
            conv_channel = 256
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)

        class proposal:
            pre_nms_top_n = 2000 if is_train else 1000
            post_nms_top_n = 2000 if is_train else 1000
            nms_thr = 0.7
            min_bbox_side = 0

        class subsample_proposal:
            proposal_wo_gt = False
            image_roi = 512
            fg_fraction = 0.25
            fg_thr = 0.5
            bg_thr_hi = 0.5
            bg_thr_lo = 0.0

        class bbox_target:
            num_reg_class = 81
            class_agnostic = False
            weight = (1.0, 1.0, 1.0, 1.0)
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class BboxParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        num_class   = 1 + 80
        image_roi   = 512
        batch_image = General.batch_image

        class regress_target:
            class_agnostic = False
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class MaskParam:
        fp16        = General.fp16
        normalizer  = NormalizeParam.normalizer
        resolution  = 28
        dim_reduced = 256
        num_fg_roi  = int(RpnParam.subsample_proposal.image_roi * RpnParam.subsample_proposal.fg_fraction)


    class RoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        out_size = 7
        stride = (4, 8, 16, 32)
        roi_canonical_scale = 224
        roi_canonical_level = 4


    class MaskRoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        out_size = 14
        stride = (4, 8, 16, 32)
        roi_canonical_scale = 224
        roi_canonical_level = 4


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2014", "coco_valminusminival2014")
        else:
            image_set = ("coco_minival2014", )


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = None

        class schedule:
            mult = 2
            begin_epoch = 0
            end_epoch = 6 * mult
            lr_iter = [60000 * mult * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80000 * mult * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3.0
            iter = 500


    class TestParam:
        min_det_score = 0.05
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: process_output(x, y)

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            type = "nms"
            thr = 0.5

        class coco:
            annotation = "data/coco/annotations/instances_minival2014.json"

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam, MaskParam)
    roi_extractor = RoiExtractor(RoiParam)
    mask_roi_extractor = RoiExtractor(MaskRoiParam)
    bbox_head = BboxHead(BboxParam)
    mask_head = MaskHead(BboxParam, MaskParam, MaskRoiParam)
    bbox_post_processer = BboxPostProcessor(TestParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head, roi_extractor, mask_roi_extractor, bbox_head, mask_head)
        test_sym = None
    else:
        train_sym = None
        test_sym = detector.get_test_symbol(backbone, neck, rpn_head, roi_extractor, mask_roi_extractor, bbox_head, mask_head, bbox_post_processer)


    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet%s_v1b" % BackboneParam.depth
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]
            excluded_param = ["mask_fcn"]

        def process_weight(sym, arg, aux):
            for stride in RpnParam.anchor_generate.stride:
                add_anchor_to_arg(
                    sym, arg, aux, RpnParam.anchor_generate.max_side,
                    stride, RpnParam.anchor_generate.scale,
                    RpnParam.anchor_generate.ratio)


    # data processing
    class NormParam:
        mean = tuple(i * 255 for i in (0.485, 0.456, 0.406)) # RGB order
        std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

    # data processing
    class ResizeParam:
        short = 800
        long = 1333


    class PadParam:
        short = 800
        long = 1333
        max_num_gt = 100
        max_len_gt_poly = 2500


    class AnchorTarget2DParam:
        def __init__(self):
            self.generate = self._generate()

        class _generate:
            def __init__(self):
                self.stride = (4, 8, 16, 32, 64)
                self.short = (200, 100, 50, 25, 13)
                self.long = (334, 167, 84, 42, 21)
            scales = (8)
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5


    class RenameParam:
        mapping = dict(image="data")


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, Norm2DImage

    from models.maskrcnn.input import PreprocessGtPoly, EncodeGtPoly, \
        Resize2DImageBboxMask, Flip2DImageBboxMask, Pad2DImageBboxMask

    from models.FPN.input import PyramidAnchorTarget2D

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            PreprocessGtPoly(),
            Resize2DImageBboxMask(ResizeParam),
            Flip2DImageBboxMask(),
            EncodeGtPoly(PadParam),
            Pad2DImageBboxMask(PadParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data"]
        label_name = ["im_info", "gt_bbox", "gt_poly"]
        if not RpnParam.nnvm_rpn_target:
            transform.append(PyramidAnchorTarget2D(AnchorTarget2DParam()))
            label_name += ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
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
    from models.maskrcnn.metric import SigmoidCELossMetric

    rpn_acc_metric = metric.AccWithIgnore(
        "RpnAcc",
        ["rpn_cls_loss_output", "rpn_cls_label_blockgrad_output"],
        []
    )
    rpn_l1_metric = metric.L1(
        "RpnL1",
        ["rpn_reg_loss_output", "rpn_cls_label_blockgrad_output"],
        []
    )
    # for bbox, the label is generated in network so it is an output
    box_acc_metric = metric.AccWithIgnore(
        "RcnnAcc",
        ["bbox_cls_loss_output", "bbox_label_blockgrad_output"],
        []
    )
    box_l1_metric = metric.L1(
        "RcnnL1",
        ["bbox_reg_loss_output", "bbox_label_blockgrad_output"],
        []
    )
    mask_cls_metric = SigmoidCELossMetric(
        "MaskCE",
        ["mask_loss_output"],
        []
    )

    metric_list = [rpn_acc_metric, rpn_l1_metric, box_acc_metric, box_l1_metric,]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list
