from symbol.builder import FasterRcnn as Detector
from models.retinanet.builder import MSRAResNet50V1FPN as Backbone
from models.retinanet.builder import RetinaNetNeck as Neck
from models.aligndet.builder import AlignRetinaNetHead as RpnHead
from models.aligndet.builder import AlignRoiExtractor as RoiExtractor
from models.aligndet.builder import AlignHead as BboxHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 2 if is_train else 1
        fp16 = True


    class KvstoreParam:
        kvstore     = "local"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16


    class NormalizeParam:
        normalizer = normalizer_factory(type="fixbn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image
        num_class   = 1 + 80
        loss_weight = 1.0

        class anchor_generate:
            scale = (4 * 2 ** (1.0 / 3.0),)
            ratio = (1.0,)
            stride = (8, 16, 32, 64, 128)
            short = (100, 50, 25, 13, 7)
            long = (167, 84, 42, 21, 11)
            image_anchor = None

        class head:
            conv_channel = 256
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (1.0, 1.0, 1.0, 1.0)

        class focal_loss:
            alpha = 0.25
            gamma = 2.0

        class proposal:
            pre_nms_top_n = 1000
            min_bbox_side = 0
            min_det_score = 0.05

        class subsample_proposal:
            pass

        class bbox_target:
            class_agnostic = False
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (1.0, 1.0, 1.0, 1.0)
            allowed_border = 9999
            pos_thr = 0.7
            neg_thr = 0.7
            min_pos_thr = 0.0


    class BboxParam:
        fp16        = General.fp16
        normalizer  = NormalizeParam.normalizer
        num_class   = 1 + 80
        loss_weight = 1.0

        class anchor_generate:
            scale = (4 * 2 ** (1.0 / 3.0),)
            ratio = (1.0,)
            stride = (8, 16, 32, 64, 128)

        class head:
            merge_score = False
            num_conv = 2
            use_1x1 = True
            conv_channel = 1024
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (1.0, 1.0, 1.0, 1.0)

        class focal_loss:
            alpha = 0.25
            gamma = 2.0

        class proposal:
            pre_nms_top_n = 1000
            min_bbox_side = 0
            min_det_score = 0.05


    class RoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        sample_bins = 7
        im2col = True
        stride = (8, 16, 32, 64, 128)
        conv_channel = 256 * 7 * 7
        scale = (4 * 2 ** (1.0 / 3.0),)
        ratio = (1.0,)


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2014", "coco_valminusminival2014")
        else:
            image_set = ("coco_minival2014", )

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam)
    roi_extractor = RoiExtractor(RoiParam)
    bbox_head = BboxHead(BboxParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)
        rpn_test_sym = None
        test_sym = None
    else:
        train_sym = None
        rpn_test_sym = None
        test_sym = detector.get_test_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)


    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym
        rpn_test_symbol = rpn_test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet-v1-50"
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = None

        class schedule:
            begin_epoch = 0
            end_epoch = 6
            lr_iter = [60000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3
            iter = 500


    class TestParam:
        min_det_score = 0 # filter appended boxes
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: x

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
        mean = (122.7717, 115.9465, 102.9801) # RGB order
        std = (1.0, 1.0, 1.0)


    class ResizeParam:
        short = 800
        long = 1333


    class PadParam:
        short = 800
        long = 1333
        max_num_gt = 100


    class AnchorTarget2DParam:
        def __init__(self):
            self.generate = self._generate()
            self.mean = (0.0, 0.0, 0.0, 0.0)
            self.std = (1.0, 1.0, 1.0, 1.0)
            self.class_agnostic = False

        class _generate:
            def __init__(self):
                self.short = (100, 50, 25, 13, 7)
                self.long = (167, 84, 42, 21, 11)
                self.stride = (8, 16, 32, 64, 128)

            scales = (4 * 2 ** (1.0 / 3.0),)
            aspects = (1.0,)

        class assign:
            allowed_border = 9999
            pos_thr = 0.4
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = None
            pos_fraction = None


    class RenameParam:
        mapping = dict(image="data")


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord
    from models.retinanet.input import PyramidAnchorTarget2D, Norm2DImage

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            PyramidAnchorTarget2D(AnchorTarget2DParam()),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "gt_bbox"]
        label_name = ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
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

    from models.retinanet import metric

    rpn_acc_metric = metric.FGAccMetric(
        "FGAcc",
        ["cls_loss_output"],
        ["rpn_cls_label"]
    )

    bbox_acc_metric = metric.FGAccMetric(
        "BboxFGAcc",
        ["align_cls_loss_output", "align_label_blockgrad_output"],
        []
    )

    metric_list = [rpn_acc_metric, bbox_acc_metric]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list
