from models.RepPoints.builder import RepPoints as Detector
from models.dcn.builder import DCNResNetFPN as Backbone
from models.RepPoints.builder import RepPointsNeck as Neck
from models.RepPoints.builder import RepPointsHead as Head
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
        # normalizer = normalizer_factory(type="syncbn", ndev=8, wd_mult=1.0)
        normalizer = normalizer_factory(type="gn")

    class BackboneParam:
        fp16 = General.fp16
        # normalizer = NormalizeParam.normalizer
        normalizer = normalizer_factory(type="fixbn")
        depth = 101
        num_c3_block = 0
        num_c4_block = 3

    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer

    class HeadParam:
        num_class = 1 + 80
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image

        class point_generate:
            num_points = 9
            scale = 4
            stride = (8, 16, 32, 64, 128)
            transform = "moment"

        class head:
            conv_channel = 256
            point_conv_channel = 256
            mean = None
            std = None

        class proposal:
            pre_nms_top_n = 1000
            post_nms_top_n = None
            nms_thr = None
            min_bbox_side = None

        class point_target:
            target_scale = 4
            num_pos = 1

        class bbox_target:
            pos_iou_thr = 0.5
            neg_iou_thr = 0.5
            min_pos_iou = 0.0

        class focal_loss:
            alpha = 0.25
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
    head = Head(HeadParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, head)
        test_sym = None
    else:
        train_sym = None
        test_sym = detector.get_test_symbol(backbone, neck, head)

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
            excluded_param = ["gn"]

    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = 35

        class schedule:
            begin_epoch = 0
            end_epoch = 12
            lr_iter = [120000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       160000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3
            iter = 2000

    class TestScaleParam:
        short_ranges = [600, 800, 1000, 1200]
        long_ranges = [2000, 2000, 2000, 2000]

        @staticmethod
        def add_resize_info(roidb):
            ms_roidb = []
            for r_ in roidb:
                for short, long in zip(TestScaleParam.short_ranges, TestScaleParam.long_ranges):
                    r = r_.copy()
                    r["resize_long"] = long
                    r["resize_short"] = short
                    ms_roidb.append(r)

            return ms_roidb

    class TestParam:
        min_det_score = 0.05  # filter appended boxes
        max_det_per_image = 100

        process_roidb = TestScaleParam.add_resize_info

        def process_output(x, y):
            return x

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
        mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))  # RGB order
        std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

    class RandResizeParam:
        short = None  # generate on the fly
        long = None
        short_ranges = [600, 800, 1000, 1200]
        long_ranges = [2000, 2000, 2000, 2000]

    class RandCropParam:
        mode = "center"  # random or center
        short = 800
        long = 1333

    class ResizeParam:
        short = 800
        long = 1333

    class PadParam:
        short = 800
        long = 1333
        max_num_gt = 100

    class RandPadParam:
        short = 1200
        long = 2000
        max_num_gt = 100

    class RenameParam:
        mapping = dict(image="data")

    from core.detection_input import ReadRoiRecord, \
        RandResize2DImageBbox, RandCrop2DImageBbox, Resize2DImageByRoidb, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord
    from models.retinanet.input import Norm2DImage

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            # Resize2DImageBbox(ResizeParam),
            RandResize2DImageBbox(RandResizeParam),
            RandCrop2DImageBbox(RandCropParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data"]
        label_name = ["gt_bbox"]
    else:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            # Resize2DImageBbox(ResizeParam),
            Resize2DImageByRoidb(),
            Pad2DImageBbox(RandPadParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "im_id", "rec_id"]
        label_name = []

    from models.retinanet import metric as cls_metric
    import core.detection_metric as box_metric

    cls_acc_metric = cls_metric.FGAccMetric(
        "FGAcc",
        ["cls_loss_output", "point_refine_labels_output"],
        []
    )
    box_init_l1_metric = box_metric.L1(
        "InitL1",
        ["pts_init_loss_output", "points_init_labels_output"],
        []
    )
    box_refine_l1_metric = box_metric.L1(
        "RefineL1",
        ["pts_refine_loss_output", "point_refine_labels_output"],
        []
    )

    metric_list = [cls_acc_metric, box_init_l1_metric, box_refine_l1_metric]

    return General, KvstoreParam, HeadParam, RoiParam, BboxParam, DatasetParam, \
        ModelParam, OptimizeParam, TestParam, \
        transform, data_name, label_name, metric_list
