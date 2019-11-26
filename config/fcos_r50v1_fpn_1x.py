from symbol.builder import RPN as Detector
from models.FCOS.builder import MSRAResNet50V1FPN as Backbone
from models.FCOS.builder import FCOSFPNNeck as Neck
from models.FCOS.builder import FCOSFPNHead as RpnHead
from mxnext.complicate import normalizer_factory

INF=1e10
throwout_param = None


def get_config(is_train):
    class General:
        log_frequency = 20
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 2 if is_train else 1
        fp16 = False
        loader_worker = 4
        loader_collector = 2
        profile = False


    class KvstoreParam:
        kvstore     = "nccl"
        batch_image = General.batch_image
        gpus        = [0,1,2,3,4,5,6,7]
        fp16        = General.fp16


    class NormalizeParam:
        normalizer = normalizer_factory(type="fix")


    class BboxParam:
        pass


    class RoiParam:
        pass


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image

        class head:
            conv_channel = 256
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)
    
        class proposal:
            pre_nms_thresh = 0.05
            pre_nms_top_n =  1000
            post_nms_top_n = 1000
            fpn_box_max_n = 100
            nms_thr = 0.6
            min_bbox_side = 0

        class subsample_proposal:
            proposal_wo_gt = False
            image_roi = 512
            fg_fraction = 0.25
            fg_thr = 0.5
            bg_thr_hi = 0.5
            bg_thr_lo = 0.0

        class loss_setting:
            focal_loss_alpha = 0.25
            focal_loss_gamma = 2.0
            ignore_label = -1
            ignore_offset = -1

        class FCOSParam:
            num_classifier = 81 - 1			# COCO: 80 object + 1 background
            stride = (8, 16, 32, 64, 128)


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


    class FCOSFPNAssignParam:
        stages = [
                  [-1, 64],
                  [64, 128],
                  [128, 256],
                  [256, 512],
                  [512, INF],
                 ]
        stride = (8, 16, 32, 64, 128)
        num_classifier = 81 - 1				# COCO: 80 object + 1 background
        ignore_label = RpnParam.loss_setting.ignore_label
        ignore_offset = RpnParam.loss_setting.ignore_offset
        data_size = [PadParam.short, PadParam.long]


    class RenameParam:
        mapping = dict(image="data")

    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class NeckParam:
        fp16 = General.fp16
        normalizer = normalizer_factory(type="dummy")


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2014", "coco_valminusminival2014")
        else:
            image_set = ("coco_minival2014", )


    # throw out param used as custom op's input
    global throwout_param
    throwout_param = FCOSFPNAssignParam		# This line MUST be in front of rpn_head = RpnHead(RpnParam)


    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head)
        rpn_test_sym = None
        test_sym = None
    else:
        rpn_test_sym = detector.get_rpn_test_symbol(backbone, neck, rpn_head)
        train_sym = None


    class ModelParam:
        train_symbol = train_sym
        test_symbol = rpn_test_sym
        rpn_test_symbol = rpn_test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet-101"
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 16 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
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
            lr = 0.01 / 16 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3.0
            iter = 500


    class TestParam:
        min_det_score = 0
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: x

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            type = "nms"
            thr = 0.6

        class coco:
            annotation = "data/coco/annotations/instances_minival2014.json"


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, Norm2DImage

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
        data_name = ["data", "im_info"]
        label_name = ["gt_bbox"]
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

    import models.FCOS.metric as metric

    centerness_loss_metric = metric.LossMeter(RpnParam.FCOSParam.stride, pred_id_start=0, pred_id_end=1, name='centernessloss_meter')
    cls_loss_metric = metric.LossMeter(RpnParam.FCOSParam.stride, pred_id_start=1, pred_id_end=2, name='clsloss_meter')
    reg_loss_metric = metric.LossMeter(RpnParam.FCOSParam.stride, pred_id_start=2, pred_id_end=3, name='offsetloss_meter')

    metric_list = [centerness_loss_metric, cls_loss_metric, reg_loss_metric]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list
