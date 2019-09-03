## Introduction

### Scripts
- `detection_train.py`
- `detection_test.py`
- `rpn_test.py`
- `mask_test.py`


### Top-level Singletons (Detectors)
Top-level singletons generally represent an unique kind of detection method (detector).
By unique we mean that the detector has a pipeline that can not be easily adapted from any existing detector.
Detectors now include
- RPN
- RetinaNet
- KD RetinaNet
- Fast R-CNN
- Faster R-CNN
- KD Faster R-CNN
- Mask R-CNN
- Cascade R-CNN

Every detector have a `get_train_symbol` method.
Each may have one or more of `get_bbox_test_symbol`, `get_mask_test_symbol`, `get_rpn_test_symbol`, and `get_kp_test_symbol` methods.

Here we leave the API design of `get_xxx_symbol` to user due to
- Detector design should not be bounded by the framework.
- The user is responsible for constructing the `train_sym` used in `detection_train.py` in the config.

We provide a detailed annotated Mask R-CNN as an example.
``` python
class MaskRcnn(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, bbox_roi_extractor, mask_roi_extractor, bbox_head, mask_head):
        # mask r-cnn needs ground truth bboxes and instance polygons to generate the target for training
        gt_bbox = X.var("gt_bbox")
        gt_poly = X.var("gt_poly")
        # im_info contains the width and height of image before padding and is use to remove anchors or
        # proposals out of image
        im_info = X.var("im_info")

        # backbone network provide feature map for sub-networks
        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        # neck is used for feature fusion across scales or dimension reduction
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        # cache anchor for generating anchor target and bbox target
        rpn_head.get_anchor()
        rpn_loss = rpn_head.get_loss(rpn_feat, gt_bbox, im_info)
        # calculate bbox_target, mask_target from rpn proposals
        proposal, bbox_cls, bbox_target, bbox_weight, mask_proposal, mask_target, mask_ind = \
            rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, gt_poly, im_info)
        roi_feat = bbox_roi_extractor.get_roi_feature(rcnn_feat, proposal)
        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, mask_proposal)

        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)
        mask_loss = mask_head.get_loss(mask_roi_feat, mask_target, mask_ind)
        return X.group(rpn_loss + bbox_loss + mask_loss)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head, bbox_roi_extractor, mask_roi_extractor, bbox_head, mask_head, bbox_post_processor):
        rec_id, im_id, im_info, proposal, proposal_score = \
            MaskRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        roi_feat = bbox_roi_extractor.get_roi_feature(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        # during test, in order to save computation, only the top 100 bbox after NMS are used for mask prediction
        post_cls_score, post_bbox_xyxy, post_cls = bbox_post_processor.get_post_processing(cls_score, bbox_xyxy)

        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, post_bbox_xyxy)
        mask = mask_head.get_prediction(mask_roi_feat)

        # the layout is fixed for mask_test.py
        return X.group([rec_id, im_id, im_info, post_cls_score, post_bbox_xyxy, post_cls, mask])

    @staticmethod
    def get_rpn_test_symbol(backbone, neck, rpn_head):
        return FasterRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)
```


### Components
Components now include
- Backbone
    - ResNet
    - ResNeXt
    - TridentNet
    - EfficientNet
    - DCNv1/v2
- Neck
    - ReduceNeck
    - FPNNeck
    - NASFPNNeck
- RpnHead
    - RpnHead
    - FPNRpnHead
    - MaskRpnHead
    - TridentRpnHead
    - RetinaNetHead
- RoIExtractor
    - RoIAlign
    - FPNRoIAlign
- BboxHead
    - BboxResNetv1C5Head
    - BboxResNetv2C5Head
    - BboxResNeXtC5Head
    - Bbox2fcHead
    - Bbox4conv1fcHead
    - BboxDualHead
- MaskHead
    - MaskHead
