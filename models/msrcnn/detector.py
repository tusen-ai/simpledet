import mxnext as X
import mxnet as mx

from symbol.detector import FasterRcnn, RpnHead
from models.FPN.builder import FPNRpnHead

from models.maskrcnn import bbox_post_processing
from models.msrcnn import maskiou_compute

from utils.patch_config import patch_config_as_nothrow


class MaskScoringRcnn(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, bbox_roi_extractor, mask_roi_extractor, bbox_head, mask_head, maskiou_head):
        gt_bbox = X.var("gt_bbox")
        gt_poly = X.var("gt_poly")
        im_info = X.var("im_info")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_head.get_anchor()
        rpn_loss = rpn_head.get_loss(rpn_feat, gt_bbox, im_info)
        proposal, bbox_cls, bbox_target, bbox_weight, mask_proposal, mask_target, mask_ind, mask_ratio = \
            rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, gt_poly, im_info)
        roi_feat = bbox_roi_extractor.get_roi_feature(rcnn_feat, proposal)
        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, mask_proposal)

        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)
        mask_loss, mask_pred_logits = mask_head.get_loss(mask_roi_feat, mask_target, mask_ind)

        iou_loss = maskiou_head.get_loss(mask_roi_feat, mask_pred_logits, mask_target, mask_ind, mask_ratio)
        return X.group(rpn_loss + bbox_loss + mask_loss + iou_loss)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head, bbox_roi_extractor, mask_roi_extractor, bbox_head, mask_head, bbox_post_processor, maskiou_head):
        rec_id, im_id, im_info, proposal, proposal_score = \
            MaskScoringRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        roi_feat = bbox_roi_extractor.get_roi_feature(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        post_cls_score, post_bbox_xyxy, post_cls = bbox_post_processor.get_post_processing(cls_score, bbox_xyxy)

        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, post_bbox_xyxy)
        mask = mask_head.get_prediction(mask_roi_feat)

        iou_pred = maskiou_head.get_maskiou_prediction(mask, mask_roi_feat, post_cls)

        return X.group([rec_id, im_id, im_info, post_cls_score, post_bbox_xyxy, post_cls, mask, iou_pred])

    @staticmethod
    def get_rpn_test_symbol(backbone, neck, rpn_head):
        return FasterRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)