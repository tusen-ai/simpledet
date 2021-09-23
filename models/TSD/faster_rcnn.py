from __future__ import print_function

import mxnet as mx
import mxnext as X

class FasterRcnn_TSD(object):
    _rpn_output = None

    def __init__(self):
        pass
    
    @classmethod
    def get_train_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_head.get_anchor()
        rpn_loss = rpn_head.get_loss(rpn_feat, gt_bbox, im_info)
        proposal, bbox_cls, bbox_target, bbox_weight = rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, im_info)

        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(proposal, roi_feat, rcnn_feat, bbox_cls, bbox_target, bbox_weight, gt_bbox)

        return X.group(rpn_loss + bbox_loss)

    @classmethod
    def get_test_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head):
        rec_id, im_id, im_info, proposal, proposal_score = \
            FasterRcnn_TSD.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        roi_feat = roi_extractor.get_roi_feature_test(rcnn_feat, proposal)

        preds = bbox_head.get_prediction(proposal, roi_feat, rcnn_feat, im_info)

        return X.group([rec_id, im_id, im_info] + list(preds))

    @classmethod
    def get_rpn_test_symbol(cls, backbone, neck, rpn_head):
        if cls._rpn_output is not None:
            return cls._rpn_output

        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        rpn_head.get_anchor()
        rpn_feat = backbone.get_rpn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)

        (proposal, proposal_score) = rpn_head.get_all_proposal(rpn_feat, im_info)

        cls._rpn_output = X.group([rec_id, im_id, im_info, proposal, proposal_score])
        return cls._rpn_output




