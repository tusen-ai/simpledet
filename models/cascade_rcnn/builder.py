from __future__ import division
from __future__ import print_function

import math
import mxnet as mx
import mxnext as X

from symbol.builder import FasterRcnn, Bbox2fcHead

class CascadeRcnn(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head, \
                         bbox_head_2nd, bbox_head_3rd):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_head.get_anchor()
        rpn_loss = rpn_head.get_loss(rpn_feat, gt_bbox, im_info)

        # stage1
        proposal, bbox_cls, bbox_target, bbox_weight = \
            rpn_head.get_sampled_proposal(
                rpn_feat,
                gt_bbox,
                im_info
            )
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(
            roi_feat,
            bbox_cls,
            bbox_target,
            bbox_weight
        )
        bbox_pred = bbox_head._bbox_delta

        # stage2
        # though call get_sampled_proposal, bbox_head does not sample rois
        proposal_2nd, bbox_cls_2nd, bbox_target_2nd, bbox_weight_2nd = \
            bbox_head.get_sampled_proposal(
                proposal,
                bbox_pred,
                gt_bbox,
                im_info
            )
        roi_feat_2nd = roi_extractor.get_roi_feature(rcnn_feat, proposal_2nd)
        bbox_loss_2nd = bbox_head_2nd.get_loss(
            roi_feat_2nd,
            bbox_cls_2nd,
            bbox_target_2nd,
            bbox_weight_2nd
        )
        bbox_pred_2nd = bbox_head_2nd._bbox_delta

        # stage3
        # though call get_sampled_proposal, bbox_head does not sample rois
        proposal_3rd, bbox_cls_3rd, bbox_target_3rd, bbox_weight_3rd = \
            bbox_head_2nd.get_sampled_proposal(
                proposal_2nd,
                bbox_pred_2nd,
                gt_bbox,
                im_info
            )
        roi_feat_3rd = roi_extractor.get_roi_feature(rcnn_feat, proposal_3rd)
        bbox_loss_3rd = bbox_head_3rd.get_loss(
            roi_feat_3rd,
            bbox_cls_3rd,
            bbox_target_3rd,
            bbox_weight_3rd
        )

        return X.group(rpn_loss + bbox_loss + bbox_loss_2nd + bbox_loss_3rd)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head, \
                        bbox_head_2nd, bbox_head_3rd):
        rec_id, im_id, im_info, proposal, proposal_score = \
            CascadeRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        # stage1
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        _, bbox_xyxy = bbox_head.get_prediction(
            roi_feat,
            im_info,
            proposal
        )

        # stage2
        proposal_2nd = bbox_xyxy
        roi_feat_2nd = roi_extractor.get_roi_feature(rcnn_feat, proposal_2nd)
        _, bbox_xyxy_2nd = bbox_head_2nd.get_prediction(
            roi_feat_2nd,
            im_info,
            proposal_2nd
        )

        # stage3
        proposal_3rd = bbox_xyxy_2nd
        roi_feat_3rd = roi_extractor.get_roi_feature(rcnn_feat, proposal_3rd)
        cls_score_3rd, bbox_xyxy_3rd = bbox_head_3rd.get_prediction(
            roi_feat_3rd,
            im_info,
            proposal_3rd
        )

        # passing feature from stage3 through stage1's weight
        bbox_head.stage = "1st_3rd"
        cls_score_1st_3rd, _ = bbox_head.get_prediction(
            roi_feat_3rd,
            im_info
        )
        # passing feature from stage3 through stage2's weight
        bbox_head_2nd.stage = "2nd_3rd"
        cls_score_2nd_3rd, _ = bbox_head_2nd.get_prediction(
            roi_feat_3rd,
            im_info
        )

        # average score between [1st_3rd, 2nd_3rd, 3rd]
        cls_score_avg = mx.sym.add_n(cls_score_1st_3rd, cls_score_2nd_3rd, cls_score_3rd) / 3

        return X.group([rec_id, im_id, im_info, cls_score_avg, bbox_xyxy_3rd])

    @staticmethod
    def get_rpn_test_symbol(backbone, neck, rpn_head):
        return FasterRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

    @staticmethod
    def get_refined_proposal(backbone, neck, rpn_head, roi_extractor, bbox_head, bbox_head_2nd, bbox_head_3rd, stage):
        rec_id, im_id, im_info, proposal, proposal_score = CascadeRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        # stage1
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        _, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        # stage2
        proposal_2nd = bbox_xyxy
        roi_feat_2nd = roi_extractor.get_roi_feature(rcnn_feat, proposal_2nd)
        _, bbox_xyxy_2nd = bbox_head_2nd.get_prediction(roi_feat_2nd, im_info, proposal_2nd)

        # stage3
        proposal_3rd = bbox_xyxy_2nd
        roi_feat_3rd = roi_extractor.get_roi_feature(rcnn_feat, proposal_3rd)
        cls_score_3rd, bbox_xyxy_3rd = bbox_head_3rd.get_prediction(roi_feat_3rd, im_info, proposal_3rd)

        # AR does not need score, just pass a dummy one
        if stage == 2:
            return X.group([rec_id, im_id, im_info, proposal_2nd, cls_score_3rd])
        elif stage == 3:
            return X.group([rec_id, im_id, im_info, proposal_3rd, cls_score_3rd])


"""
1. rename symbol via stage
2. (decode_bbox -> proposal_target) rather than (proposal -> proposal_target)
3. add bias for getting bbox head logit
"""
class CascadeBbox2fcHead(Bbox2fcHead):
    def __init__(self, pBbox):
        super().__init__(pBbox)

        self.stage          = pBbox.stage
        self._cls_logit     = None
        self._bbox_delta    = None
        self._proposal      = None

        # declare weight and bias
        stage = self.stage
        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        self.fc1_weight = X.var("bbox_fc1_%s_weight" % stage, init=xavier_init)
        self.fc2_weight = X.var("bbox_fc2_%s_weight" % stage, init=xavier_init)
        self.cls_logit_weight = X.var("bbox_cls_logit_%s_weight" % stage, init=X.gauss(0.01))
        self.bbox_delta_weight = X.var("bbox_reg_delta_%s_weight" % stage, init=X.gauss(0.001))

        self.fc1_bias = X.var("bbox_fc1_%s_bias" % stage)
        self.fc2_bias = X.var("bbox_fc2_%s_bias" % stage)
        self.cls_logit_bias = X.var("bbox_cls_logit_%s_bias" % stage)
        self.bbox_delta_bias = X.var("bbox_reg_delta_%s_bias" % stage)

    def _get_bbox_head_logit(self, conv_feat):
        # comment this for re-infer in test stage
        # if self._head_feat is not None:
        #     return self._head_feat

        p = self.p
        stage = self.stage

        flatten = X.flatten(conv_feat, name="bbox_feat_flatten_" + stage)
        reshape = X.reshape(flatten, (0, 0, 1, 1), name="bbox_feat_reshape_" + stage)

        if p.normalizer.__name__ == "fix_bn":
            fc1 = X.convrelu(
                reshape,
                filter=1024,
                weight=self.fc1_weight,
                bias=self.fc1_bias,
                no_bias=False,
                name="bbox_fc1_" + stage
            )
            fc2 = X.convrelu(
                fc1,
                filter=1024,
                weight=self.fc2_weight,
                bias=self.fc2_bias,
                no_bias=False,
                name="bbox_fc2_" + stage
            )
        elif p.normalizer.__name__ in ["sync_bn", "gn"]:
            fc1 = X.convnormrelu(
                p.normalizer,
                reshape,
                filter=1024,
                weight=self.fc1_weight,
                bias=self.fc1_bias,
                no_bias=False,
                name="bbox_fc1_" + stage
            )
            fc2 = X.convnormrelu(
                p.normalizer,
                fc1,
                filter=1024,
                weight=self.fc2_weight,
                bias=self.fc2_bias,
                no_bias=False,
                name="bbox_fc2_" + stage
            )
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))

        self._head_feat = fc2

        return self._head_feat

    def get_output(self, conv_feat):
        p = self.p
        stage = self.stage
        num_class = p.num_class
        num_reg_class = 2 if p.regress_target.class_agnostic else num_class

        head_feat = self._get_bbox_head_logit(conv_feat)

        if not isinstance(head_feat, dict):
            head_feat = dict(classification=head_feat, regression=head_feat)

        if p.fp16:
            head_feat["classification"] = X.to_fp32(head_feat["classification"], name="bbox_cls_head_to_fp32_" + stage)
            head_feat["regression"] = X.to_fp32(head_feat["regression"], name="bbox_reg_head_to_fp32_" + stage)

        cls_logit = X.fc(
            head_feat["classification"],
            filter=num_class,
            weight=self.cls_logit_weight,
            bias=self.cls_logit_bias,
            name='bbox_cls_logit_' + stage
        )
        bbox_delta = X.fc(
            head_feat["regression"],
            filter=4 * num_reg_class,
            weight=self.bbox_delta_weight,
            bias=self.bbox_delta_bias,
            name='bbox_reg_delta_' + stage
        )

        self._cls_logit = cls_logit
        self._bbox_delta = bbox_delta

        return cls_logit, bbox_delta

    def get_prediction(self, conv_feat, im_info, proposal=None):
        p = self.p
        stage = self.stage
        bbox_mean = p.regress_target.mean
        bbox_std = p.regress_target.std
        batch_image = p.batch_image
        num_class = p.num_class
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = 2 if class_agnostic else num_class

        cls_logit, bbox_delta = self.get_output(conv_feat)

        if proposal is None:
            bbox_xyxy = None
        else:
            bbox_delta = X.reshape(
                bbox_delta,
                shape=(batch_image, -1, 4 * num_reg_class),
                name='bbox_delta_reshape_' + stage
            )
            bbox_xyxy = X.decode_bbox(
                rois=proposal,
                bbox_pred=bbox_delta,
                im_info=im_info,
                name='decode_bbox_' + stage,
                bbox_mean=bbox_mean,
                bbox_std=bbox_std,
                class_agnostic=class_agnostic
            )

        cls_score = X.softmax(
            cls_logit,
            axis=-1,
            name='bbox_cls_score_' + stage
        )
        cls_score = X.reshape(
            cls_score,
            shape=(batch_image, -1, num_class),
            name='bbox_cls_score_reshape_' + stage
        )
        return cls_score, bbox_xyxy

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        stage = self.stage
        loss_weight = p.loss_weight
        batch_roi = p.image_roi * p.batch_image
        batch_image = p.batch_image

        cls_logit, bbox_delta = self.get_output(conv_feat)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # classification loss
        cls_loss = X.softmax_output(
            data=cls_logit,
            label=cls_label,
            normalization='batch',
            grad_scale=loss_weight * scale_loss_shift,
            name='bbox_cls_loss_' + stage
        )

        # bounding box regression
        reg_loss = X.smooth_l1(
            bbox_delta - bbox_target,
            scalar=1.0,
            name='bbox_reg_l1_' + stage
        )
        reg_loss = bbox_weight * reg_loss
        reg_loss = X.loss(
            reg_loss,
            grad_scale=loss_weight / batch_roi * scale_loss_shift,
            name='bbox_reg_loss_' + stage,
        )

        # append label
        cls_label = X.reshape(
            cls_label,
            shape=(batch_image, -1),
            name='bbox_label_reshape_' + stage
        )
        cls_label = X.block_grad(cls_label, name='bbox_label_blockgrad_' + stage)

        # output
        return cls_loss, reg_loss, cls_label

    def get_all_proposal(self, rois, bbox_pred, im_info):
        if self._proposal is not None:
            return self._proposal

        p = self.p
        stage = self.stage
        batch_image = p.batch_image
        bbox_mean = p.regress_target.mean
        bbox_std = p.regress_target.std
        num_class = p.num_class
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = 2 if class_agnostic else num_class

        bbox_pred = X.reshape(
            bbox_pred,
            shape=(batch_image, -1, 4 * num_reg_class),
            name='bbox_delta_reshape_' + stage
        )

        proposal = X.decode_bbox(
            rois=rois,
            bbox_pred=bbox_pred,
            im_info=im_info,
            name='decode_bbox_' + stage,
            bbox_mean=bbox_mean,
            bbox_std=bbox_std,
            class_agnostic=class_agnostic
        )

        # append None for dummy proposal score
        proposal = (proposal, None)

        self._proposal = proposal

        return proposal

    def get_sampled_proposal(self, rois, bbox_pred, gt_bbox, im_info):
        p = self.p
        stage = self.stage

        batch_image = p.batch_image

        proposal_wo_gt = p.subsample_proposal.proposal_wo_gt
        image_roi = p.subsample_proposal.image_roi
        fg_fraction = p.subsample_proposal.fg_fraction
        fg_thr = p.subsample_proposal.fg_thr
        bg_thr_hi = p.subsample_proposal.bg_thr_hi
        bg_thr_lo = p.subsample_proposal.bg_thr_lo

        num_reg_class = p.bbox_target.num_reg_class
        class_agnostic = p.bbox_target.class_agnostic
        bbox_target_weight = p.bbox_target.weight
        bbox_target_mean = p.bbox_target.mean
        bbox_target_std = p.bbox_target.std

        (proposal, proposal_score) = self.get_all_proposal(rois, bbox_pred, im_info)

        (bbox, label, bbox_target, bbox_weight) = X.proposal_target(
            rois=proposal,
            gt_boxes=gt_bbox,
            num_classes=num_reg_class,
            class_agnostic=class_agnostic,
            batch_images=batch_image,
            proposal_without_gt=proposal_wo_gt,
            image_rois=image_roi,
            fg_fraction=fg_fraction,
            fg_thresh=fg_thr,
            bg_thresh_hi=bg_thr_hi,
            bg_thresh_lo=bg_thr_lo,
            bbox_weight=bbox_target_weight,
            bbox_mean=bbox_target_mean,
            bbox_std=bbox_target_std,
            name="subsample_proposal_" + stage
        )

        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))

        return bbox, label, bbox_target, bbox_weight
