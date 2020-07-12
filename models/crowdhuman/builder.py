from __future__ import print_function

import mxnet as mx
import mxnext as X

from symbol.builder import Backbone, RpnHead, BboxHead, Neck, RoiAlign, FasterRcnn
from models.FPN import assign_layer_fpn, get_top_proposal
from models.FPN.builder import FPNRpnHead, FPNBbox2fcHead
from models.crowdhuman import bbox_sec_target, bbox_target
from utils.patch_config import patch_config_as_nothrow

class DoublePredRcnn(FasterRcnn):
    _rpn_output = None

    def __init__(self):
        super().__init__()

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
        proposal, bbox_cls, bbox_target, bbox_weight, bbox_sec_cls, bbox_sec_target, bbox_sec_weight = rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, im_info)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight, bbox_sec_cls, bbox_sec_target, bbox_sec_weight)

        return X.group(rpn_loss + bbox_loss)
    
    @classmethod
    def get_test_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head):
        rec_id, im_id, im_info, proposal, proposal_score = \
            DoublePredRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        roi_feat = roi_extractor.get_roi_feature_test(rcnn_feat, proposal)
        cls_score, bbox_xyxy, cls_sec_score, bbox_sec_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)
        cls_score_concat = mx.sym.concat(*[cls_score, cls_sec_score], axis=1)
        bbox_xyxy_concat = mx.sym.concat(*[bbox_xyxy, bbox_sec_xyxy], axis=1)
        return X.group([rec_id, im_id, im_info, cls_score_concat, bbox_xyxy_concat])

class DoublePredBboxHead(object):
    def __init__(self, pBbox):
        self.p = patch_config_as_nothrow(pBbox)

        # declare weight and bias
        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        self.fc3_weight = X.var("bbox_fc3_weight", init=xavier_init)
        self.fc3_bias = X.var("bbox_fc3_bias")

        self._head_feat = None

    def _get_bbox_head_logit(self, conv_feat):
        raise NotImplementedError

    def get_output(self, conv_feat):
        p = self.p
        num_class = p.num_class
        num_reg_class = 2 if p.regress_target.class_agnostic else num_class

        head_feat = self._get_bbox_head_logit(conv_feat)

        if not isinstance(head_feat, dict):
            head_feat = dict(classification=head_feat, regression=head_feat)

        if p.fp16:
            head_feat["classification"] = X.to_fp32(head_feat["classification"], name="bbox_cls_head_to_fp32")
            head_feat["regression"] = X.to_fp32(head_feat["regression"], name="bbox_reg_head_to_fp32")

        cls_logit = X.fc(
            head_feat["classification"],
            filter=num_class,
            name='bbox_cls_logit1',
            init=X.gauss(0.01)
        )

        cls_sec_logit = X.fc(
            head_feat["classification"],
            filter=num_class,
            name='bbox_cls_logit2',
            init=X.gauss(0.01)
        )

        bbox_delta = X.fc(
            head_feat["regression"],
            filter=4 * num_reg_class,
            name='bbox_reg_delta1',
            init=X.gauss(0.001)
        )

        bbox_sec_delta = X.fc(
            head_feat["regression"],
            filter=4 * num_reg_class,
            name='bbox_reg_delta2',
            init=X.gauss(0.001)
        )

        return cls_logit, bbox_delta, cls_sec_logit, bbox_sec_delta, head_feat['regression'] # NOTE

    def get_prediction(self, conv_feat, im_info, proposal):
        p = self.p
        bbox_mean = p.regress_target.mean
        bbox_std = p.regress_target.std
        batch_image = p.batch_image
        num_class = p.num_class
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = 2 if class_agnostic else num_class

        cls_logit, bbox_delta, cls_sec_logit, bbox_sec_delta, roi_feature = self.get_output(conv_feat)
        if p.refine_mode:
            refine_cls_logit, refine_bbox_delta, refine_cls_sec_logit, refine_bbox_sec_delta = \
                                self.get_refine_output(roi_feature, cls_logit, bbox_delta, cls_sec_logit, bbox_sec_delta)
            bbox_delta = refine_bbox_delta; bbox_sec_delta = refine_bbox_sec_delta
            cls_logit = refine_cls_logit; cls_sec_logit = refine_cls_sec_logit

        bbox_delta = X.reshape(
            bbox_delta,
            shape=(batch_image, -1, 4 * num_reg_class),
            name='bbox_delta_reshape'
        )
        bbox_sec_delta = X.reshape(
            bbox_sec_delta,
            shape=(batch_image, -1, 4 * num_reg_class),
            name='bbox_sec_delta_reshape',
        )

        bbox_xyxy = X.decode_bbox(
            rois=proposal,
            bbox_pred=bbox_delta,
            im_info=im_info,
            name='decode_bbox',
            bbox_mean=bbox_mean,
            bbox_std=bbox_std,
            class_agnostic=class_agnostic
        )

        bbox_sec_xyxy = X.decode_bbox(
            rois=proposal,
            bbox_pred=bbox_sec_delta,
            im_info=im_info,
            name='decode_bbox_sec',
            bbox_mean=bbox_mean,
            bbox_std=bbox_std,
            class_agnostic=class_agnostic,
        )

        # NOTE: avoid name conflicts(potential bug from X.decode_bbox?)
        bbox_sec_xyxy = bbox_sec_xyxy * mx.sym.ones_like(bbox_sec_xyxy)

        cls_score = X.softmax(
            cls_logit,
            axis=-1,
            name='bbox_cls_score'
        )
        cls_score = X.reshape(
            cls_score,
            shape=(batch_image, -1, num_class),
            name='bbox_cls_score_reshape'
        )
        cls_sec_score = X.softmax(
            cls_sec_logit,
            axis=-1,
            name='bbox_cls_sec_score',
        )
        cls_sec_score = X.reshape(
            cls_sec_score,
            shape=(batch_image, -1, num_class),
            name='bbox_cls_sec_score_reshape'
        )
        return cls_score, bbox_xyxy, cls_sec_score, bbox_sec_xyxy

    def softmax_entropy(self, cls_logit, cls_label, prefix=""):
        from models.crowdhuman import softmax_entropy_op
        soft_ce = mx.sym.Custom(
            data=cls_logit,
            label=cls_label,
            op_type='softmax_entropy',
            name=prefix+'softmax_entropy'
        )
        return soft_ce

    def get_refine_output(self, roi_feature, cls_logit, bbox_delta, cls_sec_logit, bbox_sec_delta):
        p = self.p
        num_class = p.num_class
        repeat_time = p.repeat_time
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = 2 if class_agnostic else num_class

        cls_logit = mx.sym.slice_axis(mx.sym.softmax(cls_logit), axis=1, begin=1, end=num_class)
        cls_sec_logit = mx.sym.slice_axis(mx.sym.softmax(cls_sec_logit), axis=1, begin=1, end=num_class)
        bbox_delta = mx.sym.slice_axis(bbox_delta, axis=1, begin=4, end=num_reg_class * 4)
        bbox_sec_delta = mx.sym.slice_axis(bbox_sec_delta, axis=1, begin=4, end=num_reg_class * 4)

        pred_feat1 = mx.sym.tile(mx.sym.concat(*[bbox_delta, cls_logit], dim=1), reps=(1, repeat_time)) 
        pred_feat2 = mx.sym.tile(mx.sym.concat(*[bbox_sec_delta, cls_sec_logit], dim=1), reps=(1, repeat_time))

        refine_feat1 = mx.sym.concat(*[roi_feature, pred_feat1], dim=1)
        refine_feat2 = mx.sym.concat(*[roi_feature, pred_feat2], dim=1)

        head_feat1 = X.fc(
            refine_feat1,
            filter=1024,
            weight=self.fc3_weight,
            bias=self.fc3_bias,
            name='fc3_conv_refine1'
        )
        head_feat1 = X.relu(head_feat1)
        head_feat2 = X.fc(
            refine_feat2,
            filter=1024,
            weight=self.fc3_weight,
            bias=self.fc3_bias,
            name='fc3_conv_refine2'
        )
        head_feat2 = X.relu(head_feat2)
        refine_cls_logit = X.fc(
            head_feat1,
            filter=num_class,
            name='refine_bbox_cls_logit1',
            init=X.gauss(0.01)
        )

        refine_cls_sec_logit = X.fc(
            head_feat2,
            filter=num_class,
            name='refine_bbox_cls_logit2',
            init=X.gauss(0.01)
        )

        refine_bbox_delta = X.fc(
            head_feat1,
            filter=4 * num_reg_class,
            name='refine_bbox_reg_delta1',
            init=X.gauss(0.001)
        )

        refine_bbox_sec_delta = X.fc(
            head_feat2,
            filter=4 * num_reg_class,
            name='refine_bbox_reg_delta2',
            init=X.gauss(0.001)
        )
        return refine_cls_logit, refine_bbox_delta, refine_cls_sec_logit, refine_bbox_sec_delta

    def emd_loss(self, cls_logit, cls_label, cls_sec_logit, cls_sec_label, bbox_delta, bbox_target, 
                                bbox_sec_delta, bbox_sec_target, bbox_weight, bbox_sec_weight, prefix=""):
        p = self.p
        smooth_l1_scalar = p.regress_target.smooth_l1_scalar or 1.0
        scale_loss_shift = 128.0 if p.fp16 else 1.0
        cls_loss11 = self.softmax_entropy(cls_logit, cls_label, prefix=prefix+'cls_loss11')
        cls_loss12 = self.softmax_entropy(cls_sec_logit, cls_sec_label, prefix=prefix+'cls_loss12')
        cls_loss1 = cls_loss11 + cls_loss12

        cls_loss21 = self.softmax_entropy(cls_logit, cls_sec_label, prefix=prefix+'cls_loss21')
        cls_loss22 = self.softmax_entropy(cls_sec_logit, cls_label, prefix=prefix+'cls_loss22')
        cls_loss2 = cls_loss21 + cls_loss22

        # bounding box regression
        reg_loss11 = X.smooth_l1(
            bbox_delta - bbox_target,
            scalar=smooth_l1_scalar,
            name=prefix+'bbox_reg_l1_11'
        )
        reg_loss11 = bbox_weight * reg_loss11
        reg_loss12 = X.smooth_l1(
            bbox_sec_delta - bbox_sec_target,
            scalar=smooth_l1_scalar,
            name=prefix+'bbox_reg_l1_12'
        )
        reg_loss12 = bbox_sec_weight * reg_loss12
        reg_loss1 = reg_loss11 + reg_loss12

        reg_loss21 = X.smooth_l1(
            bbox_delta - bbox_sec_target,
            scalar=smooth_l1_scalar,
            name=prefix+'bbox_reg_l1_21'
        )
        reg_loss21 = bbox_sec_weight * reg_loss21
        reg_loss22 = X.smooth_l1(
            bbox_sec_delta - bbox_target,
            scalar=smooth_l1_scalar,
            name=prefix+'bbox_reg_l1_22'
        )
        reg_loss22 = bbox_weight * reg_loss22
        reg_loss2 = reg_loss21 + reg_loss22

        cls_reg_loss1 = mx.sym.sum(cls_loss1, axis=-1) + mx.sym.sum(reg_loss1, axis=-1)
        cls_reg_loss2 = mx.sym.sum(cls_loss2, axis=-1) + mx.sym.sum(reg_loss2, axis=-1)

        cls_reg_loss = mx.sym.minimum(cls_reg_loss1, cls_reg_loss2)
        cls_reg_loss = mx.sym.mean(cls_reg_loss)
        cls_reg_loss = X.loss(
            cls_reg_loss,
            grad_scale=1.0 * scale_loss_shift,
            name=prefix+'cls_reg_loss'
        )
        return cls_reg_loss

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight, cls_sec_label, bbox_sec_target, bbox_sec_weight):
        p = self.p
        refine_mode = p.refine_mode

        cls_logit, bbox_delta, cls_sec_logit, bbox_sec_delta, roi_feature = self.get_output(conv_feat)
        cls_reg_loss = self.emd_loss(cls_logit, cls_label, cls_sec_logit, cls_sec_label, bbox_delta, bbox_target,\
                                        bbox_sec_delta, bbox_sec_target, bbox_weight, bbox_sec_weight)
        if refine_mode == True:
            refine_cls_logit, refine_bbox_delta, refine_cls_sec_logit, refine_bbox_sec_delta = \
                                self.get_refine_output(roi_feature, cls_logit, bbox_delta, cls_sec_logit, bbox_sec_delta)
            refine_loss = self.emd_loss(refine_cls_logit, cls_label, refine_cls_sec_logit, cls_sec_label, refine_bbox_delta, bbox_target,
                                        refine_bbox_sec_delta, bbox_sec_target, bbox_weight, bbox_sec_weight, prefix="refine_")
            return cls_reg_loss, refine_loss
        # output
        return cls_reg_loss,

class DoublePredFPNBbox2fcHead(DoublePredBboxHead):
    def __init__(self, pBbox):
        super().__init__(pBbox)

    def add_norm(self, sym):
        p = self.p
        if p.normalizer.__name__ == "fix_bn":
            pass
        elif p.normalizer.__name__ in ["sync_bn", "gn"]:
            sym = p.normalizer(sym)
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))
        return sym
    
    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        flatten = X.reshape(conv_feat, shape=(0, -1), name="bbox_feat_reshape")
        fc1 = X.fc(flatten, filter=1024, name="bbox_fc1", init=xavier_init)
        fc1 = self.add_norm(fc1)
        fc1 = X.relu(fc1)
        fc2 = X.fc(fc1, filter=1024, name="bbox_fc2", init=xavier_init)
        fc2 = self.add_norm(fc2)
        fc2 = X.relu(fc2)

        self._head_feat = fc2

        return self._head_feat

class FPNRpnHeadwithIgnore(FPNRpnHead):
    def __init__(self, pRpn):
        super().__init__(pRpn)

    def get_sampled_proposal(self, conv_fpn_feat, gt_bbox, im_info):
        p = self.p

        batch_image = p.batch_image

        proposal_wo_gt = p.subsample_proposal.proposal_wo_gt
        image_roi = p.subsample_proposal.image_roi
        fg_fraction = p.subsample_proposal.fg_fraction
        fg_thr = p.subsample_proposal.fg_thr
        bg_thr_hi = p.subsample_proposal.bg_thr_hi
        bg_thr_lo = p.subsample_proposal.bg_thr_lo
        post_nms_top_n = p.proposal.post_nms_top_n

        num_reg_class = p.bbox_target.num_reg_class
        class_agnostic = p.bbox_target.class_agnostic
        bbox_target_weight = p.bbox_target.weight
        bbox_target_mean = p.bbox_target.mean
        bbox_target_std = p.bbox_target.std

        (proposal, proposal_score) = self.get_all_proposal(conv_fpn_feat, im_info)

        (bbox, label, bbox_target, bbox_weight) = mx.sym.Custom(
            proposal=proposal,
            gt_bbox=gt_bbox,
            num_class=num_reg_class,
            add_gt_to_proposal=not proposal_wo_gt,
            image_rois=image_roi,
            fg_fraction=fg_fraction,
            fg_thresh=fg_thr,
            bg_thresh_hi=bg_thr_hi,
            bg_thresh_lo=bg_thr_lo,
            bbox_target_std=bbox_target_std,
            name="subsample_proposal",
            op_type="bbox_target"
        )

        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))

        return bbox, label, bbox_target, bbox_weight

class DoublePredFPNRpnHead(FPNRpnHead):
    def __init__(self, pRpn):
        super().__init__(pRpn)

    def get_sampled_proposal(self, conv_fpn_feat, gt_bbox, im_info):
        p = self.p

        batch_image = p.batch_image

        proposal_wo_gt = p.subsample_proposal.proposal_wo_gt
        image_roi = p.subsample_proposal.image_roi
        fg_fraction = p.subsample_proposal.fg_fraction
        fg_thr = p.subsample_proposal.fg_thr
        bg_thr_hi = p.subsample_proposal.bg_thr_hi
        bg_thr_lo = p.subsample_proposal.bg_thr_lo
        post_nms_top_n = p.proposal.post_nms_top_n

        num_reg_class = p.bbox_target.num_reg_class
        class_agnostic = p.bbox_target.class_agnostic
        bbox_target_weight = p.bbox_target.weight
        bbox_target_mean = p.bbox_target.mean
        bbox_target_std = p.bbox_target.std

        (proposal, proposal_score) = self.get_all_proposal(conv_fpn_feat, im_info)

        (bbox, label, bbox_target, bbox_weight, sec_label, bbox_sec_target, bbox_sec_weight) = mx.sym.Custom(
            proposal=proposal,
            gt_bbox=gt_bbox,
            num_class=num_reg_class,
            add_gt_to_proposal=not proposal_wo_gt,
            image_rois=image_roi,
            fg_fraction=fg_fraction,
            fg_thresh=fg_thr,
            bg_thresh_hi=bg_thr_hi,
            bg_thresh_lo=bg_thr_lo,
            bbox_target_std=bbox_target_std,
            name="subsample_proposal",
            op_type="doublebbox_target"
        )

        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))

        sec_label = X.reshape(sec_label, (-3, -2))
        bbox_sec_target = X.reshape(bbox_sec_target, (-3, -2))
        bbox_sec_weight = X.reshape(bbox_sec_weight, (-3, -2))

        return bbox, label, bbox_target, bbox_weight, sec_label, bbox_sec_target, bbox_sec_weight