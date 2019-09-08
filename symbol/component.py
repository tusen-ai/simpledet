from __future__ import print_function

import mxnet as mx
import mxnext as X
from utils.patch_config import patch_config_as_nothrow


class RpnHead(object):
    def __init__(self, pRpn):
        self.p = patch_config_as_nothrow(pRpn)

        self._cls_logit             = None
        self._bbox_delta            = None
        self._proposal              = None

    def get_anchor(self):
        pass

    def get_output(self, conv_feat):
        if self._cls_logit is not None and self._bbox_delta is not None:
            return self._cls_logit, self._bbox_delta

        p = self.p
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        conv_channel = p.head.conv_channel

        if p.normalizer.__name__ == "fix_bn":
            conv = X.convrelu(
                conv_feat,
                kernel=3,
                filter=conv_channel,
                name="rpn_conv_3x3",
                no_bias=False,
                init=X.gauss(0.01)
            )
        elif p.normalizer.__name__ in ["sync_bn", "gn"]:
            conv = X.convnormrelu(
                p.normalizer,
                conv_feat,
                kernel=3,
                filter=conv_channel,
                name="rpn_conv_3x3",
                no_bias=False,
                init=X.gauss(0.01)
            )
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))

        if p.fp16:
            conv = X.to_fp32(conv, name="rpn_conv_3x3_fp32")

        cls_logit = X.conv(
            conv,
            filter=2 * num_base_anchor,
            name="rpn_cls_logit",
            no_bias=False,
            init=X.gauss(0.01)
        )

        bbox_delta = X.conv(
            conv,
            filter=4 * num_base_anchor,
            name="rpn_bbox_delta",
            no_bias=False,
            init=X.gauss(0.01)
        )

        self._cls_logit = cls_logit
        self._bbox_delta = bbox_delta

        return self._cls_logit, self._bbox_delta

    def get_loss(self, conv_feat, gt_bboxes, im_infos):
        p = self.p
        batch_image = p.batch_image
        image_anchor = p.anchor_generate.image_anchor

        cls_logit, bbox_delta = self.get_output(conv_feat)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        cls_label = X.var("rpn_cls_label")
        bbox_target = X.var("rpn_reg_target")
        bbox_weight = X.var("rpn_reg_weight")

        # classification loss
        cls_logit_reshape = X.reshape(
            cls_logit,
            shape=(0, -4, 2, -1, 0, 0),  # (N,C,H,W) -> (N,2,C/2,H,W)
            name="rpn_cls_logit_reshape"
        )
        cls_loss = X.softmax_output(
            data=cls_logit_reshape,
            label=cls_label,
            multi_output=True,
            normalization='valid',
            use_ignore=True,
            ignore_label=-1,
            grad_scale=1.0 * scale_loss_shift,
            name="rpn_cls_loss"
        )

        # regression loss
        reg_loss = X.smooth_l1(
            (bbox_delta - bbox_target),
            scalar=3.0,
            name='rpn_reg_l1'
        )
        reg_loss = bbox_weight * reg_loss
        reg_loss = X.loss(
            reg_loss,
            grad_scale=1.0 / (batch_image * image_anchor) * scale_loss_shift,
            name='rpn_reg_loss'
        )

        return cls_loss, reg_loss

    def get_all_proposal(self, conv_feat, im_info):
        if self._proposal is not None:
            return self._proposal

        p = self.p
        rpn_stride = p.anchor_generate.stride
        anchor_scale = p.anchor_generate.scale
        anchor_ratio = p.anchor_generate.ratio
        pre_nms_top_n = p.proposal.pre_nms_top_n
        post_nms_top_n = p.proposal.post_nms_top_n
        nms_thr = p.proposal.nms_thr
        min_bbox_side = p.proposal.min_bbox_side

        cls_logit, bbox_delta = self.get_output(conv_feat)

        # TODO: remove this reshape hell
        cls_logit_reshape = X.reshape(
            cls_logit,
            shape=(0, -4, 2, -1, 0, 0),  # (N,C,H,W) -> (N,2,C/2,H,W)
            name="rpn_cls_logit_reshape_"
        )
        cls_score = X.softmax(
            cls_logit_reshape,
            axis=1,
            name='rpn_cls_score'
        )
        cls_logit_reshape = X.reshape(
            cls_score,
            shape=(0, -3, 0, 0),
            name='rpn_cls_score_reshape'
        )

        # TODO: ask all to add is_train filed in RPNParam
        proposal = X.proposal(
            cls_prob=cls_logit_reshape,
            bbox_pred=bbox_delta,
            im_info=im_info,
            name='proposal',
            feature_stride=rpn_stride,
            scales=tuple(anchor_scale),
            ratios=tuple(anchor_ratio),
            rpn_pre_nms_top_n=pre_nms_top_n,
            rpn_post_nms_top_n=post_nms_top_n,
            threshold=nms_thr,
            rpn_min_size=min_bbox_side,
            iou_loss=False,
            output_score=True
        )

        if p.use_symbolic_proposal is not None:
            batch_size = p.batch_image
            max_side = p.anchor_generate.max_side
            assert max_side is not None, "symbolic proposal requires max_side of image"

            from mxnext.tvm.proposal import proposal as Proposal
            proposal = Proposal(
                cls_prob=cls_logit_reshape,
                bbox_pred=bbox_delta,
                im_info=im_info,
                name='proposal',
                feature_stride=rpn_stride,
                scales=tuple(anchor_scale),
                ratios=tuple(anchor_ratio),
                rpn_pre_nms_top_n=pre_nms_top_n,
                rpn_post_nms_top_n=post_nms_top_n,
                threshold=nms_thr,
                batch_size=batch_size,
                max_side=max_side,
                output_score=True,
                variant="simpledet"
            )

        self._proposal = proposal

        return proposal

    def get_sampled_proposal(self, conv_feat, gt_bbox, im_info):
        p = self.p

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

        (proposal, proposal_score) = self.get_all_proposal(conv_feat, im_info)

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
            name="subsample_proposal"
        )

        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))

        return bbox, label, bbox_target, bbox_weight


class BboxHead(object):
    def __init__(self, pBbox):
        self.p = patch_config_as_nothrow(pBbox)

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
            name='bbox_cls_logit',
            init=X.gauss(0.01)
        )
        bbox_delta = X.fc(
            head_feat["regression"],
            filter=4 * num_reg_class,
            name='bbox_reg_delta',
            init=X.gauss(0.001)
        )

        return cls_logit, bbox_delta

    def get_prediction(self, conv_feat, im_info, proposal):
        p = self.p
        bbox_mean = p.regress_target.mean
        bbox_std = p.regress_target.std
        batch_image = p.batch_image
        num_class = p.num_class
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = 2 if class_agnostic else num_class

        cls_logit, bbox_delta = self.get_output(conv_feat)

        bbox_delta = X.reshape(
            bbox_delta,
            shape=(batch_image, -1, 4 * num_reg_class),
            name='bbox_delta_reshape'
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
        return cls_score, bbox_xyxy

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        batch_roi = p.image_roi * p.batch_image
        batch_image = p.batch_image
        smooth_l1_scalar = p.regress_target.smooth_l1_scalar or 1.0

        cls_logit, bbox_delta = self.get_output(conv_feat)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # classification loss
        cls_loss = X.softmax_output(
            data=cls_logit,
            label=cls_label,
            normalization='batch',
            grad_scale=1.0 * scale_loss_shift,
            name='bbox_cls_loss'
        )

        # bounding box regression
        reg_loss = X.smooth_l1(
            bbox_delta - bbox_target,
            scalar=smooth_l1_scalar,
            name='bbox_reg_l1'
        )
        reg_loss = bbox_weight * reg_loss
        reg_loss = X.loss(
            reg_loss,
            grad_scale=1.0 / batch_roi * scale_loss_shift,
            name='bbox_reg_loss',
        )

        # append label
        cls_label = X.reshape(
            cls_label,
            shape=(batch_image, -1),
            name='bbox_label_reshape'
        )
        cls_label = X.block_grad(cls_label, name='bbox_label_blockgrad')

        # output
        return cls_loss, reg_loss, cls_label


class Bbox2fcHead(BboxHead):
    def __init__(self, pBbox):
        super().__init__(pBbox)

    def add_norm(self, sym):
        p = self.p
        if p.normalizer.__name__ == "fix_bn":
            pass
        elif p.normalizer.__name__ in ["sync_bn", "local_bn", "gn", "dummy"]:
            sym = p.normalizer(sym)
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))
        return sym

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        flatten = X.reshape(conv_feat, shape=(0, -1, 1, 1), name="bbox_feat_reshape")
        fc1 = X.conv(flatten, filter=1024, name="bbox_fc1", init=xavier_init)
        fc1 = self.add_norm(fc1)
        fc1 = X.relu(fc1)
        fc2 = X.conv(fc1, filter=1024, name="bbox_fc2", init=xavier_init)
        fc2 = self.add_norm(fc2)
        fc2 = X.relu(fc2)

        self._head_feat = fc2

        return self._head_feat