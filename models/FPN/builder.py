from __future__ import print_function

import mxnet as mx
import mxnext as X

from symbol.builder import Backbone, RpnHead, BboxHead, Neck, RoiAlign
from models.FPN import assign_layer_fpn, get_top_proposal


class FPNBbox2fcHead(BboxHead):
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

        flatten = X.flatten(conv_feat, name="bbox_feat_flatten")
        fc1 = X.fc(flatten, filter=1024, name="bbox_fc1", init=xavier_init)
        fc1 = self.add_norm(fc1)
        fc1 = X.relu(fc1)
        fc2 = X.fc(fc1, filter=1024, name="bbox_fc2", init=xavier_init)
        fc2 = self.add_norm(fc2)
        fc2 = X.relu(fc2)

        self._head_feat = fc2

        return self._head_feat


class FPNBboxDualHeadSmall(BboxHead):
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

    def _reg_head(self, conv_feat):
        num_block = self.p.num_block or 4

        for i in range(num_block):
            conv_feat = X.conv(
                conv_feat, 
                kernel=3, 
                filter=256, 
                init=X.gauss(0.01), 
                name="bbox_reg_block%s" % (i + 1)
            )
            conv_feat = self.add_norm(conv_feat)
            conv_feat = X.relu(conv_feat)
        
        return conv_feat

    def _cls_head(self, conv_feat):
        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        flatten = X.flatten(conv_feat, name="bbox_feat_flatten")
        fc1 = X.fc(flatten, filter=1024, name="bbox_cls_fc1", init=xavier_init)
        fc1 = self.add_norm(fc1)
        fc1 = X.relu(fc1)
        fc2 = X.fc(fc1, filter=1024, name="bbox_cls_fc2", init=xavier_init)
        fc2 = self.add_norm(fc2)
        fc2 = X.relu(fc2)

        return fc2

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        self._head_feat = dict(
            classification=self._cls_head(conv_feat), 
            regression=self._reg_head(conv_feat)
        )

        return self._head_feat


class FPNRpnHead(RpnHead):
    def __init__(self, pRpn):
        super().__init__(pRpn)

        self.cls_logit_dict         = None
        self.bbox_delta_dict        = None
        self._proposal              = None
        self._proposal_scores       = None

    def get_output(self, conv_fpn_feat):
        if self.cls_logit_dict is not None and self.bbox_delta_dict is not None:
            return self.cls_logit_dict, self.bbox_delta_dict

        p = self.p
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        conv_channel = p.head.conv_channel

        # FPN RPN share weight
        rpn_conv_weight = X.var('rpn_conv_weight', init=X.gauss(0.01))
        rpn_conv_bias = X.var('rpn_conv_bias', init=X.zero_init())
        rpn_conv_gamma = X.var('rpn_conv_gamma')
        rpn_conv_beta = X.var('rpn_conv_beta')
        rpn_conv_mmean = X.var('rpn_conv_moving_mean')
        rpn_conv_mvar = X.var('rpn_conv_moving_var')
        rpn_conv_cls_weight = X.var('rpn_conv_cls_weight', init=X.gauss(0.01))
        rpn_conv_cls_bias = X.var('rpn_conv_cls_bias', init=X.zero_init())
        rpn_conv_bbox_weight = X.var('rpn_conv_bbox_weight', init=X.gauss(0.01))
        rpn_conv_bbox_bias = X.var('rpn_conv_bbox_bias', init=X.zero_init())

        cls_logit_dict = {}
        bbox_delta_dict = {}

        for stride in p.anchor_generate.stride:
            rpn_conv = X.conv(
                conv_fpn_feat['stride%s' % stride],
                kernel=3,
                filter=conv_channel,
                name="rpn_conv_3x3_%s" % stride,
                no_bias=False,
                weight=rpn_conv_weight,
                bias=rpn_conv_bias
            )

            if p.normalizer.__name__ == "fix_bn":
                pass
            elif p.normalizer.__name__ == "sync_bn":
                rpn_conv = p.normalizer(
                    rpn_conv,
                    gamma=rpn_conv_gamma,
                    beta=rpn_conv_beta,
                    moving_mean=rpn_conv_mmean,
                    moving_var=rpn_conv_mvar,
                    name="rpn_conv_3x3_bn_%s" % stride
                )
            elif p.normalizer.__name__ == "gn":
                rpn_conv = p.normalizer(
                    rpn_conv,
                    gamma=rpn_conv_gamma,
                    beta=rpn_conv_beta,
                    name="rpn_conv_3x3_gn_%s" % stride
                )
            else:
                raise NotImplementedError("Unsupported normalizer {}".format(p.normalizer.__name__))

            rpn_relu = X.relu(rpn_conv, name='rpn_relu_%s' % stride)
            if p.fp16:
                rpn_relu = X.to_fp32(rpn_relu, name="rpn_relu_%s_fp32" % stride)
            cls_logit = X.conv(
                rpn_relu,
                filter=2 * num_base_anchor,
                name="rpn_cls_score_stride%s" % stride,
                no_bias=False,
                weight=rpn_conv_cls_weight,
                bias=rpn_conv_cls_bias
            )

            bbox_delta = X.conv(
                rpn_relu,
                filter=4 * num_base_anchor,
                name="rpn_bbox_pred_stride%s" % stride,
                no_bias=False,
                weight=rpn_conv_bbox_weight,
                bias=rpn_conv_bbox_bias
            )

            cls_logit_dict[stride]  = cls_logit
            bbox_delta_dict[stride] = bbox_delta

        self.cls_logit_dict = cls_logit_dict
        self.bbox_delta_dict = bbox_delta_dict

        return self.cls_logit_dict, self.bbox_delta_dict

    def get_anchor_target(self, conv_fpn_feat):
        raise NotImplementedError

    def get_loss(self, conv_fpn_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        batch_image = p.batch_image
        image_anchor = p.anchor_generate.image_anchor
        rpn_stride = p.anchor_generate.stride

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_fpn_feat)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        rpn_cls_logit_list = []
        rpn_bbox_delta_list = []

        for stride in rpn_stride:
            rpn_cls_logit = cls_logit_dict[stride]
            rpn_bbox_delta = bbox_delta_dict[stride]
            rpn_cls_logit_reshape = X.reshape(
                data=rpn_cls_logit,
                shape=(0, 2, -1),
                name="rpn_cls_score_reshape_stride%s" % stride
            )
            rpn_bbox_delta_reshape = X.reshape(
                data=rpn_bbox_delta,
                shape=(0, 0, -1),
                name="rpn_bbox_pred_reshape_stride%s" % stride
            )
            rpn_bbox_delta_list.append(rpn_bbox_delta_reshape)
            rpn_cls_logit_list.append(rpn_cls_logit_reshape)

        # concat output of each level
        rpn_bbox_delta_concat = X.concat(rpn_bbox_delta_list, axis=2, name="rpn_bbox_pred_concat")
        rpn_cls_logit_concat = X.concat(rpn_cls_logit_list, axis=2, name="rpn_cls_score_concat")

        cls_loss = X.softmax_output(
            data=rpn_cls_logit_concat,
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
            (rpn_bbox_delta_concat - bbox_target),
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

    def get_all_proposal(self, conv_fpn_feat, im_info):
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
        num_anchors = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_fpn_feat)

        # rpn rois for multi level feature
        proposal_list = []
        proposal_scores_list = []
        for stride in rpn_stride:
            rpn_cls_logit = cls_logit_dict[stride]
            rpn_bbox_delta = bbox_delta_dict[stride]
            # ROI Proposal
            rpn_cls_logit_reshape = X.reshape(
                data=rpn_cls_logit,
                shape=(0, 2, -1, 0),
                name="rpn_cls_logit_reshape_stride%s" % stride)
            rpn_cls_score = mx.symbol.SoftmaxActivation(
                data=rpn_cls_logit_reshape,
                mode="channel",
                name="rpn_cls_score_stride%s" % stride)
            rpn_cls_score_reshape = X.reshape(
                data=rpn_cls_score,
                shape=(0, 2 * num_anchors, -1, 0),
                name="rpn_cls_score_reshape_stride%s" % stride)
            rpn_proposal, rpn_proposal_scores = mx.sym.contrib.Proposal_v3(
                cls_prob=rpn_cls_score_reshape,
                bbox_pred=rpn_bbox_delta,
                im_info=im_info,
                rpn_pre_nms_top_n=pre_nms_top_n,
                rpn_post_nms_top_n=post_nms_top_n,
                feature_stride=stride,
                output_score=True,
                scales=tuple(anchor_scale),
                ratios=tuple(anchor_ratio),
                rpn_min_size=min_bbox_side,
                threshold=nms_thr,
                iou_loss=False)
            proposal_list.append(rpn_proposal)
            proposal_scores_list.append(rpn_proposal_scores)

        # concat output rois of each level
        proposal_concat = X.concat(proposal_list, axis=1, name="proposal_concat")
        proposal_scores_concat = X.concat(proposal_scores_list, axis=1, name="proposal_scores_concat")

        proposal = mx.symbol.Custom(bbox=proposal_concat, score=proposal_scores_concat,
                                    op_type='get_top_proposal', top_n=post_nms_top_n)

        self._proposal = proposal

        return proposal

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


class MSRAResNet50V1FPN(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 50, "fpn", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class MSRAResNet101V1FPN(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 101, "fpn", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class FPNNeck(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)
        self.fpn_feat = None

    def add_norm(self, sym):
        p = self.p
        if p.normalizer.__name__ == "fix_bn":
            pass
        elif p.normalizer.__name__ in ["sync_bn", "gn"]:
            sym = p.normalizer(sym)
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))
        return sym

    def fpn_neck(self, data):
        if self.fpn_feat is not None:
            return self.fpn_feat

        c2, c3, c4, c5 = data

        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        # P5
        p5 = X.conv(
            data=c5,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_lateral_weight", init=xavier_init),
            bias=X.var(name="P5_lateral_bias", init=X.zero_init()),
            name="P5_lateral"
        )
        p5 = self.add_norm(p5)
        p5_conv = X.conv(
            data=p5,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_conv_weight", init=xavier_init),
            bias=X.var(name="P5_conv_bias", init=X.zero_init()),
            name="P5_conv"
        )
        p5_conv = self.add_norm(p5_conv)

        # P4
        p5_up = mx.sym.UpSampling(
            p5,
            scale=2,
            sample_type="nearest",
            name="P5_upsampling",
            num_args=1
        )
        p4_la = X.conv(
            data=c4,
            filter=256,
            no_bias=False,
            weight=X.var(name="P4_lateral_weight", init=xavier_init),
            bias=X.var(name="P4_lateral_bias", init=X.zero_init()),
            name="P4_lateral"
        )
        p4_la = self.add_norm(p4_la)
        p5_clip = mx.sym.slice_like(p5_up, p4_la, name="P4_clip")
        p4 = mx.sym.add_n(p5_clip, p4_la, name="P4_sum")

        p4_conv = X.conv(
            data=p4,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P4_conv_weight", init=xavier_init),
            bias=X.var(name="P4_conv_bias", init=X.zero_init()),
            name="P4_conv"
        )
        p4_conv = self.add_norm(p4_conv)

        # P3
        p4_up = mx.sym.UpSampling(
            p4,
            scale=2,
            sample_type="nearest",
            name="P4_upsampling",
            num_args=1
        )
        p3_la = X.conv(
            data=c3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P3_lateral_weight", init=xavier_init),
            bias=X.var(name="P3_lateral_bias", init=X.zero_init()),
            name="P3_lateral"
        )
        p3_la = self.add_norm(p3_la)
        p4_clip = mx.sym.slice_like(p4_up, p3_la, name="P3_clip")
        p3 = mx.sym.add_n(p4_clip, p3_la, name="P3_sum")

        p3_conv = X.conv(
            data=p3,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P3_conv_weight", init=xavier_init),
            bias=X.var(name="P3_conv_bias", init=X.zero_init()),
            name="P3_conv"
        )
        p3_conv = self.add_norm(p3_conv)

        # P2
        p3_up = mx.sym.UpSampling(
            p3,
            scale=2,
            sample_type="nearest",
            name="P3_upsampling",
            num_args=1
        )
        p2_la = X.conv(
            data=c2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P2_lateral_weight", init=xavier_init),
            bias=X.var(name="P2_lateral_bias", init=X.zero_init()),
            name="P2_lateral"
        )
        p2_la = self.add_norm(p2_la)
        p3_clip = mx.sym.slice_like(p3_up, p2_la, name="P2_clip")
        p2 = mx.sym.add_n(p3_clip, p2_la, name="P2_sum")

        p2_conv = X.conv(
            data=p2,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P2_conv_weight", init=xavier_init),
            bias=X.var(name="P2_conv_bias", init=X.zero_init()),
            name="P2_conv"
        )
        p2_conv = self.add_norm(p2_conv)

        # P6
        p6 = X.max_pool(
            p5_conv,
            name="P6_subsampling",
            kernel=1,
            stride=2,
        )

        conv_fpn_feat = dict(
            stride64=p6,
            stride32=p5_conv,
            stride16=p4_conv,
            stride8=p3_conv,
            stride4=p2_conv
        )

        self.fpn_feat = conv_fpn_feat
        return self.fpn_feat

    def get_rpn_feature(self, rpn_feat):
        return self.fpn_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.fpn_neck(rcnn_feat)


class FPNRoiAlign(RoiAlign):
    def __init__(self, pRoi):
        super().__init__(pRoi)

    def get_roi_feature(self, conv_fpn_feat, proposal):
        p = self.p
        rcnn_stride = p.stride
        roi_canonical_scale = p.roi_canonical_scale
        roi_canonical_level = p.roi_canonical_level

        group = mx.symbol.Custom(
            op_type="assign_layer_fpn",
            rois=proposal,
            rcnn_stride=rcnn_stride,
            roi_canonical_scale=roi_canonical_scale,
            roi_canonical_level=roi_canonical_level,
            name="assign_layer_fpn"
        )
        proposal_fpn = dict()
        for i, stride in enumerate(rcnn_stride):
            proposal_fpn["stride%s" % stride] = group[i]

        if p.fp16:
            for stride in rcnn_stride:
                conv_fpn_feat["stride%s" % stride] = X.to_fp32(
                    conv_fpn_feat["stride%s" % stride],
                    name="fpn_stride%s_to_fp32"
                )

        fpn_roi_feats = list()
        for stride in rcnn_stride:
            feat_lvl = conv_fpn_feat["stride%s" % stride]
            proposal_lvl = proposal_fpn["stride%s" % stride]
            roi_feat = X.roi_align(
                feat_lvl,
                rois=proposal_lvl,
                out_size=p.out_size,
                stride=stride,
                name="roi_align"
            )
            roi_feat = X.reshape(
                data=roi_feat,
                shape=(-3, -2),
                name='roi_feat_reshape'
            )
            fpn_roi_feats.append(roi_feat)
        roi_feat = X.add_n(*fpn_roi_feats)

        if p.fp16:
            roi_feat = X.to_fp16(roi_feat, name="roi_feat_to_fp16")

        return roi_feat
