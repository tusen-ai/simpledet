from __future__ import division
from __future__ import print_function

import mxnet as mx
import mxnext as X
import math

from symbol.builder import BboxHead, RoiExtractor
from models.retinanet.builder import RetinaNetHead


class AlignRetinaNetHead(RetinaNetHead):
    def __init__(self, pRpn):
        super(AlignRetinaNetHead, self).__init__(pRpn)
        self._proposal = None

    def get_all_proposal(self, conv_feat, im_info):
        if self._proposal is not None:
            return self._proposal

        p = self.p
        batch_image = p.batch_image
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        ratios = p.anchor_generate.ratio
        scales = p.anchor_generate.scale
        anchor_target_mean = p.head.mean
        anchor_target_std = p.head.std
        num_base_anchor = len(ratios) * len(scales)
        pick_anchor = p.pick_anchor or False
        nms = p.nms or False

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)

        proposal_dict, proposal_score_dict = dict(), dict()

        for s in stride:
            """
            cls_prob: (N, A * C, H, W)
            bbox_delta: (N, A * 4, H, W)
            """
            cls_prob = X.sigmoid(data=cls_logit_dict["stride%s" % s])
            bbox_delta = bbox_delta_dict["stride%s" % s]

            # (N, A * 4, H, W) -> (N, A, 4, H * W)
            bbox_delta = X.reshape(
                data=bbox_delta,
                shape=(0, num_base_anchor, 4, -1),
                name="bbox_delta_reshape_stride%s" % s
            )
            # (N, A, 4, H * W) -> (N, H * W, A, 4)
            bbox_delta = X.transpose(
                data=bbox_delta,
                axes=(0, 3, 1, 2),
                name="bbox_delta_reshape_transpose_stride%s" % s
            )
            # (N, H * W, A, 4) -> (N, H * W * A, 4)
            bbox_delta = X.reshape(
                data=bbox_delta,
                shape=(0, -1, 4),
                name="bbox_delta_reshape_transpose_reshape_stride%s" % s
            )

            anchor = mx.sym.contrib.GenAnchor(
                cls_prob=bbox_delta_dict["stride%s" % s],
                feature_stride=s,
                scales=tuple(scales),
                ratios=tuple(ratios),
                name="anchors_stride%s" % s
            )

            # decode anchor
            bbox_delta_list = mx.sym.split(bbox_delta, num_outputs=batch_image, axis=0, squeeze_axis=False)
            im_info_list = mx.sym.split(im_info, num_outputs=batch_image, axis=0, squeeze_axis=False)
            bbox_xyxy_list = list()
            anchor_expand_dims = anchor.expand_dims(axis=0)
            for bbox_delta_i, im_info_i in zip(bbox_delta_list, im_info_list):
                pad_zero = mx.sym.zeros_like(bbox_delta_i)
                bbox_delta_i = mx.sym.concat(pad_zero, bbox_delta_i, dim=-1)
                bbox_xyxy_i = X.decode_bbox(
                    rois=anchor_expand_dims,
                    bbox_pred=bbox_delta_i,
                    im_info=im_info_i,
                    bbox_mean=anchor_target_mean,
                    bbox_std=anchor_target_std,
                    class_agnostic=True
                )
                bbox_xyxy_list.append(bbox_xyxy_i)
            bbox_xyxy = mx.sym.concat(*bbox_xyxy_list, dim=0, name="proposal_stride%s_retina" % s)

            proposal_dict["stride%s" % s] = bbox_xyxy
            proposal_score_dict["stride%s" % s] = cls_prob

        if pick_anchor:
            for s in stride:
                cls_score = proposal_score_dict["stride%s"%s]
                bbox_xyxy = proposal_dict["stride%s"%s]

                # (N, A * C, H, W) -> (N, H * W, A), C = 1
                cls_score = cls_score.transpose((0, 2, 3, 1))
                cls_score = cls_score.reshape((0, -3, 0))
                # (N, H * W * A, 4) -> (N, H * W, A, 4)
                bbox_xyxy = bbox_xyxy.reshape((0, -1, num_base_anchor, 4))

                argmax_cls_score = cls_score.argmax(axis=2)
                argmax_cls_score_stack = mx.sym.stack(*([argmax_cls_score] * 4), axis=2)

                sample_cls_score = mx.sym.pick(cls_score, argmax_cls_score, axis=2)
                sample_bbox_xyxy = mx.sym.pick(bbox_xyxy, argmax_cls_score_stack, axis=2)

                # (N, H * W) -> (N, A * C, H * W), A = C = 1
                sample_cls_score = sample_cls_score.reshape((0, 1, -1))

                proposal_score_dict["stride%s"%s] = sample_cls_score
                proposal_dict["stride%s"%s] = sample_bbox_xyxy
        elif nms:
            nms_thr = p.nms_thr

            for s in stride:
                cls_score = proposal_score_dict["stride%s"%s]
                bbox_xyxy = proposal_dict["stride%s"%s]

                # (N, A * C, H, W) -> (N, H * W, A, C)
                cls_score = cls_score.reshape((0, 0, -1))
                cls_score = cls_score.transpose((0, 2, 1))
                cls_score = cls_score.reshape((0, 0, num_base_anchor, -1))
                # (N, H * W, A, C) -> (N, H * W, A, 1)
                cls_score = mx.sym.sum(cls_score, axis=3, keepdims=True)
                proposal_score_dict["stride%s"%s] = cls_score
                # (N, H * W * A, 4) -> (N, H * W, A, 4)
                bbox_xyxy = bbox_xyxy.reshape((0, -1, num_base_anchor, 4))

                (cls_score, bbox_xyxy) = mx.sym.contrib.InplaceNMS(
                    cls_prob=cls_score,
                    bbox_pred=bbox_xyxy,
                    nms_thr=nms_thr,
                    score_reset_value=0,
                    bbox_reset_value=999999, # larger than allowed_border
                    name="inplace_nms"
                )

                # (N, H * W, A, C) -> (N, A * C, H, W), C = 1
                cls_score = cls_score.reshape((0, 0, -3))
                cls_score = cls_score.transpose((0, 2, 1))
                cls_score.reshape_like(proposal_score_dict["stride%s"%s])
                # (N, H * W, A, 4) -> (N, H * W * A, 4)
                bbox_xyxy = bbox_xyxy.reshape_like(proposal_dict["stride%s"%s])

                proposal_dict["stride%s"%s] = bbox_xyxy
                proposal_score_dict["stride%s"%s] = cls_score

        proposal = (proposal_dict, proposal_score_dict)
        self._proposal = proposal

        return proposal

    def get_sampled_proposal(self, conv_feat, gt_bbox, im_info):
        p = self.p

        mean = p.bbox_target.mean
        std = p.bbox_target.std
        class_agnostic = p.bbox_target.class_agnostic
        short = p.anchor_generate.short
        long = p.anchor_generate.long
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        num_anchors = len(p.anchor_generate.scale) * len(p.anchor_generate.ratio)
        allowed_border = p.bbox_target.allowed_border
        pos_thr = p.bbox_target.pos_thr
        neg_thr = p.bbox_target.neg_thr
        min_pos_thr = p.bbox_target.min_pos_thr
        pick_anchor = p.pick_anchor or False
        if pick_anchor:
            num_anchors = 1
 
        (anchor_dict, anchor_score_dict) = self.get_all_proposal(conv_feat, im_info)

        # custom op to encode new target
        from models.aligndet import encode_anchor  # noqa: F401

        (label, bbox_target, bbox_weight) = mx.sym.Custom(
            op_type="encode_anchor",
            gt_boxes=gt_bbox,
            im_info=im_info,
            short=short,
            long=long,
            stride=stride,
            num_anchors=num_anchors,
            class_agnostic=class_agnostic,
            allowed_border=allowed_border,
            pos_thr=pos_thr,
            neg_thr=neg_thr,
            min_pos_thr=min_pos_thr,
            mean=mean,
            std=std,
            name="encode_anchor",
            **anchor_dict
        )

        return anchor_dict, label, bbox_target, bbox_weight

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        num_class = p.num_class
        loss_weight = p.loss_weight
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        scale_loss_shift = 128.0 if p.fp16 else 1.0
        reg_only = p.reg_only or False

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)
        cls_logit_reshape_list = []
        bbox_delta_reshape_list = []

        # reshape logit and delta
        for i, s in enumerate(stride):
            # (N, A * C, H, W) -> (N, A, C, H * W)
            cls_logit = X.reshape(
                data=cls_logit_dict["stride%s" % s],
                shape=(0, num_base_anchor, num_class-1, -1),
                name="cls_stride%s_reshape" % s
            )
            # (N, A, C, H * W) -> (N, A, H * W, C)
            cls_logit = X.transpose(
                data=cls_logit,
                axes=(0, 1, 3, 2),
                name="cls_stride%s_transpose" % s
            )
            # (N, A, H * W, C) -> (N, A * H * W, C)
            cls_logit = X.reshape(
                data=cls_logit,
                shape=(0, -3, 0),
                name="cls_stride%s_transpose_reshape" % s
            )

            # (N, A * 4, H, W) -> (N, A * 4, H * W)
            bbox_delta = X.reshape(
                data=bbox_delta_dict["stride%s" % s],
                shape=(0, 0, -1),
                name="bbox_stride%s_reshape" % s
            )

            cls_logit_reshape_list.append(cls_logit)
            bbox_delta_reshape_list.append(bbox_delta)

        cls_logit_concat = X.concat(cls_logit_reshape_list, axis=1, name="bbox_logit_concat")
        bbox_delta_concat = X.concat(bbox_delta_reshape_list, axis=2, name="bbox_delta_concat")

        # classification loss
        cls_loss = X.focal_loss(
            data=cls_logit_concat,
            label=cls_label,
            normalization='valid',
            alpha=p.focal_loss.alpha,
            gamma=p.focal_loss.gamma,
            grad_scale=0.0 if reg_only else 1.0 * loss_weight * scale_loss_shift,
            workspace=1024,
            name="cls_loss"
        )

        scalar = 0.11
        # regression loss
        bbox_norm = X.bbox_norm(
            data=bbox_delta_concat - bbox_target,
            label=cls_label,
            name="bbox_norm"
        )
        bbox_loss = bbox_weight * X.smooth_l1(
            data=bbox_norm,
            scalar=math.sqrt(1/scalar),
            name="bbox_loss"
        )
        reg_loss = X.make_loss(
            data=bbox_loss,
            grad_scale=1.0 * loss_weight * scale_loss_shift,
            name="reg_loss"
        )

        return cls_loss, reg_loss


class AlignHead(BboxHead):
    def __init__(self, pBbox):
        super(AlignHead, self).__init__(pBbox)

        p = self.p
        num_conv = p.head.num_conv
        init = p.head.init or X.gauss(0.01)
        stage = p.stage or ""
        ignore_p3 = p.head.ignore_p3 or False

        self.align_conv_weight = [X.var("align_conv_%d_weight%s" % (i + 1, stage), init=init) for i in range(num_conv)]
        self.align_conv_bias = [X.var("align_conv_%d_bias%s" % (i + 1, stage), init=X.zero_init()) for i in range(num_conv)]

        if ignore_p3:
            self.align_conv_p3_weight = [X.var("align_conv_p3_%d_weight%s" % (i + 1, stage), init=init) for i in range(num_conv)]
            self.align_conv_p3_bias = [X.var("align_conv_p3_%d_bias%s" % (i + 1, stage), init=init) for i in range(num_conv)]

        self._head_feat_dict = None
        self._cls_logit_dict = None
        self._bbox_delta_dict = None

        self.stage = stage

    def _get_bbox_head_logit(self, conv_feat, conv_channel, ignore_p3=False):
        p = self.p
        num_conv = p.head.num_conv
        use_1x1 = p.head.use_1x1 or False

        for i in range(num_conv):
            conv_feat = X.conv(
                data=conv_feat,
                kernel=1 if use_1x1 else 3,
                filter=conv_channel,
                weight=self.align_conv_p3_weight[i] if ignore_p3 else self.align_conv_weight[i],
                bias=self.align_conv_p3_bias[i] if ignore_p3 else self.align_conv_bias[i],
                no_bias=False,
                name="align_conv_%d%s" % (i + 1, self.stage)
            )
            conv_feat = X.relu(conv_feat)

        if p.fp16:
            conv_feat = X.to_fp32(conv_feat, name="align_conv_fp32")

        return conv_feat

    def get_output(self, conv_feat):
        if self._cls_logit_dict is not None and self._bbox_delta_dict is not None:
            return self._cls_logit_dict, self._bbox_delta_dict

        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        conv_channel = p.head.conv_channel
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        num_class = p.num_class
        separate_predictor = p.head.separate_predictor or False
        batch_image = p.batch_image
        ignore_p3 = p.head.ignore_p3 or False

        prior_prob = 0.01
        pi = -math.log((1 - prior_prob) / prior_prob)

        stage = self.stage
        align_conv_cls_weight = X.var("align_conv_cls_weight%s" % stage, init=X.gauss(std=0.01))
        align_conv_cls_bias = X.var("align_conv_cls_bias%s" % stage, init=X.constant(pi))
        align_conv_bbox_weight = X.var("align_conv_bbox_weight%s" % stage, init=X.gauss(std=0.01))
        align_conv_bbox_bias = X.var("align_conv_bbox_bias%s" % stage, init=X.zero_init())

        head_feat_dict = {}
        cls_logit_dict = {}
        bbox_delta_dict = {}

        for s in stride:
            align_conv_relu = self._get_bbox_head_logit(
                conv_feat=conv_feat["stride%s" % s],
                conv_channel=conv_channel,
                ignore_p3=ignore_p3 and s == 8
            )

            head_feat_dict["stride%s" % s] = align_conv_relu

            if separate_predictor:
                align_conv_relu = X.reshape(align_conv_relu, shape=(batch_image, -1, 0, 0))

                cls_logit = X.conv(
                    align_conv_relu,
                    num_group=num_base_anchor,
                    filter=num_base_anchor * (num_class - 1),
                    no_bias=False,
                    weight=align_conv_cls_weight,
                    bias=align_conv_cls_bias,
                    name="align_cls_score_stride%s" % s
                )

                bbox_delta = X.conv(
                    align_conv_relu,
                    num_group=num_base_anchor,
                    filter=num_base_anchor * 4,
                    no_bias=False,
                    weight=align_conv_bbox_weight,
                    bias=align_conv_bbox_bias,
                    name="align_bbox_pred_stride%s" % s
                )

                cls_logit_dict["stride%s" % s] = cls_logit
                bbox_delta_dict["stride%s" % s] = bbox_delta
            else:
                cls_logit = X.conv(
                    align_conv_relu,
                    filter=num_class - 1,
                    no_bias=False,
                    weight=align_conv_cls_weight,
                    bias=align_conv_cls_bias,
                    name="align_cls_score_stride%s" % s
                )

                bbox_delta = X.conv(
                    align_conv_relu,
                    filter=4,
                    no_bias=False,
                    weight=align_conv_bbox_weight,
                    bias=align_conv_bbox_bias,
                    name="align_bbox_pred_stride%s" % s
                )

                cls_logit_dict["stride%s" % s] = cls_logit.reshape(shape=(-1, num_base_anchor * (num_class - 1), 0, 0))
                bbox_delta_dict["stride%s" % s] = bbox_delta.reshape(shape=(-1, num_base_anchor * 4, 0, 0))

        self._head_feat_dict = head_feat_dict
        self._cls_logit_dict = cls_logit_dict
        self._bbox_delta_dict = bbox_delta_dict

        return cls_logit_dict, bbox_delta_dict

    def get_prediction(self, conv_feat, im_info, proposal):
        p = self.p
        merge_score = p.head.merge_score or False
        use_prev_bbox = p.head.use_prev_bbox or False
        use_prev_score = p.head.use_prev_score or False
        assert not (merge_score and use_prev_score), "merge_score confilicts with use_prev_score"
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        ratios = p.anchor_generate.ratio
        scales = p.anchor_generate.scale
        pre_nms_top_n = p.proposal.pre_nms_top_n
        min_bbox_side = p.proposal.min_bbox_side
        min_det_score = p.proposal.min_det_score
        anchor_target_mean = p.head.mean
        anchor_target_std = p.head.std
        num_anchors = len(ratios) * len(scales)

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)

        bbox_xyxy_list = []
        cls_score_list = []

        proposal, proposal_score = proposal

        for i, s in enumerate(stride):
            cls_prob = X.sigmoid(data=cls_logit_dict["stride%s" % s])
            bbox_delta = bbox_delta_dict["stride%s" % s]
            anchors = proposal["stride%s" % s]
            anchor_scores = proposal_score["stride%s" % s]

            if merge_score:
                cls_prob = cls_prob + anchor_scores

            if use_prev_bbox:
                bbox_delta = mx.sym.zeros_like(bbox_delta)

            if use_prev_score:
                cls_prob = anchor_scores

            thresh_level = 0 if s == max(stride) else min_det_score
            bbox_xyxy, cls_score = mx.sym.contrib.GenProposalRetina(
                cls_prob=cls_prob,
                bbox_pred=bbox_delta,
                im_info=im_info,
                anchors=anchors,
                feature_stride=s,
                anchor_mean=anchor_target_mean,
                anchor_std=anchor_target_std,
                num_anchors=num_anchors,
                rpn_pre_nms_top_n=pre_nms_top_n,
                rpn_min_size=min_bbox_side,
                thresh=thresh_level,
                batch_wise_anchor=True,
                workspace=512,
                name="proposal_pre_nms_stride%s" % s
            )

            bbox_xyxy_list.append(bbox_xyxy)
            cls_score_list.append(cls_score)

        bbox_xyxy = X.concat(bbox_xyxy_list, axis=1, name="align_bbox_xyxy_concat")
        cls_score = X.concat(cls_score_list, axis=1, name="align_cls_score_concat")

        return cls_score, bbox_xyxy

    def get_all_proposal(self, conv_feat, im_info, anchor_dict):
        """
        anchors are the bboxes and scores of the previous stage, not of this stage.
        """

        p = self.p
        batch_image = p.batch_image
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        anchor_target_mean = p.head.mean
        anchor_target_std = p.head.std
        ratios = p.anchor_generate.ratio
        scales = p.anchor_generate.scale
        num_base_anchor = len(ratios) * len(scales)

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)

        proposal_dict, proposal_score_dict = dict(), dict()

        for i, s in enumerate(stride):
            cls_prob = X.sigmoid(data=cls_logit_dict["stride%s" % s])
            bbox_delta = bbox_delta_dict["stride%s" % s]
            anchor = anchor_dict["stride%s" % s]

            # (N, A * 4, H, W) -> (N, A, 4, H * W)
            bbox_delta = X.reshape(
                data=bbox_delta,
                shape=(0, num_base_anchor, 4, -1),
                name="bbox_delta_reshape_stride%s" % s
            )
            # (N, A, 4, H * W) -> (N, H * W, A, 4)
            bbox_delta = X.transpose(
                data=bbox_delta,
                axes=(0, 3, 1, 2),
                name="bbox_delta_reshape_transpose_stride%s" % s
            )
            # (N, H * W, A, 4) -> (N, H * W * A, 4)
            bbox_delta = X.reshape(
                data=bbox_delta,
                shape=(0, -1, 4),
                name="bbox_delta_reshape_transpose_reshape_stride%s" % s
            )

            # decode anchor
            bbox_delta_list = mx.sym.split(bbox_delta, num_outputs=batch_image, axis=0, squeeze_axis=False)
            anchor_list = mx.sym.split(anchor, num_outputs=batch_image, axis=0, squeeze_axis=False)
            im_info_list = mx.sym.split(im_info, num_outputs=batch_image, axis=0, squeeze_axis=False)
            bbox_xyxy_list = list()
            for bbox_delta_i, anchor_i, im_info_i in zip(bbox_delta_list, anchor_list, im_info_list):
                pad_zero = mx.sym.zeros_like(bbox_delta_i)
                bbox_delta_i = mx.sym.concat(pad_zero, bbox_delta_i, dim=-1)
                bbox_xyxy_i = X.decode_bbox(
                    rois=anchor_i,
                    bbox_pred=bbox_delta_i,
                    im_info=im_info_i,
                    bbox_mean=anchor_target_mean,
                    bbox_std=anchor_target_std,
                    class_agnostic=True
                )
                bbox_xyxy_list.append(bbox_xyxy_i)
            bbox_xyxy = mx.sym.concat(*bbox_xyxy_list, dim=0, name="proposal_stride%s_%s" % (s, self.stage))

            proposal_dict["stride%s" % s] = bbox_xyxy
            proposal_score_dict["stride%s" % s] = cls_prob

        return proposal_dict, proposal_score_dict

    def get_proposal_and_label(self, conv_feat, gt_bbox, im_info, prev_anchor):
        """
        Get proposal of this head with the proposal from the previous head.
        Use proposal of this head to encode the training target of the next head.
        """
        p = self.p

        mean = p.bbox_target.mean
        std = p.bbox_target.std
        class_agnostic = p.bbox_target.class_agnostic
        short = p.anchor_generate.short
        long = p.anchor_generate.long
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        num_anchors = len(p.anchor_generate.scale) * len(p.anchor_generate.ratio)
        allowed_border = p.bbox_target.allowed_border
        pos_thr = p.bbox_target.pos_thr
        neg_thr = p.bbox_target.neg_thr
        min_pos_thr = p.bbox_target.min_pos_thr

        anchor_dict, anchor_score_dict = self.get_all_proposal(conv_feat, im_info, prev_anchor)

        # custom op to encode new target
        from models.aligndet import encode_anchor  # noqa: F401

        (label, bbox_target, bbox_weight) = mx.sym.Custom(
            op_type="encode_anchor",
            gt_boxes=gt_bbox,
            im_info=im_info,
            short=short,
            long=long,
            stride=stride,
            num_anchors=num_anchors,
            class_agnostic=class_agnostic,
            allowed_border=allowed_border,
            pos_thr=pos_thr,
            neg_thr=neg_thr,
            min_pos_thr=min_pos_thr,
            mean=mean,
            std=std,
            name="encode_anchor",
            **anchor_dict
        )

        return anchor_dict, label, bbox_target, bbox_weight

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        num_class = p.num_class
        loss_weight = p.loss_weight
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        stage = self.stage

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)
        cls_logit_reshape_list = []
        bbox_delta_reshape_list = []

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # reshape logit and delta
        for i, s in enumerate(stride):
            # (N, A * C, H, W) -> (N, A, C, H * W)
            cls_logit = X.reshape(
                data=cls_logit_dict["stride%s" % s],
                shape=(0, num_base_anchor, num_class - 1, -1),
                name="align_cls_stride%s_reshape" % s
            )
            # (N, A, C, H * W) -> (N, A, H * W, C)
            cls_logit = X.transpose(
                data=cls_logit,
                axes=(0, 1, 3, 2),
                name="align_cls_stride%s_transpose" % s
            )
            # (N, A, H * W, C) -> (N, A * H * W, C)
            cls_logit = X.reshape(
                data=cls_logit,
                shape=(0, -3, 0),
                name="align_cls_stride%s_transpose_reshape" % s
            )

            # (N, A * 4, H, W) -> (N, A * 4, H * W)
            bbox_delta = X.reshape(
                data=bbox_delta_dict["stride%s" % s],
                shape=(0, 0, -1),
                name="align_bbox_stride%s_reshape" % s
            )

            cls_logit_reshape_list.append(cls_logit)
            bbox_delta_reshape_list.append(bbox_delta)

        cls_logit_concat = X.concat(
            cls_logit_reshape_list,
            axis=1,
            name="align_bbox_logit_concat"
        )
        bbox_delta_concat = X.concat(
            bbox_delta_reshape_list,
            axis=2,
            name="align_bbox_delta_concat"
        )

        # classification loss
        cls_loss = X.focal_loss(
            data=cls_logit_concat,
            label=cls_label,
            normalization='valid',
            alpha=p.focal_loss.alpha,
            gamma=p.focal_loss.gamma,
            grad_scale=1.0 * loss_weight * scale_loss_shift,
            workspace=1024,
            name="align_cls_loss%s" % stage
        )

        scalar = 0.11
        # regression loss
        bbox_norm = X.bbox_norm(
            data=bbox_delta_concat - bbox_target,
            label=cls_label,
            name="align_bbox_norm"
        )
        bbox_loss = bbox_weight * X.smooth_l1(
            data=bbox_norm,
            scalar=math.sqrt(1 / scalar),
            name="align_bbox_loss"
        )
        reg_loss = X.make_loss(
            data=bbox_loss,
            grad_scale=1.0 * loss_weight * scale_loss_shift,
            name="align_reg_loss%s" % stage
        )

        cls_label = X.block_grad(cls_label, name="align_label_blockgrad%s" % stage)

        return cls_loss, reg_loss, cls_label


class AlignRoiExtractor(RoiExtractor):
    def __init__(self, pRoi):
        super(AlignRoiExtractor, self).__init__(pRoi)

    def get_roi_feature(self, feat_dict, anchor_dict):
        p = self.p
        stride = p.stride
        if not isinstance(stride, tuple):
            stride = (stride, )
        conv_channel = p.conv_channel
        ratios = p.ratio
        scales = p.scale
        num_anchors = len(ratios) * len(scales)
        sample_bins = p.sample_bins
        stage = p.stage or ""
        im2col = p.im2col or False
        conv3d = p.conv3d or False
        roialign = p.roialign or False
        ignore_p3 = p.ignore_p3 or False
        gauss_init = p.gauss_init or False
        guided_anchor = p.guided_anchor or False
        learned_offset = p.learned_offset or False
        assert not (guided_anchor and learned_offset)

        if p.fp16:
            for s in stride:
                feat_dict["stride%s" % s] = X.to_fp32(
                    feat_dict["stride%s" % s],
                    name="feat_stride%s_to_fp32%s" % (s, stage)
                )

        anchor_feat_dict = {}

        for s in stride:
            if ignore_p3 and s == 8:
                old_sample_bins = sample_bins
                old_conv_channel = conv_channel
                sample_bins = 3
                conv_channel = old_conv_channel // (old_sample_bins ** 2) * (sample_bins ** 2)

            if guided_anchor:
                # (N, H * W * A, 4) -> (N, A * K * K * 2, H, W)
                (x1y1, x2y2) = mx.sym.split(anchor_dict["stride%s" % s], num_outputs=2, axis=-1)
                hw = x2y2 - x1y1  # (N, H * W * A, 2)
                hw = X.reshape(hw, [0, -1, 2 * num_anchors])  # (N, H * W, A * 2)
                hw = X.transpose(hw, [0, 2, 1])  # (N, A * 2, H * W)
                hw = mx.sym.reshape_like(hw, feat_dict["stride%s" % s], lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)  # (N, A * 2, H, W)
                # normalize input hw
                hw = mx.sym.BatchNorm(hw, fix_gamma=False, use_global_stats=False, name="guided_anchor_bn_stride%s" % s)
                offset = X.conv(hw, name="guided_anchor_offset_stride%s" % s, filter=2 * num_anchors * sample_bins ** 2, init=X.gauss(0.01))
            elif learned_offset:
                offset = X.conv(feat_dict["stride%s" % s], name="learned_offset_stride%s" % s, filter=2 * num_anchors * sample_bins ** 2, init=X.gauss(0.01))
            else:
                # (N, H * W * A, 4) -> (N, A * K * K * 2, H, W)
                offset = mx.sym.contrib.GetAnchorOffset(
                    data=feat_dict["stride%s" % s],
                    anchor=anchor_dict["stride%s" % s],
                    kernel=(sample_bins, sample_bins),
                    stride=s,
                    name="get_anchor_offset_stride%s%s" % (s, stage)
                )

            anchor_feat_list = []
            for anchor_idx in range(num_anchors):
                offset_i = mx.sym.slice_axis(
                    offset,
                    begin=anchor_idx * (sample_bins * sample_bins * 2),
                    end=(anchor_idx + 1) * (sample_bins * sample_bins * 2),
                    axis=1,
                    name="anchor_offset_stride%s_slice%s%s" % (s, anchor_idx, stage)
                )
                if im2col:
                    anchor_feat_i = mx.sym.contrib.DeformableConvolutionIm2Col(
                        data=feat_dict["stride%s" % s],
                        offset=offset_i,
                        kernel=(sample_bins, sample_bins),
                        pad=(sample_bins // 2, sample_bins // 2),
                        num_filter=conv_channel,
                        name="deform_im2col_stride%s_slice%s%s" % (s, anchor_idx, stage)
                    )

                else:
                    if gauss_init:
                        w = mx.sym.var("deform_conv_stride%s_slice%s%s_weight" % (s, anchor_idx, stage), init=X.gauss(0.01))
                    else:
                        w = None
                    anchor_feat_i = mx.sym.contrib.DeformableConvolution(
                        data=feat_dict["stride%s" % s],
                        offset=offset_i,
                        weight=w,
                        kernel=(sample_bins, sample_bins),
                        pad=(sample_bins // 2, sample_bins // 2),
                        num_filter=conv_channel,
                        num_deformable_group=1,
                        no_bias=False,
                        name="deform_conv_stride%s_slice%s%s" % (s, anchor_idx, stage)
                    )
                anchor_feat_list.append(anchor_feat_i)
            anchor_feat = X.concat(
                anchor_feat_list,
                axis=1,
                name="anchor_feat_concat_stride%s%s" % (s, stage)
            )

            anchor_feat = X.reshape(anchor_feat, shape=(-1, conv_channel, 0, 0))

            if roialign:
                # (N, H * W * A, 4) -> (N, H * W * A, C, bin, bin)
                feat = feat_dict["stride%s" % s]
                anchor = anchor_dict["stride%s" % s]
                anchor_feat = X.roi_align(feat, anchor, out_size=sample_bins, stride=s, name="roialign_stride%s" % s)
                anchor_feat = X.reshape(anchor_feat, [0, 0, -1])
                anchor_feat = X.transpose(anchor_feat, [0, 2, 1])
                anchor_feat = mx.sym.reshape_like(anchor_feat, feat, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)

            anchor_feat_dict["stride%s" % s] = anchor_feat

            if ignore_p3 and s == 8:
                sample_bins = old_sample_bins
                conv_channel = old_conv_channel

        if p.fp16:
            for s in stride:
                anchor_feat_dict["stride%s" % s] = X.to_fp16(
                    anchor_feat_dict["stride%s" % s],
                    name="anchor_feat_stride%s_to_fp16%s" % (s, stage)
                )

        return anchor_feat_dict

    def get_roi_feature_test(self, feat_dict, anchor_dict):
        return self.get_roi_feature(feat_dict, anchor_dict)


class FlatRoiExtractor(RoiExtractor):
    def __init__(self, pRoi):
        super(FlatRoiExtractor, self).__init__(pRoi)

    def get_roi_feature(self, feat_dict, anchor_dict):
        return feat_dict

    def get_roi_feature_test(self, feat_dict, anchor_dict):
        return self.get_roi_feature(feat_dict, anchor_dict)


class CascadeRcnn(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, roi_extractor: AlignRoiExtractor, roi_extractor_2nd: AlignRoiExtractor, bbox_head: AlignHead, bbox_head_2nd: AlignHead, share_feat=False):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_loss = rpn_head.get_loss(rpn_feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)

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

        # stage2
        # though call get_sampled_proposal, bbox_head does not sample rois
        proposal_2nd, bbox_cls_2nd, bbox_target_2nd, bbox_weight_2nd = \
            bbox_head.get_proposal_and_label(
                roi_feat,
                gt_bbox,
                im_info,
                proposal,
            )
        if share_feat:
            feat = rcnn_feat
        else:
            feat = bbox_head._head_feat_dict
        roi_feat_2nd = roi_extractor_2nd.get_roi_feature(feat, proposal_2nd)
        bbox_loss_2nd = bbox_head_2nd.get_loss(
            roi_feat_2nd,
            bbox_cls_2nd,
            bbox_target_2nd,
            bbox_weight_2nd
        )

        return X.group(rpn_loss + bbox_loss + bbox_loss_2nd)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head: AlignRetinaNetHead, roi_extractor: AlignRoiExtractor, roi_extractor_2nd: AlignRoiExtractor, bbox_head: AlignHead, bbox_head_2nd: AlignHead, stage=3, share_feat=False):
        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        if stage == 1:
            cls_score, bbox_xyxy = rpn_head.get_prediction(rpn_feat, im_info)
        elif stage == 2:
            proposal, proposal_score = rpn_head.get_all_proposal(rpn_feat, im_info)
            roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
            cls_score, bbox_xyxy = bbox_head.get_prediction(
                roi_feat,
                im_info,
                (proposal, proposal_score)
            )
        elif stage == 3:
            proposal, proposal_score = rpn_head.get_all_proposal(rpn_feat, im_info)
            roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
            proposal_2nd, proposal_score_2nd = bbox_head.get_all_proposal(
                roi_feat,
                im_info,
                proposal
            )
            if share_feat:
                feat = rcnn_feat
            else:
                feat = bbox_head._head_feat_dict
            roi_feat_2nd = roi_extractor_2nd.get_roi_feature(feat, proposal_2nd)
            cls_score, bbox_xyxy = bbox_head_2nd.get_prediction(
                roi_feat_2nd,
                im_info,
                (proposal_2nd, proposal_score_2nd)
            )
        else:
            raise ValueError("No more stages")

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])
