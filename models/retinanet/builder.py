from __future__ import division
from __future__ import print_function

import math
import mxnext as X

from symbol.builder import RpnHead, Backbone, Neck


class RetinaNet(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head):
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")

        feat = backbone.get_rpn_feature()
        feat = neck.get_rpn_feature(feat)

        loss = head.get_loss(feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)

        return X.group(loss)

    @staticmethod
    def get_test_symbol(backbone, neck, head):
        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        feat = backbone.get_rpn_feature()
        feat = neck.get_rpn_feature(feat)

        cls_score, bbox_xyxy = head.get_prediction(feat, im_info)

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])


class RetinaNetHead(RpnHead):
    def __init__(self, pRpn):
        super().__init__(pRpn)

        # init bias for cls
        prior_prob = 0.01
        pi = -math.log((1-prior_prob) / prior_prob)

        # shared classification weight and bias
        self.cls_conv1_weight = X.var("cls_conv1_weight", init=X.gauss(std=0.01))
        self.cls_conv1_bias = X.var("cls_conv1_bias", init=X.zero_init())
        self.cls_conv2_weight = X.var("cls_conv2_weight", init=X.gauss(std=0.01))
        self.cls_conv2_bias = X.var("cls_conv2_bias", init=X.zero_init())
        self.cls_conv3_weight = X.var("cls_conv3_weight", init=X.gauss(std=0.01))
        self.cls_conv3_bias = X.var("cls_conv3_bias", init=X.zero_init())
        self.cls_conv4_weight = X.var("cls_conv4_weight", init=X.gauss(std=0.01))
        self.cls_conv4_bias = X.var("cls_conv4_bias", init=X.zero_init())
        self.cls_pred_weight = X.var("cls_pred_weight", init=X.gauss(std=0.01))
        self.cls_pred_bias = X.var("cls_pred_bias", init=X.constant(pi))

        # shared regression weight and bias
        self.bbox_conv1_weight = X.var("bbox_conv1_weight", init=X.gauss(std=0.01))
        self.bbox_conv1_bias = X.var("bbox_conv1_bias", init=X.zero_init())
        self.bbox_conv2_weight = X.var("bbox_conv2_weight", init=X.gauss(std=0.01))
        self.bbox_conv2_bias = X.var("bbox_conv2_bias", init=X.zero_init())
        self.bbox_conv3_weight = X.var("bbox_conv3_weight", init=X.gauss(std=0.01))
        self.bbox_conv3_bias = X.var("bbox_conv3_bias", init=X.zero_init())
        self.bbox_conv4_weight = X.var("bbox_conv4_weight", init=X.gauss(std=0.01))
        self.bbox_conv4_bias = X.var("bbox_conv4_bias", init=X.zero_init())
        self.bbox_pred_weight = X.var("bbox_pred_weight", init=X.gauss(std=0.01))
        self.bbox_pred_bias = X.var("bbox_pred_bias", init=X.zero_init())

        self._cls_logit_dict        = None
        self._bbox_delta_dict       = None

    def _cls_subnet(self, conv_feat, conv_channel, num_base_anchor, num_class):
        p = self.p

        # classification subnet
        cls_conv1 = X.conv(
            data=conv_feat,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv1_weight,
            bias=self.cls_conv1_bias,
            no_bias=False,
            name="cls_conv1"
        )
        cls_conv1_relu = X.relu(cls_conv1)
        cls_conv2 = X.conv(
            data=cls_conv1_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv2_weight,
            bias=self.cls_conv2_bias,
            no_bias=False,
            name="cls_conv2"
        )
        cls_conv2_relu = X.relu(cls_conv2)
        cls_conv3 = X.conv(
            data=cls_conv2_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv3_weight,
            bias=self.cls_conv3_bias,
            no_bias=False,
            name="cls_conv3"
        )
        cls_conv3_relu = X.relu(cls_conv3)
        cls_conv4 = X.conv(
            data=cls_conv3_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv4_weight,
            bias=self.cls_conv4_bias,
            no_bias=False,
            name="cls_conv4"
        )
        cls_conv4_relu = X.relu(cls_conv4)

        if p.fp16:
            cls_conv4_relu = X.to_fp32(cls_conv4_relu, name="cls_conv4_fp32")

        output_channel = num_base_anchor * (num_class - 1)
        output = X.conv(
            data=cls_conv4_relu,
            kernel=3,
            filter=output_channel,
            weight=self.cls_pred_weight,
            bias=self.cls_pred_bias,
            no_bias=False,
            name="cls_pred"
        )

        return output

    def _bbox_subnet(self, conv_feat, conv_channel, num_base_anchor, num_class):
        p = self.p

        # regression subnet
        bbox_conv1 = X.conv(
            data=conv_feat,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv1_weight,
            bias=self.bbox_conv1_bias,
            no_bias=False,
            name="bbox_conv1"
        )
        bbox_conv1_relu = X.relu(bbox_conv1)
        bbox_conv2 = X.conv(
            data=bbox_conv1_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv2_weight,
            bias=self.bbox_conv2_bias,
            no_bias=False,
            name="bbox_conv2"
        )
        bbox_conv2_relu = X.relu(bbox_conv2)
        bbox_conv3 = X.conv(
            data=bbox_conv2_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv3_weight,
            bias=self.bbox_conv3_bias,
            no_bias=False,
            name="bbox_conv3"
        )
        bbox_conv3_relu = X.relu(bbox_conv3)
        bbox_conv4 = X.conv(
            data=bbox_conv3_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv4_weight,
            bias=self.bbox_conv4_bias,
            no_bias=False,
            name="bbox_conv4"
        )
        bbox_conv4_relu = X.relu(bbox_conv4)

        if p.fp16:
            bbox_conv4_relu = X.to_fp32(bbox_conv4_relu, name="bbox_conv4_fp32")

        output_channel = num_base_anchor * 4
        output = X.conv(
            data=bbox_conv4_relu,
            kernel=3,
            filter=output_channel,
            weight=self.bbox_pred_weight,
            bias=self.bbox_pred_bias,
            no_bias=False,
            name="bbox_pred"
        )

        return output

    def get_output(self, conv_feat):
        if self._cls_logit_dict is not None and self._bbox_delta_dict is not None:
            return self._cls_logit_dict, self._bbox_delta_dict

        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        conv_channel = p.head.conv_channel
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        num_class = p.num_class

        cls_logit_dict = dict()
        bbox_delta_dict = dict()

        for s in stride:
            cls_logit = self._cls_subnet(
                conv_feat=conv_feat["stride%s" % s],
                conv_channel=conv_channel,
                num_base_anchor=num_base_anchor,
                num_class=num_class
            )

            bbox_delta = self._bbox_subnet(
                conv_feat=conv_feat["stride%s" % s],
                conv_channel=conv_channel,
                num_base_anchor=num_base_anchor,
                num_class=num_class
            )

            cls_logit_dict["stride%s" % s] = cls_logit
            bbox_delta_dict["stride%s" % s] = bbox_delta

        self._cls_logit_dict = cls_logit_dict
        self._bbox_delta_dict = bbox_delta_dict

        return self._cls_logit_dict, self._bbox_delta_dict

    def get_anchor_target(self, conv_feat):
        raise NotImplementedError

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        num_class = p.num_class
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)
        cls_logit_reshape_list = []
        bbox_delta_reshape_list = []

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # reshape logit and delta
        for s in stride:
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
            grad_scale=1.0 * scale_loss_shift,
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
            grad_scale=1.0 * scale_loss_shift,
            name="reg_loss"
        )

        return cls_loss, reg_loss

    def get_prediction(self, conv_feat, im_info):
        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        ratios = p.anchor_generate.ratio
        scales = p.anchor_generate.scale
        pre_nms_top_n = p.proposal.pre_nms_top_n
        min_bbox_side = p.proposal.min_bbox_side or 0
        min_det_score = p.proposal.min_det_score
        anchor_target_mean = p.head.mean or (0, 0, 0, 0)
        anchor_target_std = p.head.std or (1, 1, 1, 1)
        num_anchors = len(ratios) * len(scales)

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)

        import mxnet as mx
        if "GenProposalRetina" in mx.sym.contrib.__all__:
            cls_score_list = []
            bbox_xyxy_list = []

            for s in stride:
                cls_prob = X.sigmoid(data=cls_logit_dict["stride%s" % s])
                bbox_delta = bbox_delta_dict["stride%s" % s]
                anchors = mx.sym.contrib.GenAnchor(
                    cls_prob=cls_prob,
                    feature_stride=s,
                    scales=tuple(scales),
                    ratios=tuple(ratios),
                    name='anchors_stride%s' % s
                )

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
                    workspace=512,
                    name="proposal_pre_nms_stride%s" % s
                )

                cls_score_list.append(cls_score)
                bbox_xyxy_list.append(bbox_xyxy)

            cls_score = X.concat(cls_score_list, axis=1, name="cls_score_concat")
            bbox_xyxy = X.concat(bbox_xyxy_list, axis=1, name="bbox_xyxy_concat")
        else:
            cls_score_dict = dict()

            for s in stride:
                cls_score = X.sigmoid(data=cls_logit_dict["stride%s" % s])
                bbox_delta = bbox_delta_dict.pop("stride%s" % s)

                cls_score_dict["cls_score_stride%s" % s] = cls_score
                bbox_delta_dict["bbox_delta_stride%s" % s] = bbox_delta

            import models.retinanet.decode_retina  # noqa: F401
            bbox_xyxy, cls_score = mx.sym.Custom(
                op_type="decode_retina",
                im_info=im_info,
                stride=stride,
                scales=scales,
                ratios=ratios,
                per_level_top_n=pre_nms_top_n,
                thresh=min_det_score,
                **cls_score_dict,
                **bbox_delta_dict,
                name="rois"
            )

        return cls_score, bbox_xyxy


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


class RetinaNetNeck(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)

        self.neck = None

    def get_retinanet_neck(self, data):
        if self.neck is not None:
            return self.neck

        c2, c3, c4, c5 = data

        import mxnet as mx
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
        p5_conv = X.conv(
            data=p5,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_conv_weight", init=xavier_init),
            bias=X.var(name="P5_conv_bias", init=X.zero_init()),
            name="P5_conv"
        )

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

        # P6
        p6 = X.conv(
            data=c5,
            kernel=3,
            stride=2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P6_conv_weight", init=xavier_init),
            bias=X.var(name="P6_conv_bias", init=X.zero_init()),
            name="P6_conv"
        )

        # P7
        p6_relu = X.relu(data=p6, name="P6_relu")
        p7 = X.conv(
            data=p6_relu,
            kernel=3,
            stride=2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P7_conv_weight", init=xavier_init),
            bias=X.var(name="P7_conv_bias", init=X.zero_init()),
            name="P7_conv"
        )

        self.neck = dict(
            stride8=p3_conv,
            stride16=p4_conv,
            stride32=p5_conv,
            stride64=p6,
            stride128=p7
        )

        return self.neck

    def get_rpn_feature(self, rpn_feat):
        return self.get_retinanet_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.get_retinanet_neck(rcnn_feat)
