from __future__ import division
from __future__ import print_function

import math
import mxnext as X

from models.retinanet.builder import RetinaNetHead


class FreeAnchorRetinaNet(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")

        feat = backbone.get_rpn_feature()
        feat = neck.get_rpn_feature(feat)

        head.get_anchor()
        loss = head.get_loss(feat, gt_bbox, im_info)

        return X.group(loss)

    @staticmethod
    def get_test_symbol(backbone, neck, head):
        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        feat = backbone.get_rpn_feature()
        feat = neck.get_rpn_feature(feat)

        head.get_anchor()
        cls_score, bbox_xyxy = head.get_prediction(feat, im_info)

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])


class FreeAnchorRetinaNetHead(RetinaNetHead):
    def __init__(self, pRpn):
        super().__init__(pRpn)
        # reinit bias for cls
        prior_prob = 0.02
        pi = - math.log((1 - prior_prob) / prior_prob)
        self.cls_pred_bias = X.var("cls_pred_bias", init=X.constant(pi))
        self.anchor_dict = None

    def get_anchor(self):
        p = self.p

        num_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        stride = p.anchor_generate.stride

        anchor_dict = {}
        for s in stride:
            max_side = p.anchor_generate.max_side // s
            anchors = X.var("anchor_stride%s" % s,
                            shape=(1, 1, max_side, max_side, num_anchor * 4),
                            dtype='float32')  # (1, 1, long_side, long_side, #anchor * 4)
            anchor_dict["stride%s" % s] = anchors

        self.anchor_dict = anchor_dict

    def get_loss(self, conv_feat, gt_bbox, im_info):
        import mxnet as mx

        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        num_class = p.num_class
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        image_per_device = p.batch_image

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)
        cls_logit_reshape_list = []
        bbox_delta_reshape_list = []
        feat_list = []

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # reshape logit and delta
        for s in stride:
            # (N, A * C, H, W) -> (N, A * C, H * W)
            cls_logit = X.reshape(
                data=cls_logit_dict["stride%s" % s],
                shape=(0, 0, -1),
                name="cls_stride%s_reshape" % s
            )

            # (N, A * 4, H, W) -> (N, A * 4, H * W)
            bbox_delta = X.reshape(
                data=bbox_delta_dict["stride%s" % s],
                shape=(0, 0, -1),
                name="bbox_stride%s_reshape" % s
            )

            cls_logit_reshape_list.append(cls_logit)
            bbox_delta_reshape_list.append(bbox_delta)
            feat_list.append(cls_logit_dict["stride%s" % s])

        # cls_logits -> (N, H' * W' * A, C)
        cls_logits = X.concat(cls_logit_reshape_list, axis=2, name="cls_logit_concat")
        cls_logits = X.transpose(cls_logits, axes=(0, 2, 1), name="cls_logit_transpose")
        cls_logits = X.reshape(cls_logits, shape=(0, -1, num_class - 1), name="cls_logit_reshape")
        cls_prob = X.sigmoid(cls_logits)
        # bbox_deltas -> (N, H' * W' * A, 4)
        bbox_deltas = X.concat(bbox_delta_reshape_list, axis=2, name="bbox_delta_concat")
        bbox_deltas = X.transpose(bbox_deltas, axes=(0, 2, 1), name="bbox_delta_transpose")
        bbox_deltas = X.reshape(bbox_deltas, shape=(0, -1, 4), name="bbox_delta_reshape")

        anchor_list = [self.anchor_dict["stride%s" % s] for s in stride]
        bbox_thr = p.anchor_assign.bbox_thr
        pre_anchor_top_n = p.anchor_assign.pre_anchor_top_n
        alpha = p.focal_loss.alpha
        gamma = p.focal_loss.gamma
        anchor_target_mean = p.head.mean or (0, 0, 0, 0)
        anchor_target_std = p.head.std or (1, 1, 1, 1)

        from models.FreeAnchor.ops import _prepare_anchors, _positive_loss, _negative_loss
        anchors = _prepare_anchors(
            mx.sym, feat_list, anchor_list, image_per_device, num_base_anchor)

        positive_loss = _positive_loss(
            mx.sym, anchors, gt_bbox, cls_prob, bbox_deltas, image_per_device,
            alpha, pre_anchor_top_n, anchor_target_mean, anchor_target_std
        )
        positive_loss = X.make_loss(
            data=positive_loss,
            grad_scale=1.0 * scale_loss_shift,
            name="positive_loss"
        )

        negative_loss = _negative_loss(
            mx.sym, anchors, gt_bbox, cls_prob, bbox_deltas, im_info, image_per_device,
            num_class, alpha, gamma, pre_anchor_top_n, bbox_thr,
            anchor_target_mean, anchor_target_std
        )
        negative_loss = X.make_loss(
            data=negative_loss,
            grad_scale=1.0 * scale_loss_shift,
            name="negative_loss"
        )

        return positive_loss, negative_loss

    def get_prediction(self, conv_feat, im_info):
        import mxnet as mx
        p = self.p
        num_class = p.num_class
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        pre_nms_top_n = p.proposal.pre_nms_top_n
        anchor_target_mean = p.head.mean or (0, 0, 0, 0)
        anchor_target_std = p.head.std or (1, 1, 1, 1)

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_feat)

        from models.FreeAnchor.ops import _proposal_retina
        cls_score_list = []
        bbox_xyxy_list = []

        for s in stride:
            cls_prob = X.sigmoid(data=cls_logit_dict["stride%s" % s])
            bbox_delta = bbox_delta_dict["stride%s" % s]
            anchors = self.anchor_dict["stride%s" % s]

            pre_nms_top_n_level = -1 if s == max(stride) else pre_nms_top_n
            bbox_xyxy, cls_score = _proposal_retina(
                F=mx.sym,
                cls_prob=cls_prob,
                bbox_pred=bbox_delta,
                anchors=anchors,
                im_info=im_info,
                batch_size=1,
                rpn_pre_nms_top_n=pre_nms_top_n_level,
                num_class=num_class,
                anchor_mean=anchor_target_mean,
                anchor_std=anchor_target_std
            )

            cls_score_list.append(cls_score)
            bbox_xyxy_list.append(bbox_xyxy)
            cls_score = X.concat(cls_score_list, axis=1, name="cls_score_concat")
            bbox_xyxy = X.concat(bbox_xyxy_list, axis=1, name="bbox_xyxy_concat")

        return cls_score, bbox_xyxy
