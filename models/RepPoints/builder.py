from __future__ import division
from __future__ import print_function

import mxnext as X
import math
import mxnet as mx
from utils.patch_config import patch_config_as_nothrow


class RepPoints(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head):

        label = X.var("gt_bbox")

        feat = backbone.get_rpn_feature()
        feat = neck.get_rpn_feature(feat)
        loss = head.get_loss(feat, label)

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


class RepPointsHead(object):
    def __init__(self, pHead):
        self.p = patch_config_as_nothrow(pHead)
        num_points = self.p.point_generate.num_points
        self.dcn_kernel = int(math.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            "The points number should be square."
        assert self.dcn_kernel % 2 == 1, "The dcn kernel size should be odd."

        # init moment method
        dtype = "float16" if self.p.fp16 else "float32"
        self.moment_transfer = X.var(
            name="moment_transfer", shape=(2,), init=X.zero_init(), lr_mult=0.01, dtype=dtype
        )

        # init bias for cls
        prior_prob = 0.01
        pi = -math.log((1 - prior_prob) / prior_prob)

        # shared classification weight and bias
        self.cls_conv1_weight = X.var("cls_conv1_weight", init=X.gauss(std=0.01))
        self.cls_conv1_bias = X.var("cls_conv1_bias", init=X.zero_init())
        self.cls_conv2_weight = X.var("cls_conv2_weight", init=X.gauss(std=0.01))
        self.cls_conv2_bias = X.var("cls_conv2_bias", init=X.zero_init())
        self.cls_conv3_weight = X.var("cls_conv3_weight", init=X.gauss(std=0.01))
        self.cls_conv3_bias = X.var("cls_conv3_bias", init=X.zero_init())
        self.cls_conv_weight = X.var("cls_conv_weight", init=X.gauss(std=0.01))
        self.cls_conv_bias = X.var("cls_conv_bias", init=X.zero_init())
        self.cls_out_weight = X.var("cls_out_weight", init=X.gauss(std=0.01))
        self.cls_out_bias = X.var("cls_out_bias", init=X.constant(pi))

        # shared regression weight and bias
        self.reg_conv1_weight = X.var("reg_conv1_weight", init=X.gauss(std=0.01))
        self.reg_conv1_bias = X.var("reg_conv1_bias", init=X.zero_init())
        self.reg_conv2_weight = X.var("reg_conv2_weight", init=X.gauss(std=0.01))
        self.reg_conv2_bias = X.var("reg_conv2_bias", init=X.zero_init())
        self.reg_conv3_weight = X.var("reg_conv3_weight", init=X.gauss(std=0.01))
        self.reg_conv3_bias = X.var("reg_conv3_bias", init=X.zero_init())
        self.pts_init_conv_weight = X.var("pts_init_conv_weight", init=X.gauss(std=0.01))
        self.pts_init_conv_bias = X.var("pts_init_conv_bias", init=X.zero_init())
        self.pts_init_out_weight = X.var("pts_init_out_weight", init=X.gauss(std=0.01))
        self.pts_init_out_bias = X.var("pts_init_out_bias", init=X.zero_init())
        self.pts_refine_conv_weight = X.var("pts_refine_conv_weight", init=X.gauss(std=0.01))
        self.pts_refine_conv_bias = X.var("pts_refine_conv_bias", init=X.zero_init())
        self.pts_refine_out_weight = X.var("pts_refine_out_weight", init=X.gauss(std=0.01))
        self.pts_refine_out_bias = X.var("pts_refine_out_bias", init=X.zero_init())

        self._pts_out_inits = None
        self._pts_out_refines = None
        self._cls_outs = None

    def _cls_subnet(self, conv_feat, stride):
        p = self.p
        norm = p.normalizer
        conv_channel = p.head.conv_channel

        # classification subset
        cls_conv1 = X.conv(
            data=conv_feat,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv1_weight,
            bias=self.cls_conv1_bias,
            no_bias=False,
            name="cls_conv1"
        )
        cls_conv1 = norm(cls_conv1, name="cls_conv1_bn_s{}".format(stride))
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
        cls_conv2 = norm(cls_conv2, name="cls_conv2_bn_s{}".format(stride))
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
        cls_conv3 = norm(cls_conv3, name="cls_conv3_bn_s{}".format(stride))
        cls_conv3_relu = X.relu(cls_conv3)

        if p.fp16:
            cls_conv3_relu = X.to_fp32(cls_conv3_relu, name="cls_conv3_fp32")

        return cls_conv3_relu

    def _reg_subnet(self, conv_feat, stride):
        p = self.p
        norm = p.normalizer
        conv_channel = p.head.conv_channel

        # regression subnet
        reg_conv1 = X.conv(
            data=conv_feat,
            kernel=3,
            filter=conv_channel,
            weight=self.reg_conv1_weight,
            bias=self.reg_conv1_bias,
            no_bias=False,
            name="reg_conv1"
        )
        reg_conv1 = norm(reg_conv1, name="reg_conv1_bn_s{}".format(stride))
        reg_conv1_relu = X.relu(reg_conv1)
        reg_conv2 = X.conv(
            data=reg_conv1_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.reg_conv2_weight,
            bias=self.reg_conv2_bias,
            no_bias=False,
            name="reg_conv2"
        )
        reg_conv2 = norm(reg_conv2, name="reg_conv2_bn_s{}".format(stride))
        reg_conv2_relu = X.relu(reg_conv2)
        reg_conv3 = X.conv(
            data=reg_conv2_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.reg_conv3_weight,
            bias=self.reg_conv3_bias,
            no_bias=False,
            name="reg_conv3"
        )
        reg_conv3 = norm(reg_conv3, name="reg_conv3_bn_s{}".format(stride))
        reg_conv3_relu = X.relu(reg_conv3)

        if p.fp16:
            reg_conv3_relu = X.to_fp32(reg_conv3_relu, name="reg_conv3_fp32")

        return reg_conv3_relu

    def _init_pts(self, reg_feat):
        p = self.p
        point_conv_channel = p.head.point_conv_channel
        pts_output_channel = p.point_generate.num_points * 2

        pts_init_conv = X.conv(
            data=reg_feat,
            kernel=3,
            filter=point_conv_channel,
            weight=self.pts_init_conv_weight,
            bias=self.pts_init_conv_bias,
            no_bias=False,
            name="pts_init_conv"
        )
        pts_init_conv_relu = X.relu(pts_init_conv)
        pts_init_out = X.conv(
            data=pts_init_conv_relu,
            kernel=1,
            filter=pts_output_channel,
            weight=self.pts_init_out_weight,
            bias=self.pts_init_out_bias,
            no_bias=False,
            name="pts_init_out"
        )

        return pts_init_out

    def _refine_pts(self, cls_feat, reg_feat, dcn_offset, pts_init_out):
        p = self.p
        point_conv_channel = p.head.point_conv_channel
        num_class = p.num_class
        output_channel = num_class - 1
        pts_output_channel = p.point_generate.num_points * 2

        cls_conv = mx.symbol.contrib.DeformableConvolution(
            data=cls_feat,
            offset=dcn_offset,
            kernel=(self.dcn_kernel, self.dcn_kernel),
            pad=(self.dcn_pad, self.dcn_pad),
            stride=(1, 1),
            dilate=(1, 1),
            num_filter=point_conv_channel,
            weight=self.cls_conv_weight,
            bias=self.cls_conv_bias,
            no_bias=False,
            name="cls_conv"
        )
        cls_conv_relu = X.relu(cls_conv)
        cls_out = X.conv(
            data=cls_conv_relu,
            kernel=1,
            filter=output_channel,
            weight=self.cls_out_weight,
            bias=self.cls_out_bias,
            no_bias=False,
            name="cls_out"
        )

        pts_refine_conv = mx.symbol.contrib.DeformableConvolution(
            data=reg_feat,
            offset=dcn_offset,
            kernel=(self.dcn_kernel, self.dcn_kernel),
            pad=(self.dcn_pad, self.dcn_pad),
            stride=(1, 1),
            dilate=(1, 1),
            num_filter=point_conv_channel,
            weight=self.pts_refine_conv_weight,
            bias=self.pts_refine_conv_bias,
            no_bias=False,
            name="pts_refine_conv"
        )
        pts_refine_conv_relu = X.relu(pts_refine_conv)
        pts_refine_out = X.conv(
            data=pts_refine_conv_relu,
            kernel=1,
            filter=pts_output_channel,
            weight=self.pts_refine_out_weight,
            bias=self.pts_refine_out_bias,
            no_bias=False,
            name="pts_refine_out"
        )
        pts_refine_out = pts_refine_out + X.block_grad(pts_init_out)
        return pts_refine_out, cls_out

    def get_output(self, conv_feat):
        if self._pts_out_inits is not None and self._pts_out_refines is not None and \
                self._cls_outs is not None:
            return self._pts_out_inits, self._pts_out_refines, self._cls_outs

        p = self.p
        stride = p.point_generate.stride
        # init base offset for dcn
        from models.RepPoints.point_ops import _gen_offsets
        dcn_base_offset = _gen_offsets(
            mx.symbol, dcn_kernel=self.dcn_kernel, dcn_pad=self.dcn_pad
        )

        pts_out_inits = dict()
        pts_out_refines = dict()
        cls_outs = dict()

        for s in stride:
            # cls subnet with shared params across multiple strides
            cls_feat = self._cls_subnet(conv_feat=conv_feat["stride%s" % s], stride=s)
            # reg subnet with shared params across multiple strides
            reg_feat = self._reg_subnet(conv_feat=conv_feat["stride%s" % s], stride=s)
            # predict offsets on each center points
            pts_out_init = self._init_pts(reg_feat)
            # grad multiples 0.1 for offsets subnet
            pts_out_init_grad_mul = 0.9 * X.block_grad(pts_out_init) + 0.1 * pts_out_init
            # dcn uses offsets on grids as input,
            # thus the predicted offsets substract base dcn offsets here before using dcn.
            pts_out_init_offset = mx.symbol.broadcast_sub(pts_out_init_grad_mul, dcn_base_offset)
            # use offsets on features to refine box and cls
            pts_out_refine, cls_out = self._refine_pts(
                cls_feat,
                reg_feat,
                pts_out_init_offset,
                pts_out_init
            )
            pts_out_inits["stride%s" % s] = pts_out_init
            pts_out_refines["stride%s" % s] = pts_out_refine
            cls_outs["stride%s" % s] = cls_out

        self._pts_out_inits = pts_out_inits
        self._pts_out_refines = pts_out_refines
        self._cls_outs = cls_outs

        return self._pts_out_inits, self._pts_out_refines, self._cls_outs

    def get_loss(self, conv_feat, gt_bbox):
        from models.RepPoints.point_ops import (
            _gen_points, _offset_to_pts, _point_target, _offset_to_boxes, _points2bbox)
        p = self.p
        batch_image = p.batch_image
        num_points = p.point_generate.num_points
        scale = p.point_generate.scale
        stride = p.point_generate.stride
        transform = p.point_generate.transform
        target_scale = p.point_target.target_scale
        num_pos = p.point_target.num_pos
        pos_iou_thr = p.bbox_target.pos_iou_thr
        neg_iou_thr = p.bbox_target.neg_iou_thr
        min_pos_iou = p.bbox_target.min_pos_iou

        pts_out_inits, pts_out_refines, cls_outs = self.get_output(conv_feat)

        points = dict()
        bboxes = dict()
        pts_coordinate_preds_inits = dict()
        pts_coordinate_preds_refines = dict()
        for s in stride:
            # generate points on base coordinate according to stride and size of feature map
            points["stride%s" % s] = _gen_points(mx.symbol, pts_out_inits["stride%s" % s], s)
            # generate bbox after init stage
            bboxes["stride%s" % s] = _offset_to_boxes(
                mx.symbol,
                points["stride%s" % s],
                X.block_grad(pts_out_inits["stride%s" % s]),
                s,
                transform,
                moment_transfer=self.moment_transfer
            )
            # generate final offsets in init stage
            pts_coordinate_preds_inits["stride%s" % s] = _offset_to_pts(
                mx.symbol,
                points["stride%s" % s],
                pts_out_inits["stride%s" % s],
                s,
                num_points
            )
            # generate final offsets in refine stage
            pts_coordinate_preds_refines["stride%s" % s] = _offset_to_pts(
                mx.symbol,
                points["stride%s" % s],
                pts_out_refines["stride%s" % s],
                s,
                num_points
            )

        # for init stage, use points assignment
        point_proposals = mx.symbol.tile(
            X.concat([points["stride%s" % s] for s in stride], axis=1, name="point_concat"),
            reps=(batch_image, 1, 1)
        )
        points_labels_init, points_gts_init, points_weight_init = _point_target(
            mx.symbol,
            point_proposals,
            gt_bbox,
            batch_image,
            "point",
            scale=target_scale,
            num_pos=num_pos
        )
        # for refine stage, use max iou assignment
        box_proposals = X.concat(
            [bboxes["stride%s" % s] for s in stride], axis=1, name="box_concat"
        )
        points_labels_refine, points_gts_refine, points_weight_refine = _point_target(
            mx.symbol,
            box_proposals,
            gt_bbox,
            batch_image,
            "box",
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
            min_pos_iou=min_pos_iou
        )

        bboxes_out_strides = dict()
        for s in stride:
            cls_outs["stride%s" % s] = X.reshape(
                X.transpose(data=cls_outs["stride%s" % s], axes=(0, 2, 3, 1)),
                (0, -3, -2)
            )
            bboxes_out_strides["stride%s" % s] = mx.symbol.repeat(mx.symbol.ones_like(
                mx.symbol.slice_axis(cls_outs["stride%s" % s], begin=0, end=1, axis=-1)),
                repeats=4, axis=-1) * s

        # cls branch
        cls_outs_concat = X.concat(
            [cls_outs["stride%s" % s] for s in stride], axis=1, name="cls_concat"
        )
        cls_loss = X.focal_loss(
            data=cls_outs_concat,
            label=points_labels_refine,
            normalization='valid',
            alpha=p.focal_loss.alpha,
            gamma=p.focal_loss.gamma,
            grad_scale=1.0,
            workspace=1500,
            name="cls_loss"
        )

        # init box branch
        pts_inits_concat_ = X.concat(
            [pts_coordinate_preds_inits["stride%s" % s] for s in stride],
            axis=1,
            name="pts_init_concat_"
        )
        pts_inits_concat = X.reshape(pts_inits_concat_, (-3, -2), name="pts_inits_concat")
        bboxes_inits_concat_ = _points2bbox(
            mx.symbol,
            pts_inits_concat,
            transform,
            y_first=False,
            moment_transfer=self.moment_transfer
        )
        bboxes_inits_concat = X.reshape(bboxes_inits_concat_, (-4, batch_image, -1, -2))
        normalize_term = X.concat(
            [bboxes_out_strides["stride%s" % s] for s in stride], axis=1, name="normalize_term"
        ) * scale
        pts_init_loss = X.smooth_l1(
            data=(bboxes_inits_concat - points_gts_init) / normalize_term,
            scalar=3.0,
            name="pts_init_l1_loss"
        )
        pts_init_loss = pts_init_loss * points_weight_init
        pts_init_loss = X.bbox_norm(
            data=pts_init_loss,
            label=points_labels_init,
            name="pts_init_norm_loss"
        )
        pts_init_loss = X.make_loss(
            data=pts_init_loss,
            grad_scale=0.5,
            name="pts_init_loss"
        )
        points_init_labels = X.block_grad(points_labels_refine, name="points_init_labels")

        # refine box branch
        pts_refines_concat_ = X.concat(
            [pts_coordinate_preds_refines["stride%s" % s] for s in stride],
            axis=1,
            name="pts_refines_concat_"
        )
        pts_refines_concat = X.reshape(pts_refines_concat_, (-3, -2), name="pts_refines_concat")
        bboxes_refines_concat_ = _points2bbox(
            mx.symbol,
            pts_refines_concat,
            transform,
            y_first=False,
            moment_transfer=self.moment_transfer
        )
        bboxes_refines_concat = X.reshape(bboxes_refines_concat_, (-4, batch_image, -1, -2))
        pts_refine_loss = X.smooth_l1(
            data=(bboxes_refines_concat - points_gts_refine) / normalize_term,
            scalar=3.0,
            name="pts_refine_l1_loss"
        )
        pts_refine_loss = pts_refine_loss * points_weight_refine
        pts_refine_loss = X.bbox_norm(
            data=pts_refine_loss,
            label=points_labels_refine,
            name="pts_refine_norm_loss"
        )
        pts_refine_loss = X.make_loss(
            data=pts_refine_loss,
            grad_scale=1.0,
            name="pts_refine_loss"
        )
        points_refine_labels = X.block_grad(points_labels_refine, name="point_refine_labels")

        return cls_loss, pts_init_loss, pts_refine_loss, points_init_labels, points_refine_labels

    def get_prediction(self, conv_feat, im_info):
        from models.RepPoints.point_ops import _gen_points, _points2bbox
        p = self.p
        batch_image = p.batch_image
        stride = p.point_generate.stride
        transform = p.point_generate.transform
        pre_nms_top_n = p.proposal.pre_nms_top_n

        pts_out_inits, pts_out_refines, cls_outs = self.get_output(conv_feat)

        cls_score_dict = dict()
        bbox_xyxy_dict = dict()
        for s in stride:
            # NOTE: pre_nms_top_n_ is hard-coded as -1 because the number of proposals is less
            # than pre_nms_top_n in these low-resolution feature maps. Also note that one should
            # select the appropriate params here if using low-resolution images as input.
            pre_nms_top_n_ = pre_nms_top_n if s <= 32 else -1
            points_ = _gen_points(mx.symbol, pts_out_inits["stride%s" % s], s)
            preds_refines_ = _points2bbox(
                mx.symbol,
                pts_out_refines["stride%s" % s],
                transform,
                moment_transfer=self.moment_transfer
            )
            preds_refines_ = X.reshape(
                X.transpose(data=preds_refines_, axes=(0, 2, 3, 1)),
                (0, -3, -2)
            )
            cls_ = X.reshape(
                X.transpose(data=cls_outs["stride%s" % s], axes=(0, 2, 3, 1)),
                (0, -3, -2)
            )
            scores_ = X.sigmoid(cls_)
            max_scores_ = mx.symbol.max(scores_, axis=-1)
            max_index_ = mx.symbol.topk(max_scores_, axis=1, k=pre_nms_top_n_)
            scores_dict = dict()
            bboxes_dict = dict()
            for i in range(batch_image):
                max_index_i = X.reshape(
                    mx.symbol.slice_axis(max_index_, axis=0, begin=i, end=i + 1), (-1,)
                )
                scores_i = X.reshape(
                    mx.symbol.slice_axis(scores_, axis=0, begin=i, end=i + 1), (-3, -2)
                )
                points_i = X.reshape(points_, (-3, -2))
                preds_refines_i = X.reshape(
                    mx.symbol.slice_axis(preds_refines_, axis=0, begin=i, end=i + 1), (-3, -2)
                )
                scores_i = mx.symbol.take(scores_i, max_index_i)
                points_i = mx.symbol.take(points_i, max_index_i)
                preds_refines_i = mx.symbol.take(preds_refines_i, max_index_i)
                points_i = mx.symbol.slice_axis(points_i, axis=-1, begin=0, end=2)
                points_xyxy_i = X.concat(
                    [points_i, points_i], axis=-1, name="points_xyxy_b{}_s{}".format(i, s)
                )
                bboxes_i = preds_refines_i * s + points_xyxy_i
                im_info_i = mx.symbol.slice_axis(im_info, axis=0, begin=i, end=i + 1)
                h_i, w_i, _ = mx.symbol.split(im_info_i, num_outputs=3, axis=1)
                l_i, t_i, r_i, b_i = mx.symbol.split(bboxes_i, num_outputs=4, axis=1)
                clip_l_i = mx.symbol.maximum(mx.symbol.broadcast_minimum(l_i, w_i - 1.0), 0.0)
                clip_t_i = mx.symbol.maximum(mx.symbol.broadcast_minimum(t_i, h_i - 1.0), 0.0)
                clip_r_i = mx.symbol.maximum(mx.symbol.broadcast_minimum(r_i, w_i - 1.0), 0.0)
                clip_b_i = mx.symbol.maximum(mx.symbol.broadcast_minimum(b_i, h_i - 1.0), 0.0)
                clip_bboxes_i = X.concat(
                    [clip_l_i, clip_t_i, clip_r_i, clip_b_i],
                    axis=1,
                    name="clip_bboxes_b{}_s{}".format(i, s)
                )
                scores_dict["img%s" % i] = scores_i
                bboxes_dict["img%s" % i] = clip_bboxes_i
            cls_score_ = mx.symbol.stack(
                *[scores_dict["img%s" % i] for i in range(batch_image)], axis=0
            )
            pad_zeros_ = mx.symbol.zeros_like(
                mx.symbol.slice_axis(cls_score_, axis=-1, begin=0, end=1)
            )
            cls_score_ = X.concat([pad_zeros_, cls_score_], axis=-1, name="cls_score_s{}".format(s))
            bboxes_ = mx.symbol.stack(
                *[bboxes_dict["img%s" % i] for i in range(batch_image)], axis=0
            )
            cls_score_dict["stride%s" % s] = cls_score_
            bbox_xyxy_dict["stride%s" % s] = bboxes_

        cls_score = X.concat(
            [cls_score_dict["stride%s" % s] for s in stride], axis=1, name="cls_score_concat"
        )
        bbox_xyxy = X.concat(
            [bbox_xyxy_dict["stride%s" % s] for s in stride], axis=1, name="bbox_xyxy_concat"
        )

        return cls_score, bbox_xyxy


class RepPointsNeck(object):
    def __init__(self, pNeck):
        self.p = patch_config_as_nothrow(pNeck)
        self.fpn_feat = None

    def add_norm(self, sym):
        p = self.p
        if p.normalizer.__name__ == "fix_bn":
            pass
        elif p.normalizer.__name__ in ["sync_bn", "local_bn", "gn", "dummy"]:
            sym = p.normalizer(sym)
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))
        return sym

    def get_fpn_neck(self, data):
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

        # P6
        p6 = X.conv(
            data=p5_conv,
            kernel=3,
            stride=2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P6_conv_weight", init=xavier_init),
            bias=X.var(name="P6_conv_bias", init=X.zero_init()),
            name="P6_conv"
        )
        p6 = self.add_norm(p6)

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
        p7 = self.add_norm(p7)

        self.fpn_feat = dict(
            stride8=p3_conv,
            stride16=p4_conv,
            stride32=p5_conv,
            stride64=p6,
            stride128=p7
        )

        return self.fpn_feat

    def get_rpn_feature(self, rpn_feat):
        return self.get_fpn_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.get_fpn_neck(rcnn_feat)
