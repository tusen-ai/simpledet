import math
import mxnet as mx
from mxnext.tvm.decode_bbox import decode_bbox


def _bbox_encode(F, ex_rois, gt_rois, means=[0., 0., 0., 0.], stds=[1., 1., 1., 1.]):
    """ decode bbox
    inputs:
        F: symbol or ndarray
        ex_rois: F, (#img, ..., 4)
        gt_rois: F, (#img, ..., 4)
        means: list, (4,)
        stds: list, (4,)
    outputs:
        targets: symbol or ndarray, (#img, ..., 4)
    """

    ex_x1, ex_y1, ex_x2, ex_y2 = F.split(ex_rois, num_outputs=4, axis=-1)
    gt_x1, gt_y1, gt_x2, gt_y2 = F.split(gt_rois, num_outputs=4, axis=-1)

    ex_widths = ex_x2 - ex_x1 + 1.0
    ex_heights = ex_y2 - ex_y1 + 1.0
    ex_ctr_x = ex_x1 + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_y1 + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_x2 - gt_x1 + 1.0
    gt_heights = gt_y2 - gt_y1 + 1.0
    gt_ctr_x = gt_x1 + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_y1 + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = F.log(gt_widths / ex_widths)
    targets_dh = F.log(gt_heights / ex_heights)

    targets_dx = (targets_dx - means[0]) / stds[0]
    targets_dy = (targets_dy - means[1]) / stds[1]
    targets_dw = (targets_dw - means[2]) / stds[2]
    targets_dh = (targets_dh - means[3]) / stds[3]

    return F.concat(targets_dx, targets_dy, targets_dw, targets_dh, dim=-1)


def _prepare_anchors(F, feat_list, anchor_list, num_image, num_anchor):
    """ crop pre-comuputed anchors into the shape of the features
    inputs:
        F: symbol or ndarray
        feat_list: list of symbols or ndarrays, [(#img, #c, #h1, #w1), ...]
        anchor_list: list of symbols or ndarrays, [(1, 1, #h1', #w1', #anchor * 4), ...]
        num_image: int
        num_anchor: int
    outputs:
        anchors: symbol or ndarray, (#img, H * W * #anchor, 4)
    """
    lvl_anchors = []
    for features, anchors in zip(feat_list, anchor_list):
        anchors = F.slice_like(anchors, features, axes=(2, 3))  # (1, 1, h, w, #anchor * 4)
        anchors = F.reshape(anchors, shape=(1, -1, num_anchor * 4))  # (1, h * w, #anchor * 4)
        lvl_anchors.append(anchors)
    anchors = F.concat(*lvl_anchors, dim=1)  # (1, H * W, #anchor * 4)
    anchors = F.reshape(anchors, shape=(0, -1, 4))  # (1, H * W * #anchor, 4)
    anchors = F.broadcast_axis(anchors, axis=0, size=num_image)  # (#img, H * W * #anchor, 4)

    return anchors


def _positive_loss(F, anchors, gt_bboxes, cls_prob, bbox_pred, num_image,
                   alpha, pre_anchor_top_n, target_means, target_stds):
    """
    inputs:
        F: symbol or ndarray
        anchors: F, (#img, h * w * #anchor, 4)
        gt_bboxes: F, (#img, #gt, 5)
        cls_prob: F, (#img, h * w * #anchor, #class)
        bbox_pred: F, (#img, h * w * #anchor, 4)
        alpha: float
        pre_anchor_top_n: int
        target_means: list, (4,)
        target_stds: list, (4,)
    outputs:
        positive_loss, F, (1,)
    """

    with mx.name.Prefix("free_anchor_pos_target: "):
        gt_labels = F.slice_axis(gt_bboxes, axis=-1, begin=4, end=5)  # (n, #gt, 1)
        gt_bboxes = F.slice_axis(gt_bboxes, axis=-1, begin=0, end=4)  # (n, #gt, 4)

        # assigning
        an_index_list = list()
        for i in range(num_image):
            anchors_this = F.slice_axis(anchors, axis=0, begin=i, end=i + 1).reshape([-1, 4])
            gt_bboxes_this = F.slice_axis(gt_bboxes, axis=0, begin=i, end=i + 1).reshape([-1, 4])
            iou_gt2a_this = F.contrib.box_iou(
                gt_bboxes_this, anchors_this, format="corner")  # (#gt, h * w * #anchor)
            iou_gt2a_topk_this = F.topk(iou_gt2a_this, axis=1, k=pre_anchor_top_n)  # (#gt, top_n)
            an_index_list.append(iou_gt2a_topk_this)
        anchor_index = F.stack(*an_index_list, axis=0)  # (n, #gt, top_n)
        batch_index = F.arange(0, num_image).reshape((num_image, 1, 1))  # (n, 1, 1)
        batch_index = F.broadcast_like(batch_index, anchor_index)  # (n, #gt, top_n)
        gt_index = F.maximum(gt_labels - 1, 0).reshape((0, 0, 1))  # (n, #gt, 1)
        gt_index = F.broadcast_like(gt_index, anchor_index)  # (n, #gt, top_n)

        # matched cls
        cls_index = F.stack(*[batch_index, anchor_index, gt_index], axis=0)  # (3, n, #gt, top_n)
        matched_cls_prob = F.gather_nd(cls_prob, cls_index)  # (n, #gt, top_n)

        # matched bbox
        bbox_index = F.stack(*[batch_index, anchor_index], axis=0)  # (2, n, #gt, top_n)
        matched_bbox_pred = F.gather_nd(bbox_pred, bbox_index)  # (n, #gt, top_n, 4)
        matched_anchors = F.gather_nd(anchors, bbox_index)  # (n, #gt, top_n, 4)
        matched_gt_bboxes = F.reshape(gt_bboxes, (0, 0, -1, 4))  # (n, #gt, 1, 4)
        matched_gt_bboxes = F.broadcast_like(
            matched_gt_bboxes, matched_anchors)  # (n, #gt, top_n, 4)
        bbox_targets = _bbox_encode(
            F, matched_anchors, matched_gt_bboxes, target_means, target_stds)  # (n, #gt, top_n, 4)
        scalar = 0.11
        bbox_loss_weight = 0.75
        bbox_loss = F.smooth_l1(
            matched_bbox_pred - bbox_targets, scalar=math.sqrt(1 / scalar))  # (n, #gt, top_n, 4)
        bbox_loss = bbox_loss * bbox_loss_weight  # (n, #gt, top_n, 4)
        matched_box_prob = F.exp(-F.sum(bbox_loss, axis=-1))  # (n, #gt, top_n)

        # positive part of the loss
        matched_prob = matched_cls_prob * matched_box_prob  # (n, #gt, top_n)
        valid = (gt_labels > 0).reshape((0, 0, 1))  # (n, #gt, 1)
        valid = F.broadcast_like(valid, matched_prob)  # (n, #gt, top_n)
        matched_prob = F.where(valid, matched_prob, F.ones_like(matched_prob))  # (n, #gt, top_n)
        prob_weight = 1. / F.maximum(1. - matched_prob, 1e-12)  # (n, #gt, top_n)
        prob_weight = F.broadcast_div(prob_weight, F.sum(
            prob_weight, axis=-1, keepdims=True))  # (n, #gt, top_n)
        bag_prob = F.sum(prob_weight * matched_prob, axis=-1)  # (n, #gt)
        positive_loss = - alpha * F.log(F.clip(bag_prob, 1e-12, 1.))  # (n, #gt)
        positive_loss = F.broadcast_div(
            positive_loss, F.maximum(F.sum(gt_labels > 0), 1))  # (n, #gt)
        positive_loss = F.sum(positive_loss)

        return positive_loss


def _negative_loss(F, anchors, gt_bboxes, cls_prob, bbox_pred, im_infos, num_image,
                   num_class, alpha, gamma, pre_anchor_top_n, bbox_thr, target_means, target_stds):
    """
    inputs:
        F: symbol or ndarray
        anchors: F, (#img, H * W * #anchor, 4)
        gt_bboxes: F, (#img, #gt, 5)
        cls_prob: F, (#img, H * W * #anchor, #class)
        bbox_pred: F, (#img, H * W * #anchor, 4)
        im_infos: F, (#img, 3)
        num_image: int
        num_class: int
        alpha: float
        gamma: float
        pre_anchor_top_n: int
        bbox_thr: float
        target_means: list, (4,)
        target_stds: list, (4,)
    outputs:
        negative_loss: F, (1,)
    """

    with mx.name.Prefix("free_anchor_neg_target: "):
        gt_labels = F.slice_axis(gt_bboxes, axis=-1, begin=4, end=5)  # (n, #gt, 1)
        gt_bboxes = F.slice_axis(gt_bboxes, axis=-1, begin=0, end=4)  # (n, #gt, 4)

        pred_bboxes = decode_bbox(F, anchors, bbox_pred, im_infos,
                                  target_means, target_stds, True)  # (n, h * w * anchor, 4)
        iou_gt2pred_list = list()
        for i in range(num_image):
            gt_bboxes_this = F.slice_axis(gt_bboxes, axis=0, begin=i, end=i + 1).reshape([-1, 4])
            pred_bboxes_this = F.slice_axis(
                pred_bboxes, axis=0, begin=i, end=i + 1).reshape([-1, 4])
            iou_gt2pred_this = F.contrib.box_iou(
                gt_bboxes_this, pred_bboxes_this, format="corner")  # (#gt, h * w * #anchor)
            iou_gt2pred_list.append(iou_gt2pred_this)
        iou_gt2pred = F.stack(*iou_gt2pred_list, axis=0)  # (n, #gt, h * w * #anchor)

        # select positive boxes after decoding
        t1 = bbox_thr
        t2 = F.maximum(F.max(iou_gt2pred, axis=2, keepdims=True), t1 + 1e-12)  # (n, #gt, 1)
        gt_pred_prob = F.clip(F.broadcast_div(iou_gt2pred - t1, t2 - t1),
                              a_min=0., a_max=1.)  # (n, #gt, h * w * #anchor)

        # box prob
        gt_index = F.argmax(gt_pred_prob, axis=1)  # (n, h * w * #anchor)
        batch_index = F.arange(0, num_image).reshape((num_image, 1))  # (n, 1)
        batch_index = F.broadcast_like(batch_index, gt_index)  # (n, h * w * #anchor)
        gt_labels_index = F.stack(*[batch_index, gt_index])  # (2, n, h * w * #anchor)
        gt_labels_reshape = gt_labels.reshape((0, -1))  # (n, #gt)
        gt_labels_gather = F.gather_nd(gt_labels_reshape, gt_labels_index)  # (n, h * w * #anchor)
        cls_index = gt_labels_gather - 1  # (n, h * w * #anchor)
        one_hot = F.one_hot(cls_index, depth=num_class - 1)  # (n, h * w * #anchor, #class)
        gt_pred_prob_gather = F.max(gt_pred_prob, axis=1).reshape(
            (0, 0, 1))  # (n, h * w * #anchor, 1)
        box_prob = F.broadcast_mul(one_hot, gt_pred_prob_gather)  # (n, h * w * #anchor, #class)
        box_prob = F.BlockGrad(box_prob)

        # negative part of the loss
        prob = cls_prob * (1. - box_prob)  # (n, h * w * #anchor, #class)
        valid = (gt_labels_gather > 0).reshape((0, 0, 1))  # (n, h * w * #anchor, 1)
        valid = F.broadcast_like(valid, prob)  # (n, h * w * #anchor, #class)
        prob = F.where(valid, prob, F.zeros_like(prob))  # (n, h * w * #anchor, #class)
        negative_loss = - prob ** gamma * \
            F.log(F.clip(1. - prob, 1e-12, 1.))  # (n, h * w * #anchor, #class)
        negative_loss = (1. - alpha) * negative_loss
        negative_loss = F.broadcast_div(negative_loss, (F.maximum(
            F.sum(gt_labels > 0) * pre_anchor_top_n, 1)))  # (n, h * w * #anchor, #class)
        negative_loss = F.sum(negative_loss)

        return negative_loss


def _proposal_retina(
        F=mx.ndarray,
        cls_prob=None,
        bbox_pred=None,
        anchors=None,
        im_info=None,
        batch_size=1,
        rpn_pre_nms_top_n=None,
        num_class=None,
        anchor_mean=None,
        anchor_std=None):
    """
    inputs:
        F: symbol or ndarray
        cls_prob: F, (#img, h * w * #anchor, #class)
        bbox_pred: F, (#img, h * w * #anchor, 4)
        anchors: F, (#img, h * w * #anchor, 4)
        im_info: F, (#img, 3)
        batch_size: int
        rpn_pre_nms_top_n: int
        num_class: int
        anchor_means: list, (4,)
        anchor_stds: list, (4,)
    outputs:
        sort_bbox: F, (#img, #n, 4)
        sort_cls_prob: F, (#img, n, #class)
    """

    # slice anchor
    anchors = F.slice_like(anchors, cls_prob, axes=(2, 3))  # (1, 1, h, w, #anchor * 4)
    anchors = F.reshape(anchors, shape=(-3, -3, -4, -1, 4))  # (1, h * w, #anchor, 4)
    anchors = F.broadcast_axis(anchors, axis=0, size=batch_size)  # (#img, h * w, #anchor, 4)
    anchors = anchors.reshape((0, -1, 4))  # (#img, h * w * #anchor, 4)

    # argsort
    cls_prob = F.transpose(cls_prob, axes=(0, 2, 3, 1))  # (#img, h, w, #anchor * #class)
    cls_prob = F.reshape(cls_prob, shape=(0, -1, num_class - 1))  # (#img, h * w * #anchor, #class)
    bbox_pred = F.transpose(bbox_pred, axes=(0, 2, 3, 1))  # (#img, h, w, #anchor * 4)
    bbox_pred = F.reshape(bbox_pred, shape=(0, -1, 4))  # (#img, h * w * #anchor, 4)
    cls_prob_max = F.max(cls_prob, axis=2)
    cls_prob_topk = F.topk(cls_prob_max, axis=1, k=rpn_pre_nms_top_n)
    cls_prob_topk_reshape = F.reshape(cls_prob_topk, shape=(-1,))
    arange_index = F.arange(0, batch_size).reshape((batch_size, 1))
    arange_index = F.broadcast_like(
        lhs=arange_index, rhs=cls_prob_topk, lhs_axes=(1,), rhs_axes=(1,)).reshape(-1)
    topk_indexes = F.stack(arange_index, cls_prob_topk_reshape)
    sort_cls_prob = F.gather_nd(cls_prob, topk_indexes).reshape((batch_size, -1, num_class - 1))
    zero_array = F.zeros_like(F.slice_axis(sort_cls_prob, axis=2, begin=0, end=1))
    sort_cls_prob = F.concat(zero_array, sort_cls_prob, dim=2)
    sort_bbox_pred = F.gather_nd(bbox_pred, topk_indexes).reshape(
        (batch_size, rpn_pre_nms_top_n, 4))
    sort_anchor = F.gather_nd(anchors, topk_indexes).reshape((batch_size, rpn_pre_nms_top_n, 4))

    # decode
    sort_bbox = decode_bbox(F, sort_anchor, sort_bbox_pred, im_info,
                            anchor_mean, anchor_std, True)

    return sort_bbox, sort_cls_prob
