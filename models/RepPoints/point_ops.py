import mxnet as mx
import numpy as np


def _gen_offsets(F, dcn_kernel=3, dcn_pad=1):
    """ Generate offset for deformable convolutions.
    :param dcn_kernel: the kernel of deformable convolutions
    :param dcn_pad: the padding of deformable convolutions
    :return: the offsets
    """
    dcn_base = F.arange(-dcn_pad, dcn_pad + 1)
    dcn_base_y = F.repeat(dcn_base, dcn_kernel)
    dcn_base_x = F.tile(dcn_base, dcn_kernel)
    dcn_base_offset = F.reshape(F.stack(dcn_base_y, dcn_base_x, axis=1), (1, -1, 1, 1))
    return dcn_base_offset


def _gen_points(F, feat_map, stride):
    """ Get points according to feature map sizes.
    :param feat_map: feature map
    :param stride: stride of feature map
    :return: generated points
    """
    shift_x = F.reshape(F.contrib.arange_like(feat_map, axis=3, start=0), (1, -1)) * stride
    shift_y = F.reshape(F.contrib.arange_like(feat_map, axis=2, start=0), (-1, 1)) * stride
    shift_xx = F.reshape(F.broadcast_like(shift_x, shift_y, lhs_axes=0, rhs_axes=0), (-1,))
    shift_yy = F.reshape(F.broadcast_like(shift_y, shift_x, lhs_axes=1, rhs_axes=1), (-1,))
    strides = F.ones_like(shift_xx) * stride
    all_points = F.reshape(F.stack(shift_xx, shift_yy, strides, axis=-1), (-4, 1, -1, -2))
    return all_points


def _offset_to_pts(F, center, pred, stride, num_points):
    """ Change from point offset to point coordinate.
    :param center: the initial points center
    :param pred: the predicted point offsets
    :param stride: the stride of the offsets
    :param num_points: the initial number of points for each positions
    :return pts: the predicted point coordinate
    """
    center_xy = F.slice(center, begin=(None, None, 0), end=(None, None, -1))
    pts_center = F.tile(center_xy, reps=(1, 1, num_points))
    pred_transpose = F.transpose(pred, axes=(0, 2, 3, 1))
    pred_reshape = F.reshape(pred_transpose, (0, -3, -4, num_points, 2))
    pred_flip = F.flip(pred_reshape, axis=3)
    xy_pts_shift = F.reshape(pred_flip, (0, 0, -3))
    pts = F.broadcast_add(xy_pts_shift * stride, pts_center)
    return pts


def _offset_to_boxes(F, center, pred, stride, transform="minmax", **kwargs):
    """ Change from point offset to bbox.
    :param center: the initial points center
    :param pred: the predicted point offsets
    :param stride: the stride of the offsets
    :param transform: the transform from points to bbox: "minmax", "partial_minmax" or "moment"
    :return pts: the predicted point coordinate
    """
    center_xy = F.slice(center, begin=(None, None, 0), end=(None, None, -1))
    bbox_center = F.concat(center_xy, center_xy, dim=2)
    bbox_pred_init = _points2bbox(F, pred, transform, **kwargs)
    bbox_shift = F.reshape(F.transpose(bbox_pred_init * stride, (0, 2, 3, 1)), (0, -3, -1))
    bbox_refine = F.broadcast_add(bbox_center, bbox_shift)
    return bbox_refine


def _point_assign(F, points, gt_boxes, scale, num_pos, inf=np.inf):
    """ This method assign a gt bbox to every points set, each points set will be assigned
    with 0, or a positive number. 0 means negative sample, positive number is the index
    (1-based) of assigned gt. The assignment is done in following steps, the order matters.

    1. assign every points to 0
    2. A point is assigned to some gt bbox if
        (i) the point is within the k closest points to the gt bbox
        (ii) the distance between this point and the gt is smaller than other gt bboxes

    :param points: points to be assigned, shape(n, 3) while last dimension
        stands for (x, y, stride)
    :param gt_boxes: Groundtruth boxes, shape (k, 4)
    :param scale: the scale factor for assignment in gt level
    :param num_pos: the number of positive samples
    :param inf: the initial value, default is np.inf
    :return: the assigned label and the corresponding ground truth box
    """
    # set levels of points
    points_x, points_y, points_stride = F.split(points, num_outputs=3, axis=1, squeeze_axis=True)
    points_lvl = F.floor(F.log2(points_stride))
    lvl_min = F.min(points_lvl)
    lvl_max = F.max(points_lvl)

    # assign gt box
    gt_boxes_l, gt_boxes_t, gt_boxes_r, gt_boxes_b, gt_boxes_cls = F.split(
        gt_boxes, num_outputs=5, axis=1, squeeze_axis=True
    )
    gt_boxes_x = (gt_boxes_l + gt_boxes_r) / 2.0
    gt_boxes_y = (gt_boxes_t + gt_boxes_b) / 2.0
    gt_boxes_w = F.maximum(gt_boxes_r - gt_boxes_l, 1e-6)
    gt_boxes_h = F.maximum(gt_boxes_b - gt_boxes_t, 1e-6)
    gt_boxes_lvl = F.floor((F.log2(gt_boxes_w / scale) + F.log2(gt_boxes_h / scale)) / 2.0)
    gt_boxes_lvl = F.broadcast_maximum(F.broadcast_minimum(gt_boxes_lvl, lvl_max), lvl_min)

    # stores the assigned gt index of each point
    gt_boxes_xy = F.reshape(F.stack(gt_boxes_x, gt_boxes_y, axis=-1), (-4, -1, 1, -2))
    gt_boxes_wh = F.reshape(F.stack(gt_boxes_w, gt_boxes_h, axis=-1), (-4, -1, 1, -2))
    points_xy = F.reshape(F.stack(points_x, points_y, axis=-1), (-4, 1, -1, -2))

    # compute the distance between gt center and all points in this level
    points_gts_dist = F.norm(
        F.broadcast_div(F.broadcast_sub(points_xy, gt_boxes_xy), gt_boxes_wh),
        axis=-1
    )
    points_gts_inf = F.ones_like(points_gts_dist) * inf
    # get the points in this level
    points_gts_lvl = F.broadcast_equal(
        F.reshape(gt_boxes_lvl, (-1, 1)), F.reshape(points_lvl, (1, -1))
    )
    # get the valid gt in this level
    points_gts_valid = F.reshape(
        F.broadcast_greater(gt_boxes_cls, F.zeros_like(gt_boxes_cls)),
        (-1, 1)
    )
    points_gts_mask = F.broadcast_mul(points_gts_lvl, points_gts_valid)
    # find the nearest k points to gt center in this level
    points_gts_dist = F.where(points_gts_mask, points_gts_dist, points_gts_inf)
    # the index of nearest k points to gt center in this level
    points_gts_topk = F.topk(points_gts_dist, ret_typ="mask", axis=-1, k=num_pos, is_ascend=True)
    points_gts_dist = F.where(points_gts_topk, points_gts_dist, points_gts_inf)

    # assign the result
    min_gts_dist = F.reshape(F.topk(points_gts_dist, ret_typ='value', axis=0, is_ascend=1), (-1,))
    min_gts_index = F.reshape(F.topk(points_gts_dist, axis=0, is_ascend=1), (-1,))
    min_gts_label = F.take(gt_boxes_cls, min_gts_index)
    points_label = F.where(min_gts_dist < inf, min_gts_label, -F.ones_like(min_gts_label))
    gt_boxes_ = F.stack(gt_boxes_l, gt_boxes_t, gt_boxes_r, gt_boxes_b, axis=-1)
    min_gt_boxes = F.take(gt_boxes_, min_gts_index)
    points_gts = F.where(min_gts_dist < inf, min_gt_boxes, F.zeros_like(min_gt_boxes))
    return points_label, points_gts


def _iou_assign(F, p_boxes, gt_boxes, pos_iou_thr, neg_iou_thr, min_pos_iou):
    """ This method assign a gt bbox to every predicted bbox using iou assignment
    :param p_boxes: predicted bbox
    :param gt_boxes: ground truth bbox
    :param pos_iou_thr: the box with largest IOU larger than the threshold would be selected as
                        positive samples
    :param neg_iou_thr: the box with largest IOU smaller than the threshold would be selected as
                        negative samples
    :param min_pos_iou: the min IOU for positive samples
    :return: the assigned label and the corresponding ground truth box
    """
    gt_boxes_ = F.slice_axis(gt_boxes, begin=0, end=4, axis=-1)
    gt_boxes_cls = F.reshape(F.slice_axis(gt_boxes, begin=4, end=5, axis=-1), (-1,))
    p_gts_iou = F.contrib.box_iou(p_boxes, gt_boxes_, format="corner")
    max_gts_index = F.argmax(p_gts_iou, axis=-1)
    max_gts_iou = F.max(p_gts_iou, axis=-1)
    max_p_iou = F.max(p_gts_iou, axis=-2)
    p_assigned = F.ones_like(max_gts_index) * -1

    # assign bgs
    p_assigned = F.where(max_gts_iou < neg_iou_thr, F.zeros_like(p_assigned), p_assigned)
    # assign max fg
    max_p_iou = F.broadcast_mul(
        F.broadcast_equal(p_gts_iou, F.reshape(max_p_iou, (1, -1))),
        F.reshape(max_p_iou, (1, -1)) > min_pos_iou
    )
    p_assigned = F.where(F.sum(max_p_iou, axis=-1) > 0, F.ones_like(p_assigned), p_assigned)
    # assign fg > thresh
    p_assigned = F.where(max_gts_iou >= pos_iou_thr, F.ones_like(p_assigned), p_assigned)

    # assign labels
    max_gts_label = F.take(gt_boxes_cls, max_gts_index)
    p_label = F.where(p_assigned > 0, max_gts_label, p_assigned)
    max_gt_boxes = F.take(gt_boxes_, max_gts_index)
    p_gts = F.where(p_assigned > 0, max_gt_boxes, F.zeros_like(max_gt_boxes))
    return p_label, p_gts


def _point_target(F, proposals, gt_boxes, num_imgs, assigner="point", **kwargs):
    """Compute corresponding GT box and classification targets for proposals.
    :param proposals: proposals.
    :param gt_boxes: ground truth bboxes.
    :param num_imgs: number of images.
    :param assigner: method to assign proposals, "point" or "iou"
    :return: the assigned label, weight and the corresponding ground truth box
    """
    points_labels = dict()
    points_gts = dict()
    for i in range(num_imgs):
        proposals_this = F.reshape(F.slice_axis(proposals, axis=0, begin=i, end=i + 1), (-3, -2))
        gt_boxes_this = F.reshape(F.slice_axis(gt_boxes, axis=0, begin=i, end=i + 1), (-3, -2))
        if assigner == "point":
            scale = kwargs.get("scale", 4)
            num_pos = kwargs.get("num_pos", 3)
            points_label_this, points_gts_this = _point_assign(
                F, proposals_this, gt_boxes_this, scale, num_pos
            )
            points_labels["img%s" % i] = points_label_this
            points_gts["img%s" % i] = points_gts_this
        elif assigner == "box":
            pos_iou_thr = kwargs.get("pos_iou_thr", 0.5)
            neg_iou_thr = kwargs.get("neg_iou_thr", 0.4)
            min_pos_iou = kwargs.get("min_pos_iou", 0.0)
            points_label_this, points_gts_this = _iou_assign(
                F, proposals_this, gt_boxes_this, pos_iou_thr, neg_iou_thr, min_pos_iou
            )
            points_labels["img%s" % i] = points_label_this
            points_gts["img%s" % i] = points_gts_this
        else:
            raise NotImplementedError("{} is not implemented.".format(assigner))

    points_labels = F.stack(*[points_labels["img%s" % i] for i in range(num_imgs)], axis=0)
    points_gts = F.stack(*[points_gts["img%s" % i] for i in range(num_imgs)], axis=0)
    points_weights = F.repeat(F.expand_dims(F.where(points_labels > 0, F.ones_like(
        points_labels), F.zeros_like(points_labels)), -1), repeats=4, axis=-1)

    return points_labels, points_gts, points_weights


def _points2bbox(F, pts, transform="minmax", y_first=True, **kwargs):
    """ Converting the points set into bounding box.
    :param pts: the input points sets (fields), each points set (fields) is represented
        as 2n scalar.
    :param transform: the transform from points to bbox: "minmax", "partial_minmax" or "moment"
    :param y_first: if y_first=True, the point set is represented as
        [y1, x1, y2, x2 ... yn, xn], otherwise the point set is represented as
        [x1, y1, x2, y2 ... xn, yn].
    :return: each points set is converting to a bbox [x1, y1, x2, y2].
    """
    pts = F.reshape(pts, (0, -4, -1, 2, -2))
    if y_first:
        pts_y, pts_x = F.split(pts, num_outputs=2, axis=2, squeeze_axis=True)
    else:
        pts_x, pts_y = F.split(pts, num_outputs=2, axis=2, squeeze_axis=True)

    if transform == "minmax":
        bbox_left = F.min(pts_x, axis=1, keepdims=True)
        bbox_right = F.max(pts_x, axis=1, keepdims=True)
        bbox_top = F.min(pts_y, axis=1, keepdims=True)
        bbox_bottom = F.max(pts_y, axis=1, keepdims=True)
        bbox = F.concat(bbox_left, bbox_top, bbox_right, bbox_bottom, dim=1)
    elif transform == "partial_minmax":
        pts_x = F.slice_axis(pts_x, begin=0, end=4, axis=1)
        pts_y = F.slice_axis(pts_y, begin=0, end=4, axis=1)
        bbox_left = F.min(pts_x, axis=1, keepdims=True)
        bbox_right = F.max(pts_x, axis=1, keepdims=True)
        bbox_top = F.min(pts_y, axis=1, keepdims=True)
        bbox_bottom = F.max(pts_y, axis=1, keepdims=True)
        bbox = F.concat(bbox_left, bbox_top, bbox_right, bbox_bottom, dim=1)
    elif transform == "moment":
        moment_transfer = kwargs.get("moment_transfer", None)
        if moment_transfer is None:
            raise ValueError("Variable moment_transfer is needed for moment transfrom.")
        pts_y_mean = F.mean(pts_y, axis=1, keepdims=True)
        pts_x_mean = F.mean(pts_x, axis=1, keepdims=True)
        pts_y_std = F.sqrt(
            F.mean(F.square(F.broadcast_sub(pts_y, pts_y_mean)), axis=1, keepdims=True)
        )
        pts_x_std = F.sqrt(
            F.mean(F.square(F.broadcast_sub(pts_x, pts_x_mean)), axis=1, keepdims=True)
        )
        moment_width_transfer, moment_height_transfer = F.split(
            moment_transfer, num_outputs=2, axis=0, squeeze_axis=False
        )
        half_width = F.broadcast_mul(pts_x_std, F.exp(moment_width_transfer))
        half_height = F.broadcast_mul(pts_y_std, F.exp(moment_height_transfer))
        bbox = F.concat(
            pts_x_mean - half_width, pts_y_mean - half_height,
            pts_x_mean + half_width, pts_y_mean + half_height,
            dim=1
        )
    else:
        raise NotImplementedError("Transform {} is not implemented.".format(transform))

    return bbox


if __name__ == "__main__":
    F = mx.ndarray

    # _gen_offsets
    _gen_offsets_nd = _gen_offsets(F, dcn_kernel=3, dcn_pad=1)
    _gen_offsets_ = [-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1]
    _gen_offsets_np = np.array(_gen_offsets_).reshape((1, 18, 1, 1))
    assert np.allclose(_gen_offsets_nd.asnumpy(), _gen_offsets_np)

    # _gen_points
    feat_map = F.ones((2, 2, 2, 3))
    stride = 8
    _gen_points_nd = F.reshape(_gen_points(F, feat_map, stride), (0, -4, 2, 3, -2))
    _gen_points_ = [0, 0, 8, 8, 0, 8, 16, 0, 8, 0, 8, 8, 8, 8, 8, 16, 8, 8]
    _gen_points_np = np.array(_gen_points_).reshape((1, 2, 3, 3))
    assert np.allclose(_gen_points_nd.asnumpy(), _gen_points_np)

    # _points2bbox
    pts = F.arange(36).reshape((1, 18, 2, 1))
    _points2bbox_nd = _points2bbox(F, pts, transform="minmax", y_first=True)
    _points2bbox_ = [2, 3, 0, 1, 34, 35, 32, 33]
    _points2bbox_np = np.array(_points2bbox_).reshape((1, 4, 2, 1))
    assert np.allclose(_points2bbox_nd.asnumpy(), _points2bbox_np)

    # _point_assign
    points = F.concat(*[_gen_points(F, F.zeros((1, 3, int(64 / s), int(128 / s))), s)
                        for s in [32, 64]], dim=1).reshape(-3, -1)
    gt_boxes = F.array([[63, 923, 123, 1800, 2], [200, 50, 600, 120, 3], [21, 456, 123, 712, 4],
                        [325, 123, 523, 612, 5], [-1, -1, 5000, 5000, 6]])
    l_nd, gt_nd = _point_assign(F, points, gt_boxes, scale=4, num_pos=1)
    l_ = [-1, -1, -1, -1, -1, -1, 4, 3, -1, 6]
    gt_ = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
           [21, 456, 123, 712], [200, 50, 600, 120], [0, 0, 0, 0], [-1, -1, 5000, 5000]]
    assert np.allclose(l_nd.asnumpy(), np.array(l_))
    assert np.allclose(gt_nd.asnumpy(), np.array(gt_))

    # _iou_assign
    proposals = F.array([[45, 23, 452, 45], [12, 798, 45, 902], [103, 563, 345, 609], [
                        34, 452, 123, 623], [12, 23, 43, 134], [341, 78, 587, 102]])
    gt_boxes = F.array([[63, 923, 123, 1800, 2], [200, 50, 600, 120, 3], [21, 456, 123, 712, 4]])
    l_nd, gt_nd = _iou_assign(F, proposals, gt_boxes, 0.5, 0.4, 0.0)
    l_ = [0, 0, 0, 4, 0, 3]
    gt_ = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [21, 456, 123, 712],
           [0, 0, 0, 0], [200, 50, 600, 120]]
    assert np.allclose(l_nd.asnumpy(), np.array(l_))
    assert np.allclose(gt_nd.asnumpy(), np.array(gt_))
