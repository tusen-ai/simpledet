"""
Assign Layer operator for FPN
author: Yi Jiang, Chenxia Han
"""

import mxnet as mx
import numpy as np


class AssignLayerFPNOperator(mx.operator.CustomOp):
    def __init__(self, rcnn_stride, roi_canonical_scale, roi_canonical_level):
        super(AssignLayerFPNOperator, self).__init__()
        self.rcnn_stride = rcnn_stride
        self.roi_canonical_scale = roi_canonical_scale
        self.roi_canonical_level = roi_canonical_level

    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0]

        rcnn_stride = self.rcnn_stride
        scale0 = self.roi_canonical_scale
        lvl0 = self.roi_canonical_level
        k_min = np.log2(min(rcnn_stride))
        k_max = np.log2(max(rcnn_stride))

        rois_area = (all_rois[:, :, 2] - all_rois[:, :, 0] + 1) \
                    * (all_rois[:, :, 3] - all_rois[:, :, 1] + 1)

        scale = mx.nd.sqrt(rois_area)
        target_lvls = mx.nd.floor(lvl0 + mx.nd.log2(scale / scale0 + 1e-6))
        target_lvls = mx.nd.clip(target_lvls, k_min, k_max)
        target_stride = (2 ** target_lvls).astype('uint8')

        for i, s in enumerate(rcnn_stride):
            lvl_rois = mx.nd.zeros_like(all_rois)
            lvl_inds = mx.nd.expand_dims(target_stride == s, axis=2).astype('float32')
            lvl_inds = mx.nd.broadcast_like(lvl_inds, lvl_rois)
            lvl_rois = mx.nd.where(lvl_inds, all_rois, lvl_rois)

            self.assign(out_data[i], req[i], lvl_rois)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('assign_layer_fpn')
class AssignLayerFPNProp(mx.operator.CustomOpProp):
    def __init__(self, rcnn_stride, roi_canonical_scale, roi_canonical_level):
        super(AssignLayerFPNProp, self).__init__(need_top_grad=False)
        self.rcnn_stride = eval(rcnn_stride)
        self.roi_canonical_scale = int(roi_canonical_scale)
        self.roi_canonical_level = int(roi_canonical_level)

    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        rois_list = ['rois_s{}'.format(s) for s in self.rcnn_stride]
        return rois_list

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]

        output_rois_shape = [rpn_rois_shape] * len(self.rcnn_stride)

        return [rpn_rois_shape], output_rois_shape

    def create_operator(self, ctx, shapes, dtypes):
        return AssignLayerFPNOperator(self.rcnn_stride, self.roi_canonical_scale,
                                      self.roi_canonical_level)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
