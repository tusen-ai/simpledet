"""
IoU convert Operator
Compute MaskIoU Target Given feature, mask ratio, mask target and mask predict logits
"""

import mxnet as mx
import numpy as np


class MaskIoUComputeOperator(mx.operator.CustomOp):
    def __init__(self):
        super().__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        mask_pred_logits = in_data[0].asnumpy()
        mask_target = in_data[1].asnumpy()
        mask_ratio = in_data[2].asnumpy().reshape(-1, )
        mask_inds = in_data[3].asnumpy().reshape(-1, )

        mask_pred = np.array(mask_pred_logits > 0.5, dtype=np.bool)

        intersec = mask_target * mask_pred
        mask_pred_sum = np.sum(mask_pred, axis=(1, 2))
        intersec_sum = np.sum(intersec, axis=(1, 2))
        mask_target_sum = np.sum(mask_target, axis=(1, 2)).astype(np.float)
        mask_target_sum /= mask_ratio
        union = mask_target_sum + mask_pred_sum - intersec_sum
        union = np.maximum(union, 1)
        intersec_sum = np.maximum(intersec_sum, 0)
        iou = np.reshape(intersec_sum / union , (-1, 1))

        positive_inds = np.where(mask_inds > 0)[0]
        weight_list = np.zeros_like(mask_inds)
        weight_list[positive_inds] = 1
        weight_list = weight_list.reshape(-1, 1)

        self.assign(out_data[0], req[0], iou)
        self.assign(out_data[1], req[1], weight_list)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

@mx.operator.register('maskiou_compute')
class MaskIoUComputeProp(mx.operator.CustomOpProp):
    def __init__(self):
        super().__init__(need_top_grad=False)

    def list_arguments(self):
        return ['mask_pred_logits', 'mask_target', 'mask_ratio', 'mask_inds']

    def infer_shape(self, in_shape):
        mask_pred_logits_shape = in_shape[0]
        mask_target_shape = in_shape[1]
        mask_ratio_shape = in_shape[2]
        mask_ind_shape = in_shape[3]

        maskiou_target_shape = (mask_target_shape[0], 1)
        weight_shape = (mask_target_shape[0], 1)
        return [mask_pred_logits_shape, mask_target_shape, mask_ratio_shape, mask_ind_shape], [maskiou_target_shape, weight_shape]

    def list_outputs(self):
        return ['maskiou_target', 'weight_list']

    def create_operator(self, ctx, shapes, dtypes):
        return MaskIoUComputeOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

