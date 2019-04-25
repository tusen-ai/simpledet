"""
Collect top proposals across all levels for FPN
author: Yi Jiang, Chenxia Han
"""

import mxnet as mx
import numpy as np


class GetTopProposalOperator(mx.operator.CustomOp):
    def __init__(self, top_n):
        super(GetTopProposalOperator, self).__init__()
        self.top_n = top_n

    def forward(self, is_train, req, in_data, out_data, aux):
        bboxes = in_data[0]
        scores = in_data[1]

        num_image = bboxes.shape[0]
        top_n = self.top_n
        top_bboxes = []
        top_scores = []

        for i in range(num_image):
            image_bboxes = bboxes[i]
            image_scores = scores[i]
            argsort_ind = mx.nd.argsort(image_scores[:,0], is_ascend=False)
            image_bboxes = image_bboxes[argsort_ind]
            image_bboxes = image_bboxes[:top_n]
            image_scores = image_scores[argsort_ind]
            image_scores = image_scores[:top_n]

            top_bboxes.append(image_bboxes)
            top_scores.append(image_scores)

        top_bboxes = mx.nd.stack(*top_bboxes)
        top_scores = mx.nd.stack(*top_scores)

        self.assign(out_data[0], req[0], top_bboxes)
        self.assign(out_data[1], req[1], top_scores)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('get_top_proposal')
class GetTopProposalProp(mx.operator.CustomOpProp):
    def __init__(self, top_n):
        super(GetTopProposalProp, self).__init__(need_top_grad=False)
        self.top_n = int(top_n)

    def list_arguments(self):
        return ['bbox', 'score']

    def list_outputs(self):
        return ['bbox', 'score']

    def infer_shape(self, in_shape):
        bbox_shape = in_shape[0]
        score_shape = in_shape[1]
        num_image = bbox_shape[0]

        top_bbox_shape = (num_image, self.top_n, 4)
        top_score_shape = (num_image, self.top_n, 1)

        return [bbox_shape, score_shape], \
               [top_bbox_shape, top_score_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return GetTopProposalOperator(self.top_n)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
