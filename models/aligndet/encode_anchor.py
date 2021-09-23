"""
Encode boxes for anchors w.r.t matching gt_boxes
author: Chenxia Han

input:
    anchor:   (N, H * W * A, 4)
    gt_boxes: (N, MAX_NUM_GT, 5)
    im_info:  (N, 3)
output:
    label:   (N, \sum{A * H * W})
    target:  (N, A * 4, \sum{H * W})
    weight:  (N, A * 4, \sum{H * W})
"""

import mxnet as mx
import numpy as np

from models.aligndet.input import AlignPyramidAnchorTarget2D

class AnchorTarget2DParam:
    def __init__(self, short, long, stride, mean, std, class_agnostic,
                 all_anchor_list, allowed_border, pos_thr, neg_thr, min_pos_thr):
        self.mean = mean
        self.std = std
        self.class_agnostic = class_agnostic

        # input anchor
        self.all_anchor_list = all_anchor_list

        self.generate.short = tuple(short)
        self.generate.long = tuple(long)
        self.generate.stride = tuple(stride)
        self.assign.allowed_border = allowed_border
        self.assign.pos_thr = pos_thr
        self.assign.neg_thr = neg_thr
        self.assign.min_pos_thr = min_pos_thr

    class generate:
        short = None
        long = None
        stride = None

    class assign:
        allowed_border = None
        pos_thr = None
        neg_thr = None
        min_pos_thr = None


class EncodeAnchorOperator(mx.operator.CustomOp):
    def __init__(self, short, long, stride, mean, std, class_agnostic,
                 allowed_border, pos_thr, neg_thr, min_pos_thr):
        self.short = short
        self.long = long
        self.stride = stride
        self.mean = mean
        self.std = std
        self.class_agnostic = class_agnostic
        self.allowed_border = allowed_border
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.min_pos_thr = min_pos_thr

    def forward(self, is_train, req, in_data, out_data, aux):
        anchor_list = in_data[:-2]
        gt_boxes = in_data[-2]
        im_info = in_data[-1]

        nbatch = in_data[0].shape[0]

        short = self.short
        long = self.long
        stride = self.stride
        mean = self.mean
        std = self.std
        class_agnostic = self.class_agnostic
        allowed_border = self.allowed_border
        pos_thr = self.pos_thr
        neg_thr = self.neg_thr
        min_pos_thr = self.min_pos_thr

        label_list = []
        target_list = []
        weight_list = []

        for i in range(nbatch):
            anchor_list_np = []
            for anchor in anchor_list:
                anchor_list_np.append(anchor[i].asnumpy())

            anchor_param = AnchorTarget2DParam(
                short,
                long,
                stride,
                mean,
                std,
                class_agnostic,
                anchor_list_np,
                allowed_border,
                pos_thr,
                neg_thr,
                min_pos_thr
            )
            anchor_target_2d = AlignPyramidAnchorTarget2D(anchor_param)
            input_record = {"im_info": im_info[i].asnumpy(), "gt_bbox": gt_boxes[i].asnumpy()}
            anchor_target_2d.apply(input_record)

            label_list.append(mx.nd.array(input_record["rpn_cls_label"]))
            target_list.append(mx.nd.array(input_record["rpn_reg_target"]))
            weight_list.append(mx.nd.array(input_record["rpn_reg_weight"]))

        label = mx.nd.stack(*label_list, axis=0)
        target = mx.nd.stack(*target_list, axis=0)
        weight = mx.nd.stack(*weight_list, axis=0)

        self.assign(out_data[0], req[0], label)
        self.assign(out_data[1], req[1], target)
        self.assign(out_data[2], req[2], weight)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        num_input = len(in_data)
        for i in range(num_input):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register("encode_anchor")
class EncodeAnchorProp(mx.operator.CustomOpProp):
    def __init__(self, short, long, stride, num_anchors, class_agnostic,
                 allowed_border, pos_thr, neg_thr, min_pos_thr,
                 mean="(0, 0, 0, 0)", std="(1, 1, 1, 1)"):
        super(EncodeAnchorProp, self).__init__(need_top_grad=False)
        self.short = eval(short)
        self.long = eval(long)
        self.stride = eval(stride)
        self.num_anchors = eval(num_anchors)
        self.class_agnostic = eval(class_agnostic)
        self.allowed_border = int(allowed_border)
        self.pos_thr = float(pos_thr)
        self.neg_thr = float(neg_thr)
        self.min_pos_thr = float(min_pos_thr)
        self.mean = eval(mean)
        self.std = eval(std)

    def list_arguments(self):
        args_list = []
        for s in self.stride:
            args_list.append("stride%s" % s)
        args_list += ["gt_boxes", "im_info"]

        return args_list

    def list_outputs(self):
        return ["label", "target", "weight"]

    def infer_shape(self, in_shape):
        anchor_shape_list = in_shape[:-2]
        gt_boxes_shape = in_shape[-2]
        im_info_shape = in_shape[-1]

        nbatch = im_info_shape[0]

        assert(anchor_shape_list[0][2] == 4)
        assert(gt_boxes_shape[2] == 5)
        assert(im_info_shape[1] == 3)
        assert(anchor_shape_list[0][0] == nbatch)
        assert(gt_boxes_shape[0] == nbatch)

        num_anchors = self.num_anchors
        total_anchors = np.sum([x[1] for x in anchor_shape_list])

        label_shape = (nbatch, total_anchors)
        target_shape = (nbatch, num_anchors * 4, total_anchors // num_anchors)
        weight_shape = (nbatch, num_anchors * 4, total_anchors // num_anchors)

        return anchor_shape_list + [gt_boxes_shape, im_info_shape], \
               [label_shape, target_shape, weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return EncodeAnchorOperator(self.short, self.long, self.stride,
                                    self.mean, self.std, self.class_agnostic,
                                    self.allowed_border, self.pos_thr,
                                    self.neg_thr, self.min_pos_thr)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
