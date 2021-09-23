from __future__ import division
from __future__ import print_function

import numpy as np

from models.retinanet.input import PyramidAnchorTarget2DBase

class AlignPyramidAnchorTarget2D(PyramidAnchorTarget2DBase):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 4)
    output: anchor_label, ndarray(num_anchor * h * w)
            anchor_bbox_target, ndarray(num_anchor * 4, h * w)
            anchor_bbox_weight, ndarray(num_anchor * 4, h * w)
    """

    def __init__(self, pAnchor):
        super(AlignPyramidAnchorTarget2D, self).__init__(pAnchor)

        self.pyramid_levels = len(self.p.generate.stride)

        self.anchor_target_2d = PyramidAnchorTarget2DBase(self.p)
        self.all_anchor_list = self.p.all_anchor_list

        self.anchor_target_2d.v_all_anchor = self.v_all_anchor
        self.anchor_target_2d.h_all_anchor = self.h_all_anchor

    @property
    def v_all_anchor(self):
        anchors = np.concatenate(self.all_anchor_list)
        return anchors

    @property
    def h_all_anchor(self):
        anchors = np.concatenate(self.all_anchor_list)
        return anchors

    def apply(self, input_record):
        anchor_size = [0] + [x.shape[0] for x in self.all_anchor_list]
        anchor_size = np.cumsum(anchor_size)
        cls_label, reg_target, reg_weight = \
            self.anchor_target_2d.apply(input_record)

        im_info = input_record["im_info"]
        h, w = im_info[:2]

        mean = np.array(self.p.mean)
        std = np.array(self.p.std)

        cls_label_list = []
        reg_target_list = []
        reg_weight_list = []
        for i in range(self.pyramid_levels):
            p = self.p

            cls_label_level = cls_label[anchor_size[i]:anchor_size[i+1]]
            reg_target_level = reg_target[anchor_size[i]:anchor_size[i+1]]
            reg_weight_level = reg_weight[anchor_size[i]:anchor_size[i+1]]
            """
            label: (h * w * A) -> (A * h * w)
            bbox_target: (h * w * A, 4) -> (A * 4, h * w)
            bbox_weight: (h * w * A, 4) -> (A * 4, h * w)
            """
            if h >= w:
                fh, fw = p.generate.long[i], p.generate.short[i]
            else:
                fh, fw = p.generate.short[i], p.generate.long[i]

            reg_target_level = (reg_target_level - mean) / std

            cls_label_level = cls_label_level.reshape((fh, fw, -1)).transpose(2, 0, 1).reshape(-1)
            reg_target_level = reg_target_level.reshape((fh, fw, -1)).transpose(2, 0, 1)
            reg_weight_level = reg_weight_level.reshape((fh, fw, -1)).transpose(2, 0, 1)

            reg_target_level = reg_target_level.reshape(-1, fh * fw)
            reg_weight_level = reg_weight_level.reshape(-1, fh * fw)

            cls_label_list.append(cls_label_level)
            reg_target_list.append(reg_target_level)
            reg_weight_list.append(reg_weight_level)

        cls_label = np.concatenate(cls_label_list, axis=0)
        reg_target = np.concatenate(reg_target_list, axis=1)
        reg_weight = np.concatenate(reg_weight_list, axis=1)

        input_record["rpn_cls_label"] = cls_label
        input_record["rpn_reg_target"] = reg_target
        input_record["rpn_reg_weight"] = reg_weight

        return input_record["rpn_cls_label"], \
               input_record["rpn_reg_target"], \
               input_record["rpn_reg_weight"]
