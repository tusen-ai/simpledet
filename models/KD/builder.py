from __future__ import division
from __future__ import print_function

import math
import mxnet as mx
import mxnext as X

from symbol.builder import FasterRcnn, RpnHead, Backbone, Neck
from models.retinanet.builder import RetinaNet


class FitNetHead(object):
    def __init__(self, pKD):
        super().__init__()
        self.p = pKD
        self._student_feat = None
    
    def get_student_feat(self, mimic_feat, mimic_channel):
        if self._student_feat:
            return self._student_feat

        mimic_channel = self.p.channel
        student_hint = mx.sym.Convolution(data=mimic_feat,
                                          num_filter=mimic_channel,
                                          kernel=(1, 1),
                                          stride=(1, 1),
                                          pad=(0, 0),
                                          name="student_hint_conv")
        student_hint = mx.sym.Activation(data=student_hint, 
                                         act_type='relu', 
                                         name="student_hint_relu")
        return student_hint
            
    def get_loss(self, feat_dict, label):
        mimic_stage = self.p.stage
        mimic_channel = self.p.channel
        mimic_grad_scale = self.p.grad_scale
        
        student_feat = self.get_student_feat(feat_dict[mimic_stage], mimic_channel)
        fit_loss = mx.sym.mean(mx.sym.square(student_feat - label))
        fit_loss = mx.sym.MakeLoss(fit_loss, grad_scale=mimic_grad_scale, name="fit_loss")
        return fit_loss
        
        
class FitNetRetinaNet(RetinaNet):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_train_symbol(backbone, neck, head, kd_head):
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")
        teacher_label = X.var("teacher_label")

        feat = backbone.get_rpn_feature()
        c2, c3, c4, c5 = feat
        feat_dict = {'c2': c2,
                     'c3': c3,
                     'c4': c4,
                     'c5': c5}
        feat = neck.get_rpn_feature(feat)

        loss = head.get_loss(feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)
        kd_loss = kd_head.get_loss(feat_dict, teacher_label)

        return X.group(loss + (kd_loss, ))


class FitNetFasterRcnn(FasterRcnn):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_train_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head, kd_head):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")
        teacher_label = X.var("teacher_label")

        rpn_feat = backbone.get_rpn_feature()
        c2, c3, c4, c5 = rpn_feat
        feat_dict = {'c2': c2,
                     'c3': c3,
                     'c4': c4,
                     'c5': c5}
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_head.get_anchor()
        rpn_loss = rpn_head.get_loss(rpn_feat, gt_bbox, im_info)
        proposal, bbox_cls, bbox_target, bbox_weight = rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, im_info)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)

        kd_loss = kd_head.get_loss(feat_dict, teacher_label)

        return X.group(rpn_loss + bbox_loss + (kd_loss, ))

    @classmethod
    def get_test_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head):
        rec_id, im_id, im_info, proposal, proposal_score = \
            FitNetFasterRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        roi_feat = roi_extractor.get_roi_feature_test(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])
