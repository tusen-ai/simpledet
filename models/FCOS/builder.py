from __future__ import print_function

import math
import mxnet as mx
import mxnext as X
from utils.patch_config import patch_config_as_nothrow

from symbol.builder import Backbone, RpnHead, Neck
from models.FCOS.loss import IoULoss, make_sigmoid_focal_loss, make_binary_cross_entropy_loss
from models.FCOS.utils import GetProposalSingleStageProp, GetBatchProposalProp
from models.FCOS.input import make_fcos_gt

class FCOSFPNHead():
    def __init__(self, pRpn):
        self.p = patch_config_as_nothrow(pRpn)

        self.centerness_logit_dict  = None
        self.cls_logit_dict   = None
        self.offset_logit_dict  = None

    def get_anchor(self):
        raise NotImplementedError("FCOS: it's an anchor free model.")

    def get_output(self, conv_fpn_feat):
        p = self.p

        centerness_logit_dict = {}
        cls_logit_dict = {}
        offset_logit_dict = {}

        # heads are shared across stages
        shared_conv1_w = X.var(name="shared_conv1_3x3_weight", init=X.gauss(0.01))
        shared_conv1_b = X.var(name="shared_conv1_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        shared_conv2_w = X.var(name="shared_conv2_3x3_weight", init=X.gauss(0.01))
        shared_conv2_b = X.var(name="shared_conv2_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        shared_conv3_w = X.var(name="shared_conv3_3x3_weight", init=X.gauss(0.01))
        shared_conv3_b = X.var(name="shared_conv3_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        shared_conv4_w = X.var(name="shared_conv4_3x3_weight", init=X.gauss(0.01))
        shared_conv4_b = X.var(name="shared_conv4_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        centerness_conv_w = X.var(name="centerness_conv_3x3_weight", init=X.gauss(0.01))
        centerness_conv_b = X.var(name="centerness_conv_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        cls_conv_w = X.var(name="cls_conv_3x3_weight", init=X.gauss(0.01))
        cls_conv_b = X.var(name="cls_conv_3x3_bias", init=X.constant(-math.log(99)), lr_mult=2, wd_mult=0)	# init with -log((1-0.01)/0.01)
        offset_conv1_w = X.var(name="offset_conv1_3x3_weight", init=X.gauss(0.01))
        offset_conv1_b = X.var(name="offset_conv1_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        offset_conv2_w = X.var(name="offset_conv2_3x3_weight", init=X.gauss(0.01))
        offset_conv2_b = X.var(name="offset_conv2_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        offset_conv3_w = X.var(name="offset_conv3_3x3_weight", init=X.gauss(0.01))
        offset_conv3_b = X.var(name="offset_conv3_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        offset_conv4_w = X.var(name="offset_conv4_3x3_weight", init=X.gauss(0.01))
        offset_conv4_b = X.var(name="offset_conv4_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)
        offset_conv5_w = X.var(name="offset_conv5_3x3_weight", init=X.gauss(0.01))
        offset_conv5_b = X.var(name="offset_conv5_3x3_bias", init=X.zero_init(), lr_mult=2, wd_mult=0)


        for stride in p.FCOSParam.stride:
            # centerness & cls shared layer
            shared_conv1 = X.conv(
                           conv_fpn_feat['stride%s' % stride],
                           kernel=3, filter=256, no_bias=False,
                           name="shared_conv1_3x3_%s" % stride,
                           weight=shared_conv1_w,
                           bias=shared_conv1_b,
                          )
            shared_gn1 = X.gn(shared_conv1, name='shared_gn1_3x3_%s' % stride, num_group=32)
            shared_relu1 = X.relu(shared_gn1, name='shared_relu1_3x3_%s' % stride)
            shared_conv2 = X.conv(
                           shared_relu1,
                           kernel=3, filter=256, no_bias=False,
                           name="shared_conv2_3x3_%s" % stride,
                           weight=shared_conv2_w,
                           bias=shared_conv2_b,
                          )
            shared_gn2 = X.gn(shared_conv2, name='shared_gn2_3x3_%s' % stride, num_group=32)
            shared_relu2 = X.relu(shared_gn2, name='shared_relu2_3x3_%s' % stride)
            shared_conv3 = X.conv(
                           shared_relu2,
                           kernel=3, filter=256, no_bias=False,
                           name="shared_conv3_3x3_%s" % stride,
                           weight=shared_conv3_w,
                           bias=shared_conv3_b,
                          )
            shared_gn3 = X.gn(shared_conv3, name='shared_gn3_3x3_%s' % stride, num_group=32)
            shared_relu3 = X.relu(shared_gn3, name='shared_relu3_3x3_%s' % stride)
            shared_conv4 = X.conv(
                           shared_relu3,
                           kernel=3, filter=256, no_bias=False,
                           name="shared_conv4_3x3_%s" % stride,
                           weight=shared_conv4_w,
                           bias=shared_conv4_b,
                          )
            shared_gn4 = X.gn(shared_conv4, name='shared_gn4_3x3_%s' % stride, num_group=32)
            shared_relu4 = X.relu(shared_gn4, name='shared_relu4_3x3_%s' % stride)
            # centerness head
            center_logit = X.conv(
                            shared_relu4,
                            kernel=3,
                            filter=1,
                            name="center_conv_3x3_%s" % stride,
                            no_bias=False,
                            weight=centerness_conv_w,
                            bias=centerness_conv_b,
                           )
            # cls head
            cls_logit = X.conv(
                          shared_relu4,
                          kernel=3,
                          filter=p.FCOSParam.num_classifier,		# remove bg channel
                          name="cls_conv_3x3_%s" % stride,
                          no_bias=False,
                          weight=cls_conv_w,
                          bias=cls_conv_b,
                         )
            # offset head
            offset_conv1 = X.conv(
                           conv_fpn_feat['stride%s' % stride],
                           kernel=3,
                           filter=256,
                           name="offset_conv1_3x3_%s" % stride,
                           no_bias=False,
                           weight=offset_conv1_w,
                           bias=offset_conv1_b,
                          )
            offset_gn1 = X.gn(offset_conv1, name='offset_gn1_3x3_%s' % stride, num_group=32)
            offset_relu1 = X.relu(offset_gn1, name='offset_relu1_3x3_%s' % stride)
            offset_conv2 = X.conv(
                           offset_relu1,
                           kernel=3,
                           filter=256,
                           name="offset_conv2_3x3_%s" % stride,
                           no_bias=False,
                           weight=offset_conv2_w,
                           bias=offset_conv2_b,
                          )
            offset_gn2 = X.gn(offset_conv2, name='offset_gn2_3x3_%s' % stride, num_group=32)
            offset_relu2 = X.relu(offset_gn2, name='offset_relu2_3x3_%s' % stride)
            offset_conv3 = X.conv(
                           offset_relu2,
                           kernel=3,
                           filter=256,
                           name="offset_conv3_3x3_%s" % stride,
                           no_bias=False,
                           weight=offset_conv3_w,
                           bias=offset_conv3_b,
                          )
            offset_gn3 = X.gn(offset_conv3, name='offset_gn3_3x3_%s' % stride, num_group=32)
            offset_relu3 = X.relu(offset_gn3, name='offset_relu3_3x3_%s' % stride)
            offset_conv4 = X.conv(
                           offset_relu3,
                           kernel=3,
                           filter=256,
                           name="offset_conv1_3x3_%s" % stride,
                           no_bias=False,
                           weight=offset_conv4_w,
                           bias=offset_conv4_b,
                          )
            offset_gn4 = X.gn(offset_conv4, name='offset_gn4_3x3_%s' % stride, num_group=32)
            offset_relu4 = X.relu(offset_gn4, name='offset_relu4_3x3_%s' % stride)
            offset_logit = X.conv(
                            offset_relu4,
                            kernel=3,
                            filter=4,
                            name="offset_conv5_3x3_%s" % stride,
                            no_bias=False,
                            weight=offset_conv5_w,
                            bias=offset_conv5_b,
                           )
            offset_logit = mx.sym.broadcast_mul(lhs=offset_logit, rhs=X.var(name="offset_scale_%s_w" % stride, init=X.constant(1), shape=(1,1,1,1)))
            offset_logit = mx.sym.exp(offset_logit)

            centerness_logit_dict[stride]  = center_logit
            cls_logit_dict[stride]  = cls_logit
            offset_logit_dict[stride] = offset_logit

        self.centerness_logit_dict = centerness_logit_dict
        self.cls_logit_dict = cls_logit_dict
        self.offset_logit_dict = offset_logit_dict

        return self.centerness_logit_dict, self.cls_logit_dict, self.offset_logit_dict

    def get_loss(self, conv_fpn_feat, gt_bbox, im_info):
        p = self.p
        bs = p.batch_image	# batch_size on a single gpu
        centerness_logit_dict, cls_logit_dict, offset_logit_dict = self.get_output(conv_fpn_feat)

        centerness_loss_list = []
        cls_loss_list = []
        offset_loss_list = []

        # prepare gt
        ignore_label = X.block_grad(X.var('ignore_label', init=X.constant(p.loss_setting.ignore_label), shape=(1,1)))
        ignore_offset = X.block_grad(X.var('ignore_offset', init=X.constant(p.loss_setting.ignore_offset), shape=(1,1,1)))
        gt_bbox = X.var('gt_bbox')
        im_info = X.var('im_info')
        centerness_labels, cls_labels, offset_labels = make_fcos_gt(gt_bbox, im_info,
                                                                    p.loss_setting.ignore_offset,
                                                                    p.loss_setting.ignore_label,
                                                                    p.FCOSParam.num_classifier)
        centerness_labels = X.block_grad(centerness_labels)
        cls_labels = X.block_grad(cls_labels)
        offset_labels = X.block_grad(offset_labels)

        # gather output logits
        cls_logit_dict_list = []
        centerness_logit_dict_list = []
        offset_logit_dict_list = []
        for idx, stride in enumerate(p.FCOSParam.stride):
            # (c,H1,W1), (c,H2,W2), ..., (c,H5,W5) -> (H1W1+H2W2+...+H5W5), ...c..., (H1W1+H2W2+...+H5W5)
            cls_logit_dict_list.append(mx.sym.reshape(cls_logit_dict[stride], shape=(0,0,-1)))
            centerness_logit_dict_list.append(mx.sym.reshape(centerness_logit_dict[stride], shape=(0,0,-1)))
            offset_logit_dict_list.append(mx.sym.reshape(offset_logit_dict[stride], shape=(0,0,-1)))
        cls_logits = mx.sym.reshape(mx.sym.concat(*cls_logit_dict_list, dim=2), shape=(0,-1))
        centerness_logits = mx.sym.reshape(mx.sym.concat(*centerness_logit_dict_list, dim=2), shape=(0,-1))
        offset_logits = mx.sym.reshape(mx.sym.concat(*offset_logit_dict_list, dim=2), shape=(0,4,-1))

        # make losses
        nonignore_mask = mx.sym.broadcast_not_equal(lhs=cls_labels, rhs=ignore_label)
        nonignore_mask = X.block_grad(nonignore_mask)
        cls_loss = make_sigmoid_focal_loss(gamma=p.loss_setting.focal_loss_gamma, alpha=p.loss_setting.focal_loss_alpha,
                                           logits=cls_logits, labels=cls_labels, nonignore_mask=nonignore_mask)
        cls_loss = X.loss(cls_loss, grad_scale=1)

        nonignore_mask = mx.sym.broadcast_logical_and(lhs=mx.sym.broadcast_not_equal( lhs=X.block_grad(centerness_labels), rhs=ignore_label ),
                                                      rhs=mx.sym.broadcast_greater( lhs=centerness_labels, rhs=mx.sym.full((1,1), 0) )
                                                     )
        nonignore_mask = X.block_grad(nonignore_mask)
        centerness_loss = make_binary_cross_entropy_loss(centerness_logits, centerness_labels, nonignore_mask)
        centerness_loss = X.loss(centerness_loss, grad_scale=1)

        offset_loss = IoULoss(offset_logits, offset_labels, ignore_offset, centerness_labels, name='offset_loss')
        return centerness_loss, cls_loss, offset_loss

    # inference
    def get_all_proposal(self, conv_fpn_feat, im_info):
        p = self.p
        centerness_logit_dict, cls_logit_dict, offset_logit_dict = self.get_output(conv_fpn_feat)
        all_stage_bboxes = []

        # iterate over all stages
        for stride in p.FCOSParam.stride:

            centerness_logit = mx.sym.sigmoid(centerness_logit_dict[stride])
            cls_logit = mx.sym.sigmoid(cls_logit_dict[stride])
            offset_logit = offset_logit_dict[stride]

            # get proposals in a single stage
            boxes = mx.sym.Custom(centerness_logit, cls_logit, offset_logit, im_info, 
                                  pre_nms_top_n=int(p.proposal.pre_nms_top_n), stride=stride,
                                  pre_nms_thresh=p.proposal.pre_nms_thresh,
                                  op_type='get_proposal_single_stage')   

            all_stage_bboxes.append(boxes)     

        # fuse results from all stages in a batch
        all_stage_bboxes = mx.sym.concat(*all_stage_bboxes, dim=1)
        bboxes, score, cls_id = mx.sym.Custom(bbox=all_stage_bboxes, op_type='get_batch_proposal')
        self._proposal = score, bboxes

        return score, bboxes	# (cls_scores [N,81], bbox [N,x,y,x,y])

    def get_sampled_proposal(self, conv_fpn_feat, gt_bbox, im_info):
        raise NotImplementedError("FCOS: it's an anchor free model.")


class MSRAResNet50V1FPN(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 50, "fpn", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class MSRAResNet101V1FPN(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 101, "fpn", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class FCOSFPNNeck(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)
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

    def fpn_neck(self, data):
        if self.fpn_feat is not None:
            return self.fpn_feat

        _, c3, c4, c5 = data

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
        p6_conv = X.conv(
            data=p5_conv,
            kernel=3,
            stride=2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P6_conv_weight", init=xavier_init),
            bias=X.var(name="P6_conv_bias", init=X.zero_init()),
            name="P6_subsampling_conv"
        )
        p6_conv = self.add_norm(p6_conv)

        # P7
        p6_relu = X.relu(p6_conv, name="P6_relu")	# here exists a relu
        p7_conv = X.conv(
            data=p6_relu,
            kernel=3,
            stride=2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P7_conv_weight", init=xavier_init),
            bias=X.var(name="P7_conv_bias", init=X.zero_init()),
            name="P7_subsampling_conv"
        )
        p7_conv = self.add_norm(p7_conv)

        conv_fpn_feat = dict(
            stride128=p7_conv,
            stride64=p6_conv,
            stride32=p5_conv,
            stride16=p4_conv,
            stride8=p3_conv,
        )

        self.fpn_feat = conv_fpn_feat
        return self.fpn_feat

    def get_rpn_feature(self, rpn_feat):
        return self.fpn_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.fpn_neck(rcnn_feat)
