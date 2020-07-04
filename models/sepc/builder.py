import mxnet as mx
import mxnext as X
from models.NASFPN.builder import RetinaNetNeck, RetinaNetNeckWithBN
from models.sepc.sepc_neck import SEPCFPN
from utils.patch_config import patch_config_as_nothrow
from models.NASFPN.builder import RetinaNetHeadWithBN
import math


class RetinaNetNeckWithBNWithSEPC(RetinaNetNeckWithBN):
    def __init__(self, pNeck, pSEPC):
        super().__init__(pNeck)
        self.psepc = patch_config_as_nothrow(pSEPC)
        self.neck_with_sepc = None
        stride, pad_sizes = pSEPC.stride, pSEPC.pad_sizes
        for i in range(len(stride)):
            if pad_sizes[0] % stride[i] != 0 or pad_sizes[1] % stride[i] != 0:
                print('Warning: This implementation of ibn used in SEPC expects (it\'s better) the (padded) input sizes {} dividable by the stride {}. '\
                'When this is not satisfied, you should manually check that the feature_sizes at stride \'s\' statisfy the following: ' \
                '\'ceil(pad_sizes[0]/s)==feature_sizes[0]\' and \'ceil(pad_sizes[1]/s)==feature_size[1]\''.format(pad_sizes, stride[i]))                
        self.feat_sizes = [[math.ceil(pad_sizes[0]/stride[i]), math.ceil(pad_sizes[1]/stride[i])] for i in range(len(stride))]

    def get_retinanet_neck(self, data):
        if self.neck_with_sepc is not None:
            return self.neck_with_sepc
        fpn_outs = super().get_retinanet_neck(data)
        p3_conv, p4_conv, p5_conv, p6, p7 = fpn_outs['stride8'], fpn_outs['stride16'], fpn_outs['stride32'], fpn_outs['stride64'], fpn_outs['stride128']

        # add SEPC module after default FPN
        sepc_inputs = [p3_conv, p4_conv, p5_conv, p6, p7]
        sepc_outs = SEPCFPN(
            sepc_inputs,
            out_channels=self.psepc.out_channels,
            pconv_deform=self.psepc.pconv_deform,
            ibn=self.psepc.ibn or False,
            Pconv_num=self.psepc.pconv_num,
            start_level=self.psepc.start_level or 1,
            norm=self.psepc.normalizer,
            lcconv_deform=self.psepc.lcconv_deform or False,
            bilinear_upsample=self.psepc.bilinear_upsample or False,
            feat_sizes=self.feat_sizes,
        )
        self.neck_with_sepc = dict(
            stride128=sepc_outs[4],
            stride64=sepc_outs[3],
            stride32=sepc_outs[2],
            stride16=sepc_outs[1],
            stride8=sepc_outs[0]
        )
        return self.neck_with_sepc


class RetinaNetHeadWithBNWithSEPC(RetinaNetHeadWithBN):
    def __init__(self, pRpn):
        super().__init__(pRpn)

    def _cls_subnet(self, conv_feat, conv_channel, num_base_anchor, num_class, stride, nb_conv=0):
        p = self.p
        if nb_conv <= 0:
            cls_conv4_relu = conv_feat
            if p.fp16:
                cls_conv4_relu = X.to_fp32(cls_conv4_relu, name="cls_conv4_fp32")
            output_channel = num_base_anchor * (num_class - 1)
            output = X.conv(
                data=cls_conv4_relu,
                kernel=3,
                filter=output_channel,
                weight=self.cls_pred_weight,
                bias=self.cls_pred_bias,
                no_bias=False,
                name="cls_pred"
            )
            return output
        return super()._cls_subnet(conv_feat, conv_channel, num_base_anchor, num_class, stride)

    def _bbox_subnet(self, conv_feat, conv_channel, num_base_anchor, num_class, stride, nb_conv=0):
        p = self.p
        if nb_conv <= 0:
            bbox_conv4_relu = conv_feat
            if p.fp16:
                bbox_conv4_relu = X.to_fp32(bbox_conv4_relu, name="bbox_conv4_fp32")
            output_channel = num_base_anchor * 4
            output = X.conv(
                data=bbox_conv4_relu,
                kernel=3,
                filter=output_channel,
                weight=self.bbox_pred_weight,
                bias=self.bbox_pred_bias,
                no_bias=False,
                name="bbox_pred"
            )
            return output
        return super()._bbox_subnet(conv_feat, conv_channel, num_base_anchor, num_class, stride)

    def get_output(self, conv_feat):
        if self._cls_logit_dict is not None and self._bbox_delta_dict is not None:
            return self._cls_logit_dict, self._bbox_delta_dict
        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        conv_channel = p.head.conv_channel
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        num_class = p.num_class
        cls_logit_dict = dict()
        bbox_delta_dict = dict()
        for s in stride:
            conv_feat_cls, conv_feat_loc = mx.sym.split(conv_feat["stride%s" % s], num_outputs=2, axis=1)
            cls_logit = self._cls_subnet(
                # conv_feat=conv_feat["stride%s" % s],
                conv_feat=conv_feat_cls,
                conv_channel=conv_channel,
                num_base_anchor=num_base_anchor,
                num_class=num_class,
                stride=s,
                nb_conv=self.p.nb_conv if self.p.nb_conv is not None else 4,
            )
            bbox_delta = self._bbox_subnet(
                # conv_feat=conv_feat["stride%s" % s],
                conv_feat=conv_feat_loc,
                conv_channel=conv_channel,
                num_base_anchor=num_base_anchor,
                num_class=num_class,
                stride=s,
                nb_conv=self.p.nb_conv if not None else 4,
            )
            cls_logit_dict["stride%s" % s] = cls_logit
            bbox_delta_dict["stride%s" % s] = bbox_delta
        self._cls_logit_dict = cls_logit_dict
        self._bbox_delta_dict = bbox_delta_dict
        return self._cls_logit_dict, self._bbox_delta_dict