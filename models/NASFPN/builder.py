import math
import mxnet as mx
import mxnext as X

from mxnext.complicate import normalizer_factory
from symbol.builder import Backbone, Neck
from models.retinanet.builder import RetinaNetHead
from models.retinanet.builder import RetinaNetNeck


def merge_sum(f1, f2, name):
    """
    :param f1: feature 1
    :param f2: feature 2
    :param name: name
    :return: sum(f1, f2)
    """
    return mx.sym.ElementWiseSum(f1, f2, name=name + '_sum')

def merge_gp(f1, f2, name):
    """
    :param f1: feature 1, attention feature
    :param f2: feature 2, major feature
    :param name: name
    :return: global pooling fusion of f1 and f2
    """
    gp = mx.sym.Pooling(f1, name=name + '_gp', kernel=(1, 1), pool_type="max", global_pool=True)
    gp = mx.sym.Activation(gp, act_type='sigmoid', name=name + '_sigmoid')
    fuse_mul = mx.sym.broadcast_mul(f2, gp, name=name + '_mul')
    fuse_sum = mx.sym.ElementWiseSum(f1, fuse_mul, name=name + '_sum')
    return fuse_sum


def reluconvbn(data, num_filter, init, norm, name, prefix):
    """
    :param data: data
    :param num_filter: number of convolution filter
    :param init: init method of conv weight
    :param norm: normalizer
    :param name: name
    :return: relu-3x3conv-bn
    """
    data = mx.sym.Activation(data, name=name + '_relu', act_type='relu')
    weight = mx.sym.var(name=prefix + name + "_weight", init=init)
    bias = mx.sym.var(name=prefix + name + "_bias")
    data = mx.sym.Convolution(data, name=prefix + name, weight=weight, bias=bias, num_filter=num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    data = norm(data, name=name+'_bn')
    return data


class NASFPNNeck(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)
    
    @staticmethod
    def get_P0_features(c_features, p_names, dim_reduced, init, norm, kernel=1):
        p_features = {}
        for c_feature, p_name in zip(c_features, p_names):
            p = X.conv(
                data=c_feature,
                filter=dim_reduced,
                kernel=kernel,
                no_bias=False,
                weight=X.var(name=p_name + "_weight", init=init),
                bias=X.var(name=p_name + "_bias", init=X.zero_init()),
                name=p_name
            )
            p_features[p_name] = p
        return p_features

    @staticmethod
    def get_fused_P_feature(p_features, stage, dim_reduced, init, norm):
        prefix = "S{}_".format(stage)
        with mx.name.Prefix(prefix):
            P3_0 = p_features['S{}_P3'.format(stage-1)] # s8
            P4_0 = p_features['S{}_P4'.format(stage-1)] # s16
            P5_0 = p_features['S{}_P5'.format(stage-1)] # s32
            P6_0 = p_features['S{}_P6'.format(stage-1)] # s64
            P7_0 = p_features['S{}_P7'.format(stage-1)] # s128
            # P4_1 = gp(P6_0, P4_0)
            P6_0_to_P4 = mx.sym.UpSampling(
                P6_0,
                scale=4,
                sample_type='nearest',
                name="P6_0_to_P4",
                num_args=1
            )
            P6_0_to_P4 = mx.sym.slice_like(P6_0_to_P4, P4_0)
            P4_1 = merge_gp(P6_0_to_P4, P4_0, name="gp_P6_0_P4_0")
            P4_1 = reluconvbn(P4_1, dim_reduced, init, norm, name="P4_1", prefix=prefix)
            # P4_2 = sum(P4_0, P4_1)
            P4_2 = merge_sum(P4_0, P4_1, name="sum_P4_0_P4_1")
            P4_2 = reluconvbn(P4_2, dim_reduced, init, norm, name="P4_2", prefix=prefix)
            # P3_3 = sum(P4_2, P3_0) end node
            P4_2_to_P3 = mx.sym.UpSampling(
                P4_2,
                scale=2,
                sample_type='nearest',
                name="P4_2_to_P3",
                num_args=1
            )
            P4_2_to_P3 = mx.sym.slice_like(P4_2_to_P3, P3_0)
            P3_3 = merge_sum(P4_2_to_P3, P3_0, name="sum_P4_2_P3_0")
            P3_3 = reluconvbn(P3_3, dim_reduced, init, norm, name="P3_3", prefix=prefix)
            P3 = P3_3
            # P4_4 = sum(P4_2, P3_3) end node
            P3_3_to_P4 = X.pool(P3_3, name="P3_3_to_P4", kernel=2, stride=2, pad=0)
            P3_3_to_P4 = mx.sym.slice_like(P3_3_to_P4, P4_2)
            P4_4 = merge_sum(P4_2, P3_3_to_P4, name="sum_P4_4_P3_3")
            P4_4 = reluconvbn(P4_4, dim_reduced, init, norm, name="P4_4", prefix=prefix)
            P4 = P4_4
            # P5_5 = sum(gp(P4_4, P3_3), P5_0) end node
            P4_4_to_P5 = X.pool(P4_4, kernel=2, stride=2, name="P4_4_to_P5", pad=0)
            P4_4_to_P5 = mx.sym.slice_like(P4_4_to_P5, P5_0)
            P3_3_to_P5 = X.pool(P3_3, kernel=4, stride=4, name="P3_3_to_P5", pad=0)
            P3_3_to_P5 = mx.sym.slice_like(P3_3_to_P5, P5_0)
            gp_P4_4_P3_3 = merge_gp(P4_4_to_P5, P3_3_to_P5, name="gp_P4_4_P3_3")
            P5_5 = merge_sum(gp_P4_4_P3_3, P5_0, name="sum_[gp_P4_4_P3_3]_P5_0")
            P5_5 = reluconvbn(P5_5, dim_reduced, init, norm, name="P5_5", prefix=prefix)
            P5 = P5_5
            # P7_6 = sum(gp(P5_5, P4_2), P7_0) end node 
            P4_2_to_P7 = X.pool(P4_2, name="P4_2_to_P7", kernel=8, stride=8, pad=0)
            P4_2_to_P7 = mx.sym.slice_like(P4_2_to_P7, P7_0)
            P5_5_to_P7 = X.pool(P5_5, name="P5_5_to_P7", kernel=4, stride=4, pad=0)
            P5_5_to_P7 = mx.sym.slice_like(P5_5_to_P7, P7_0)
            gp_P5_5_P4_2 = merge_gp(P5_5_to_P7, P4_2_to_P7, name="gp_P5_5_P4_2")
            P7_6 = merge_sum(gp_P5_5_P4_2, P7_0, name="sum_[gp_P5_5_P4_2]_P7_0")
            P7_6 = reluconvbn(P7_6, dim_reduced, init, norm, name="P7_6", prefix=prefix)
            P7 = P7_6
            # P6_7 = gp(P7_6, P5_5) end node
            P7_6_to_P6 = mx.sym.UpSampling(
                P7_6,
                scale=2,
                sample_type='nearest',
                name="P7_6_to_P6",
                num_args=1                
            )
            P7_6_to_P6 = mx.sym.slice_like(P7_6_to_P6, P6_0)
            P5_5_to_P6 = X.pool(P5_5, name="p5_5_to_P6", kernel=2, stride=2, pad=0)
            P5_5_to_P6 = mx.sym.slice_like(P5_5_to_P6, P6_0)
            P6_7 = merge_gp(P7_6_to_P6, P5_5_to_P6, name="gp_P7_6_to_P6_P5_5_to_P6")
            P6_7 = reluconvbn(P6_7, dim_reduced, init, norm, name="P6_7", prefix=prefix)
            P6 = P6_7

            return {'S{}_P3'.format(stage): P3,
                    'S{}_P4'.format(stage): P4,
                    'S{}_P5'.format(stage): P5,
                    'S{}_P6'.format(stage): P6,
                    'S{}_P7'.format(stage): P7}

    def get_nasfpn_neck(self, data):
        dim_reduced = self.p.dim_reduced
        norm = self.p.normalizer
        num_stage = self.p.num_stage
        S0_kernel = self.p.S0_kernel

        import mxnet as mx
        xavier_init = mx.init.Xavier(factor_type="avg", rnd_type="uniform", magnitude=3)

        c2, c3, c4, c5 = data
        c6 = X.pool(data=c5, name="C6", kernel=2, stride=2, pad=0)
        c7 = X.pool(data=c5, name="C7", kernel=4, stride=4, pad=0)

        c_features = [c3, c4, c5, c6, c7]
        # 0 stage
        p0_names = ['S0_P3', 'S0_P4', 'S0_P5', 'S0_P6', 'S0_P7']
        p_features = self.get_P0_features(c_features, p0_names, dim_reduced, xavier_init, norm, S0_kernel)
        
        # stack stage
        for i in range(num_stage):
            p_features = self.get_fused_P_feature(p_features, i + 1, dim_reduced, xavier_init, norm)
        return p_features['S{}_P3'.format(num_stage)], \
               p_features['S{}_P4'.format(num_stage)], \
               p_features['S{}_P5'.format(num_stage)], \
               p_features['S{}_P6'.format(num_stage)], \
               p_features['S{}_P7'.format(num_stage)]

    def get_rpn_feature(self, rpn_feat):
        return self.get_nasfpn_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.get_nasfpn_neck(rcnn_feat)


class RetinaNetHeadWithBN(RetinaNetHead):
    def __init__(self, pRpn):
        super().__init__(pRpn)

    def _cls_subnet(self, conv_feat, conv_channel, num_base_anchor, num_class, stride):
        p = self.p
        norm = p.normalizer

        # classification subnet
        cls_conv1 = X.conv(
            data=conv_feat,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv1_weight,
            bias=self.cls_conv1_bias,
            no_bias=False,
            name="cls_conv1"
        )
        cls_conv1 = norm(cls_conv1, name="cls_conv1_bn_s{}".format(stride))
        cls_conv1_relu = X.relu(cls_conv1)
        cls_conv2 = X.conv(
            data=cls_conv1_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv2_weight,
            bias=self.cls_conv2_bias,
            no_bias=False,
            name="cls_conv2"
        )
        cls_conv2 = norm(cls_conv2, name="cls_conv2_bn_s{}".format(stride))
        cls_conv2_relu = X.relu(cls_conv2)
        cls_conv3 = X.conv(
            data=cls_conv2_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv3_weight,
            bias=self.cls_conv3_bias,
            no_bias=False,
            name="cls_conv3"
        )
        cls_conv3 = norm(cls_conv3, name="cls_conv3_bn_s{}".format(stride))
        cls_conv3_relu = X.relu(cls_conv3)
        cls_conv4 = X.conv(
            data=cls_conv3_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.cls_conv4_weight,
            bias=self.cls_conv4_bias,
            no_bias=False,
            name="cls_conv4"
        )
        cls_conv4 = norm(cls_conv4, name="cls_conv4_bn_s{}".format(stride))
        cls_conv4_relu = X.relu(cls_conv4)

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

    def _bbox_subnet(self, conv_feat, conv_channel, num_base_anchor, num_class, stride):
        p = self.p
        norm = p.normalizer

        # regression subnet
        bbox_conv1 = X.conv(
            data=conv_feat,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv1_weight,
            bias=self.bbox_conv1_bias,
            no_bias=False,
            name="bbox_conv1"
        )
        bbox_conv1 = norm(bbox_conv1, name="bbox_conv1_bn_s{}".format(stride))
        bbox_conv1_relu = X.relu(bbox_conv1)
        bbox_conv2 = X.conv(
            data=bbox_conv1_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv2_weight,
            bias=self.bbox_conv2_bias,
            no_bias=False,
            name="bbox_conv2"
        )
        bbox_conv2 = norm(bbox_conv2, name="bbox_conv2_bn_s{}".format(stride))
        bbox_conv2_relu = X.relu(bbox_conv2)
        bbox_conv3 = X.conv(
            data=bbox_conv2_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv3_weight,
            bias=self.bbox_conv3_bias,
            no_bias=False,
            name="bbox_conv3"
        )
        bbox_conv3 = norm(bbox_conv3, name="bbox_conv3_bn_s{}".format(stride))
        bbox_conv3_relu = X.relu(bbox_conv3)
        bbox_conv4 = X.conv(
            data=bbox_conv3_relu,
            kernel=3,
            filter=conv_channel,
            weight=self.bbox_conv4_weight,
            bias=self.bbox_conv4_bias,
            no_bias=False,
            name="bbox_conv4"
        )
        bbox_conv4 = norm(bbox_conv4, name="bbox_conv4_bn_s{}".format(stride))
        bbox_conv4_relu = X.relu(bbox_conv4)

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

    def get_output(self, conv_feat):
        if self._cls_logit_list is not None and self._bbox_delta_list is not None:
            return self._cls_logit_list, self._bbox_delta_list

        p = self.p
        stride = p.anchor_generate.stride
        if not isinstance(stride, tuple):
            stride = (stride)
        conv_channel = p.head.conv_channel
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        num_class = p.num_class

        prior_prob = 0.01
        pi = -math.log((1-prior_prob) / prior_prob)

        # shared classification weight and bias
        self.cls_conv1_weight = X.var("cls_conv1_weight", init=X.gauss(std=0.01))
        self.cls_conv1_bias = X.var("cls_conv1_bias", init=X.zero_init())
        self.cls_conv2_weight = X.var("cls_conv2_weight", init=X.gauss(std=0.01))
        self.cls_conv2_bias = X.var("cls_conv2_bias", init=X.zero_init())
        self.cls_conv3_weight = X.var("cls_conv3_weight", init=X.gauss(std=0.01))
        self.cls_conv3_bias = X.var("cls_conv3_bias", init=X.zero_init())
        self.cls_conv4_weight = X.var("cls_conv4_weight", init=X.gauss(std=0.01))
        self.cls_conv4_bias = X.var("cls_conv4_bias", init=X.zero_init())
        self.cls_pred_weight = X.var("cls_pred_weight", init=X.gauss(std=0.01))
        self.cls_pred_bias = X.var("cls_pred_bias", init=X.constant(pi))

        # shared regression weight and bias
        self.bbox_conv1_weight = X.var("bbox_conv1_weight", init=X.gauss(std=0.01))
        self.bbox_conv1_bias = X.var("bbox_conv1_bias", init=X.zero_init())
        self.bbox_conv2_weight = X.var("bbox_conv2_weight", init=X.gauss(std=0.01))
        self.bbox_conv2_bias = X.var("bbox_conv2_bias", init=X.zero_init())
        self.bbox_conv3_weight = X.var("bbox_conv3_weight", init=X.gauss(std=0.01))
        self.bbox_conv3_bias = X.var("bbox_conv3_bias", init=X.zero_init())
        self.bbox_conv4_weight = X.var("bbox_conv4_weight", init=X.gauss(std=0.01))
        self.bbox_conv4_bias = X.var("bbox_conv4_bias", init=X.zero_init())
        self.bbox_pred_weight = X.var("bbox_pred_weight", init=X.gauss(std=0.01))
        self.bbox_pred_bias = X.var("bbox_pred_bias", init=X.zero_init())

        cls_logit_list = []
        bbox_delta_list = []

        for i, s in enumerate(stride):
            cls_logit = self._cls_subnet(
                conv_feat=conv_feat[i],
                conv_channel=conv_channel,
                num_base_anchor=num_base_anchor,
                num_class=num_class,
                stride=s
            )

            bbox_delta = self._bbox_subnet(
                conv_feat=conv_feat[i],
                conv_channel=conv_channel,
                num_base_anchor=num_base_anchor,
                num_class=num_class,
                stride=s
            )

            cls_logit_list.append(cls_logit)
            bbox_delta_list.append(bbox_delta)

        self._cls_logit_list = cls_logit_list
        self._bbox_delta_list = bbox_delta_list

        return self._cls_logit_list, self._bbox_delta_list


class RetinaNetNeckWithBN(RetinaNetNeck):
    def __init__(self, pNeck):
        super().__init__(pNeck)

    def get_retinanet_neck(self, data):
        norm = self.p.normalizer
        c2, c3, c4, c5 = data

        import mxnet as mx
        xavier_init = mx.init.Xavier(factor_type="avg", rnd_type="uniform", magnitude=3)
        # P5
        p5 = X.conv(
            data=c5,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_lateral_weight", init=xavier_init),
            bias=X.var(name="P5_lateral_bias", init=X.zero_init()),
            name="P5_lateral"
        )
        p5_conv = X.conv(
            data=p5,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_conv_weight", init=xavier_init),
            bias=X.var(name="P5_conv_bias", init=X.zero_init()),
            name="P5_conv"
        )

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

        # P6
        P6 = X.conv(
            data=c5,
            kernel=3,
            stride=2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P6_conv_weight", init=xavier_init),
            bias=X.var(name="P6_conv_bias", init=X.zero_init()),
            name="P6_conv"
        )

        # P7
        P6_relu = X.relu(data=P6, name="P6_relu")
        P7 = X.conv(
            data=P6_relu,
            kernel=3,
            stride=2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P7_conv_weight", init=xavier_init),
            bias=X.var(name="P7_conv_bias", init=X.zero_init()),
            name="P7_conv"
        )

        p3_conv = norm(p3_conv, name="P3_conv_bn")
        p4_conv = norm(p4_conv, name="P4_conv_bn")
        p5_conv = norm(p5_conv, name="P5_conv_bn")
        P6 = norm(P6, name="P6_conv_bn")
        P7 = norm(P7, name="P7_conv_bn")

        return p3_conv, p4_conv, p5_conv, P6, P7


class MSRAResNet50V1bFPN(Backbone):
    def __init__(self, pBackbone):
        super(MSRAResNet50V1bFPN, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v1b import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 50, "fpn", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol
