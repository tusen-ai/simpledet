import mxnet as mx
import mxnext as X
from mxnext.complicate import normalizer_factory
from symbol.builder import Neck


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
    data = mx.sym.Activation(data, name=name+'_relu', act_type='relu')
    weight = mx.sym.var(name=prefix + name + "_weight", init=init)
    bias = mx.sym.var(name=prefix + name + "_bias")
    data = mx.sym.Convolution(data, name=prefix + name, weight=weight, bias=bias, num_filter=num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    data = norm(data, name=name+'_bn')
    return data


class NASFPNNeck(Neck):
    def __init__(self, pNeck):
        super(NASFPNNeck, self).__init__(pNeck)
        self.norm = self.pNeck.normalizer
        self.dim_reduced = self.pNeck.dim_reduced
        self.num_stage = self.pNeck.num_stage
        self.S0_kernel = self.pNeck.S0_kernel
    
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
        dim_reduced = self.dim_reduced
        norm = self.norm
        num_stage = self.num_stage
        S0_kernel = self.S0_kernel

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

