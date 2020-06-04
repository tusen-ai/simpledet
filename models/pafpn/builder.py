import mxnet as mx
import mxnext as X
from symbol.builder import RoiAlign, Neck
from models.NASFPN.builder import NASFPNNeck, merge_sum, reluconvbn


# NOTE not used here for PAFPN
class AdaptiveFPNRoiAlign(RoiAlign):
    def __init__(self, pRoi):
        super().__init__(pRoi)

    def get_roi_feature(self, conv_fpn_feat, proposal):
        p = self.p
        rcnn_stride = p.stride
        if p.fp16:
            for stride in rcnn_stride:
                conv_fpn_feat['stride%s'%stride] = X.to_fp32(conv_fpn_feat['stride%s'%stride],
                    name='fpn_stride%s_to_fp32')
        fpn_roi_feats = list()
        for stride in rcnn_stride:
            feat_lvl = conv_fpn_feat['stride%s'%stride]
            proposal_fpn = proposal  # NOTE use all proposals
            roi_feat = X.roi_align(feat_lvl, rois=proposal_fpn, out_size=p.out_size, stride=stride, name='roi_align')
            fpn_roi_feats.append(roi_feat)
        roi_feat = X.add_n(*fpn_roi_feats)
        if p.fp16:
            roi_feat = X.to_fp16(roi_feat, name='roi_feat_to_fp16')
        return roi_feat


class TopDownBottomUpFPNNeckP2P6(NASFPNNeck):
    def __init__(self, pNeck):
        super().__init__(pNeck)
        self.neck = None

    @staticmethod
    def get_fused_P_feature(p_features, stage, dim_reduced, init, norm):
        prefix = "S{}_".format(stage)
        with mx.name.Prefix(prefix):
            P2_0 = p_features['S{}_P2'.format(stage-1)] # s8
            P3_0 = p_features['S{}_P3'.format(stage-1)] # s8
            P4_0 = p_features['S{}_P4'.format(stage-1)] # s16
            P5_0 = p_features['S{}_P5'.format(stage-1)] # s32
            P6_0 = p_features['S{}_P6'.format(stage-1)] # s64

            P6_1 = P6_0
            P6_1_to_P5 = mx.sym.UpSampling(
                P6_1,
                scale=2,
                sample_type='nearest',
                name="P6_1_to_P5",
                num_args=1
            )
            P6_1_to_P5 = mx.sym.slice_like(P6_1_to_P5, P5_0)
            P5_1 = merge_sum(P5_0, P6_1_to_P5, name="sum_P5_0_P6_1")
            P5_1 = reluconvbn(P5_1, dim_reduced, init, norm, name="P5_1", prefix=prefix)

            # P4_1 = sum(P4_0, P5_1)
            P5_1_to_P4 = mx.sym.UpSampling(
                P5_1,
                scale=2,
                sample_type='nearest',
                name="P5_1_to_P4",
                num_args=1
            )
            P5_1_to_P4 = mx.sym.slice_like(P5_1_to_P4, P4_0)
            P4_1 = merge_sum(P4_0, P5_1_to_P4, name="sum_P4_0_P5_1")
            P4_1 = reluconvbn(P4_1, dim_reduced, init, norm, name="P4_1", prefix=prefix)

            P4_1_to_P3 = mx.sym.UpSampling(
                P4_1,
                scale=2,
                sample_type='nearest',
                name="P4_1_to_P3",
                num_args=1
            )
            P4_1_to_P3 = mx.sym.slice_like(P4_1_to_P3, P3_0)
            P3_1 = merge_sum(P3_0, P4_1_to_P3, name="sum_P3_0_P4_1")
            P3_1 = reluconvbn(P3_1, dim_reduced, init, norm, name="P3_1", prefix=prefix)

            P3_1_to_P2 = mx.sym.UpSampling(
                P3_1,
                scale=2,
                sample_type='nearest',
                name="P3_1_to_P2",
                num_args=1
            )
            P3_1_to_P2 = mx.sym.slice_like(P3_1_to_P2, P2_0)
            P2_1 = merge_sum(P2_0, P3_1_to_P2, name="sum_P2_0_P3_1")
            P2_1 = reluconvbn(P2_1, dim_reduced, init, norm, name="P2_1", prefix=prefix)

            P2_2 = P2_1
            P2 = P2_2

            P2_2_to_P3 = X.pool(P2_2, name="P2_2_to_P3", kernel=3, stride=2, pad=1)   # NOTE
            P3_2 = merge_sum(P3_1, P2_2_to_P3, name="sum_P3_1_P2_2")
            P3_2 = reluconvbn(P3_2, dim_reduced, init, norm, name="P3_2", prefix=prefix)
            P3 = P3_2

            P3_2_to_P4 = X.pool(P3_2, name="P3_2_to_P4", kernel=3, stride=2, pad=1)  # NOTE
            P4_2 = merge_sum(P4_1, P3_2_to_P4, name="sum_P4_1_P3_2")
            P4_2 = reluconvbn(P4_2, dim_reduced, init, norm, name="P4_2", prefix=prefix)
            P4 = P4_2

            P4_2_to_P5 = X.pool(P4_2, name="P4_2_to_P5", kernel=3, stride=2, pad=1)   # NOTE
            P5_2 = merge_sum(P5_1, P4_2_to_P5, name="sum_P5_1_P4_2")
            P5_2 = reluconvbn(P5_2, dim_reduced, init, norm, name="P5_2", prefix=prefix)
            P5 = P5_2

            P5_2_to_P6 = X.pool(P5_2, name="P5_2_to_P6", kernel=3, stride=2, pad=1)  # NOTE
            P6_2 = merge_sum(P6_1, P5_2_to_P6, name="sum_P6_1_P5_2")
            P6_2 = reluconvbn(P6_2, dim_reduced, init, norm, name="P6_2", prefix=prefix)
            P6 = P6_2

            return {
                'S{}_P2'.format(stage): P2,
                'S{}_P3'.format(stage): P3,
                'S{}_P4'.format(stage): P4,
                'S{}_P5'.format(stage): P5,
                'S{}_P6'.format(stage): P6,
                }

    def get_pafpn_neck(self, data):
        if self.neck is not None:
            return self.neck

        dim_reduced = self.p.dim_reduced
        norm = self.p.normalizer
        num_stage = self.p.num_stage
        S0_kernel = self.p.S0_kernel

        import mxnet as mx
        xavier_init = mx.init.Xavier(factor_type="avg", rnd_type="uniform", magnitude=3)

        c2, c3, c4, c5 = data
        c6 = X.pool(data=c5, name='C6', kernel=3, stride=2, pad=1)  # NOTE

        c_features = [c2, c3, c4, c5, c6]
        # 0 stage
        p0_names = ['S0_P2', 'S0_P3', 'S0_P4', 'S0_P5', 'S0_P6']
        p_features = self.get_P0_features(c_features, p0_names, dim_reduced, xavier_init, norm, S0_kernel)

        # stack stage
        for i in range(num_stage):
            p_features = self.get_fused_P_feature(p_features, i + 1, dim_reduced, xavier_init, norm)

        self.neck = dict(
            stride4=p_features['S{}_P2'.format(num_stage)],
            stride8=p_features['S{}_P3'.format(num_stage)],
            stride16=p_features['S{}_P4'.format(num_stage)],
            stride32=p_features['S{}_P5'.format(num_stage)],
            stride64=p_features['S{}_P6'.format(num_stage)],
        )

        return self.neck
    
    def get_rpn_feature(self, rpn_feat):
        return self.get_pafpn_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.get_pafpn_neck(rcnn_feat)

    
