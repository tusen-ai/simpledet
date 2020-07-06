import mxnet as mx
import mxnext as X
from symbol.builder import Neck

class FPGNeck(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)
        self.feature_grids = []
        self.neck = None

    @staticmethod
    def get_P0_features(c_features, p_names, dim_reduced, init, norm, kernel=1):
        p_features = []
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
            p_features.append(p)
        return p_features

    def get_fused_P_feature(self, stage, dim_reduced, init, norm):
        prefix = "S{}_".format(stage)
        level_list = [2, 3, 4, 5, 6]
        feature_list = []
        self.feature_grids.append(feature_list)
        with mx.name.Prefix(prefix):
            for index, level in enumerate(level_list):
                fusion_list = []
                # bottom feature (Same Up)
                bottom_ind_x = stage; bottom_ind_y = index - 1
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.same_up(bottom_ind_x, bottom_ind_y, dim_reduced, init, norm))
                
                # bottom left feature (Across Up)
                bottom_ind_x = stage - 1; bottom_ind_y = index - 1
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_up(bottom_ind_x, bottom_ind_y, dim_reduced, init, norm))
                
                # left feature (Across Same)
                bottom_ind_x = stage - 1; bottom_ind_y = index
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_same(bottom_ind_x, bottom_ind_y, dim_reduced, init, norm))

                # upper left feature (Across Down)
                bottom_ind_x = stage - 1; bottom_ind_y = index + 1
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_down(bottom_ind_x, bottom_ind_y, dim_reduced, init, norm))

                # stage 0 feature (Across Skip)
                bottom_ind_x = 0; bottom_ind_y = index
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_skip(bottom_ind_x, bottom_ind_y, stage, dim_reduced, init, norm))
                
                name = "S%s_P%s"%(stage, level)
                fusion_feature = X.merge_sum(fusion_list, name='sum_' + name)
                feature_grid = X.conv(
                    data=fusion_feature,
                    filter=dim_reduced,
                    kernel=3,
                    pad=1,
                    stride=1,
                    no_bias=False,
                    weight=X.var(name=name + "_weight", init=init),
                    bias=X.var(name=name + "_bias", init=X.zero_init()),
                    name=name + '_conv',
                )
                self.feature_grids[stage].append(feature_grid)
        


    def across_down(self, stage_num, level_num, dim_reduced, init, norm):
        feature = self.feature_grids[stage_num][level_num]
        feature = mx.sym.UpSampling(
            feature,
            scale=2,
            sample_type='nearest',
            name='P%s_%s_acrossdown_upsample' % (level_num, stage_num),
            num_args=1,
        )
        feature = X.reluconvbn(data=feature,filters=dim_reduced, kernel=3, pad=1, stride=1, init=init, norm=norm,\
                                name='P%s_acrossdown_conv'%level_num, prefix='S%s_'%stage_num)
        p0_feature = self.feature_grids[0][level_num-1]
        feature = mx.sym.slice_like(feature, p0_feature, name='P%s_%s_slicelike'%(stage_num, level_num))
        return feature
    
    def across_same(self, stage_num, level_num, dim_reduced, init, norm):
        feature = self.feature_grids[stage_num][level_num]
        feature = X.reluconvbn(data=feature, filters=dim_reduced, kernel=1, pad=0, stride=1, init=init, norm=norm,\
                                name='P%s_acrosssame_conv'%level_num, prefix='S%s_'%stage_num)
        return feature

    def across_up(self, stage_num, level_num, dim_reduced, init, norm):
        feature = self.feature_grids[stage_num][level_num]
        feature = X.reluconvbn(data=feature,filters=dim_reduced, kernel=3, pad=1, stride=2, init=init, norm=norm,\
                                name='P%s_acrossup_conv'%level_num, prefix='S%s_'%stage_num)
        return feature

    def same_up(self, stage_num, level_num, dim_reduced, init, norm):
        feature = self.feature_grids[stage_num][level_num]
        feature = X.reluconvbn(data=feature,filters=dim_reduced, kernel=3, pad=1, stride=2, init=init, norm=norm,\
                                name='P%s_sameup_conv'%level_num, prefix='S%s_'%stage_num)
        return feature

    def across_skip(self, stage_num, level_num, curr_stage_num, dim_reduced, init, norm):
        feature = self.feature_grids[stage_num][level_num]
        feature = X.reluconvbn(data=feature,filters=dim_reduced, kernel=1, pad=0, stride=1, init=init, norm=norm,\
                                name='P%s_acrossskip_conv'%level_num, prefix='S%s_2_%s'%(stage_num, curr_stage_num))
        return feature

    def get_rpn_feature(self, rpn_feat):
        return self.get_fpg_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.get_fpg_neck(rcnn_feat)

class FPGNeckP2P6(FPGNeck):
    def __init__(self, pNeck):
        super().__init__(pNeck)

    def get_fpg_neck(self, data):
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
        self.feature_grids.append(p_features)

        # stack stage
        for i in range(num_stage):
            self.get_fused_P_feature(i + 1, dim_reduced, xavier_init, norm)
        
        self.neck = dict(
            stride4=self.feature_grids[-1][0],  #P2
            stride8=self.feature_grids[-1][1],  #P3
            stride16=self.feature_grids[-1][2], #P4
            stride32=self.feature_grids[-1][3], #P5
            stride64=self.feature_grids[-1][4], #P6
        )

        return self.neck

class PAFPNNeck(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)
        self.neck = None
    
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
            P2_0 = p_features['S{}_P2'.format(stage-1)] # s4
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
            P5_1 = X.merge_sum([P5_0, P6_1_to_P5], name="sum_P5_0_P6_1")
            P5_1 = X.reluconvbn(P5_1, dim_reduced, init, norm, name="P5_1", prefix=prefix)

            # P4_1 = sum(P4_0, P5_1)
            P5_1_to_P4 = mx.sym.UpSampling(
                P5_1,
                scale=2,
                sample_type='nearest',
                name="P5_1_to_P4",
                num_args=1
            )
            P5_1_to_P4 = mx.sym.slice_like(P5_1_to_P4, P4_0)
            P4_1 = X.merge_sum([P4_0, P5_1_to_P4], name="sum_P4_0_P5_1")
            P4_1 = X.reluconvbn(P4_1, dim_reduced, init, norm, name="P4_1", prefix=prefix)

            P4_1_to_P3 = mx.sym.UpSampling(
                P4_1,
                scale=2,
                sample_type='nearest',
                name="P4_1_to_P3",
                num_args=1
            )
            P4_1_to_P3 = mx.sym.slice_like(P4_1_to_P3, P3_0)
            P3_1 = X.merge_sum([P3_0, P4_1_to_P3], name="sum_P3_0_P4_1")
            P3_1 = X.reluconvbn(P3_1, dim_reduced, init, norm, name="P3_1", prefix=prefix)

            P3_1_to_P2 = mx.sym.UpSampling(
                P3_1,
                scale=2,
                sample_type='nearest',
                name="P3_1_to_P2",
                num_args=1
            )
            P3_1_to_P2 = mx.sym.slice_like(P3_1_to_P2, P2_0)
            P2_1 = X.merge_sum([P2_0, P3_1_to_P2], name="sum_P2_0_P3_1")
            P2_1 = X.reluconvbn(P2_1, dim_reduced, init, norm, name="P2_1", prefix=prefix)

            P2_2 = P2_1
            P2 = P2_2

            P2_2_to_P3 = X.pool(P2_2, name="P2_2_to_P3", kernel=3, stride=2, pad=1)
            P3_2 = X.merge_sum([P3_1, P2_2_to_P3], name="sum_P3_1_P2_2")
            P3_2 = X.reluconvbn(P3_2, dim_reduced, init, norm, name="P3_2", prefix=prefix)
            P3 = P3_2

            P3_2_to_P4 = X.pool(P3_2, name="P3_2_to_P4", kernel=3, stride=2, pad=1)
            P4_2 = X.merge_sum([P4_1, P3_2_to_P4], name="sum_P4_1_P3_2")
            P4_2 = X.reluconvbn(P4_2, dim_reduced, init, norm, name="P4_2", prefix=prefix)
            P4 = P4_2

            P4_2_to_P5 = X.pool(P4_2, name="P4_2_to_P5", kernel=3, stride=2, pad=1)
            P5_2 = X.merge_sum([P5_1, P4_2_to_P5], name="sum_P5_1_P4_2")
            P5_2 = X.reluconvbn(P5_2, dim_reduced, init, norm, name="P5_2", prefix=prefix)
            P5 = P5_2

            P5_2_to_P6 = X.pool(P5_2, name="P5_2_to_P6", kernel=3, stride=2, pad=1)
            P6_2 = X.merge_sum([P6_1, P5_2_to_P6], name="sum_P6_1_P5_2")
            P6_2 = X.reluconvbn(P6_2, dim_reduced, init, norm, name="P6_2", prefix=prefix)
            P6 = P6_2

            return {
                'S{}_P2'.format(stage): P2,
                'S{}_P3'.format(stage): P3,
                'S{}_P4'.format(stage): P4,
                'S{}_P5'.format(stage): P5,
                'S{}_P6'.format(stage): P6,
                }
    
    def get_rpn_feature(self, rpn_feat):
        return self.get_pafpn_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):  
        return self.get_pafpn_neck(rcnn_feat)

class PAFPNNeckP2P6(PAFPNNeck):
    def __init__(self, pNeck):
        super().__init__(pNeck)

    def get_pafpn_neck(self, data):
        if self.neck is not None:
            return self.neck

        dim_reduced = self.p.dim_reduced
        norm = self.p.normalizer
        num_stage = self.p.num_stage
        S0_kernel = self.p.S0_kernel

        xavier_init = mx.init.Xavier(factor_type="avg", rnd_type="uniform", magnitude=3)

        c2, c3, c4, c5 = data
        c6 = X.pool(data=c5, name='C6', kernel=3, stride=2, pad=1)

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

class PAFPNNeckP3P7(PAFPNNeck):
    def __init__(self, pNeck):
        super().__init__(pNeck)

    @staticmethod
    def get_fused_P_feature(p_features, stage, dim_reduced, init, norm):
        prefix = "S{}_".format(stage)
        with mx.name.Prefix(prefix):
            P3_0 = p_features['S{}_P3'.format(stage-1)] # s8
            P4_0 = p_features['S{}_P4'.format(stage-1)] # s16
            P5_0 = p_features['S{}_P5'.format(stage-1)] # s32
            P6_0 = p_features['S{}_P6'.format(stage-1)] # s64
            P7_0 = p_features['S{}_P7'.format(stage-1)] # s128

            P7_1 = P7_0
            P7_1_to_P6 = mx.sym.UpSampling(
                P7_1,
                scale=2,
                sample_type='nearest',
                name="P7_1_to_P6",
                num_args=1
            )
            P7_1_to_P6 = mx.sym.slice_like(P7_1_to_P6, P6_0)
            P6_1 = X.merge_sum([P6_0, P7_1_to_P6], name="sum_P6_0_P7_1")
            P6_1 = X.reluconvbn(P6_1, dim_reduced, init, norm, name="P6_1", prefix=prefix)

            P6_1_to_P5 = mx.sym.UpSampling(
                P6_1,
                scale=2,
                sample_type='nearest',
                name="P6_1_to_P5",
                num_args=1
            )
            P6_1_to_P5 = mx.sym.slice_like(P6_1_to_P5, P5_0)
            P5_1 = X.merge_sum([P5_0, P6_1_to_P5], name="sum_P5_0_P6_1")
            P5_1 = X.reluconvbn(P5_1, dim_reduced, init, norm, name="P5_1", prefix=prefix)

            P5_1_to_P4 = mx.sym.UpSampling(
                P5_1,
                scale=2,
                sample_type='nearest',
                name="P5_1_to_P4",
                num_args=1
            )
            P5_1_to_P4 = mx.sym.slice_like(P5_1_to_P4, P4_0)
            P4_1 = X.merge_sum([P4_0, P5_1_to_P4], name="sum_P4_0_P5_1")
            P4_1 = X.reluconvbn(P4_1, dim_reduced, init, norm, name="P4_1", prefix=prefix)

            P4_1_to_P3 = mx.sym.UpSampling(
                P4_1,
                scale=2,
                sample_type='nearest',
                name="P4_1_to_P3",
                num_args=1
            )
            P4_1_to_P3 = mx.sym.slice_like(P4_1_to_P3, P3_0)
            P3_1 = X.merge_sum([P3_0, P4_1_to_P3], name="sum_P3_0_P4_1")
            P3_1 = X.reluconvbn(P3_1, dim_reduced, init, norm, name="P3_1", prefix=prefix)

            P3_2 = P3_1
            P3 = P3_2

            P3_2_to_P4 = X.pool(P3_2, name="P3_2_to_P4", kernel=3, stride=2, pad=1)
            P4_2 = X.merge_sum([P4_1, P3_2_to_P4], name="sum_P4_1_P3_2")
            P4_2 = X.reluconvbn(P4_2, dim_reduced, init, norm, name="P4_2", prefix=prefix)
            P4 = P4_2

            P4_2_to_P5 = X.pool(P4_2, name="P4_2_to_P5", kernel=3, stride=2, pad=1)
            P5_2 = X.merge_sum([P5_1, P4_2_to_P5], name="sum_P5_1_P4_2")
            P5_2 = X.reluconvbn(P5_2, dim_reduced, init, norm, name="P5_2", prefix=prefix)
            P5 = P5_2

            P5_2_to_P6 = X.pool(P5_2, name="P5_2_to_P6", kernel=3, stride=2, pad=1)
            P6_2 = X.merge_sum([P6_1, P5_2_to_P6], name="sum_P6_1_P5_2")
            P6_2 = X.reluconvbn(P6_2, dim_reduced, init, norm, name="P6_2", prefix=prefix)
            P6 = P6_2

            P6_2_to_P7 = X.pool(P6_2, name="P6_2_to_P7", kernel=3, stride=2, pad=1)
            P7_2 = X.merge_sum([P7_1, P6_2_to_P7], name="sum_P7_1_P6_2")
            P7_2 = X.reluconvbn(P7_2, dim_reduced, init, norm, name="P7_2", prefix=prefix)
            P7 = P7_2

            return {
                'S{}_P3'.format(stage): P3,
                'S{}_P4'.format(stage): P4,
                'S{}_P5'.format(stage): P5,
                'S{}_P6'.format(stage): P6,
                'S{}_P7'.format(stage): P7,
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
        c6 = X.pool(data=c5, name="C6", kernel=2, stride=2, pad=0)
        c7 = X.pool(data=c5, name="C7", kernel=4, stride=4, pad=0)

        c_features = [c3, c4, c5, c6, c7]
        # 0 stage
        p0_names = ['S0_P3', 'S0_P4', 'S0_P5', 'S0_P6', 'S0_P7']
        p_features = self.get_P0_features(c_features, p0_names, dim_reduced, xavier_init, norm, S0_kernel)
        
        # stack stage
        for i in range(num_stage):
            p_features = self.get_fused_P_feature(p_features, i + 1, dim_reduced, xavier_init, norm)

        self.neck = dict(
            stride8=p_features['S{}_P3'.format(num_stage)],
            stride16=p_features['S{}_P4'.format(num_stage)],
            stride32=p_features['S{}_P5'.format(num_stage)],
            stride64=p_features['S{}_P6'.format(num_stage)],
            stride128=p_features['S{}_P7'.format(num_stage)]
        )
        
        return self.neck