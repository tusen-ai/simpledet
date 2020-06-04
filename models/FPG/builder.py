import mxnet as mx
import mxnext as X
from symbol.builder import RoiAlign, Neck
from models.NASFPN.builder import reluconvbn

def merge_sum(sum_list, name):
    return mx.sym.ElementWiseSum(*sum_list, name=name + '_sum')

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
            p_features.append(p) # NOTE list[p2, p3, p4, p5, p6]
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
                    fusion_list.append(self.same_up(bottom_ind_x, bottom_ind_y, dim_reduced))
                
                # bottom left feature (Across Up)
                bottom_ind_x = stage - 1; bottom_ind_y = index - 1
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_up(bottom_ind_x, bottom_ind_y, dim_reduced))
                
                # left feature (Across Same)
                bottom_ind_x = stage - 1; bottom_ind_y = index
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_same(bottom_ind_x, bottom_ind_y, dim_reduced))

                # upper left feature (Across Down)
                bottom_ind_x = stage - 1; bottom_ind_y = index + 1
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_down(bottom_ind_x, bottom_ind_y, dim_reduced))

                # stage 0 feature (Across Skip)
                bottom_ind_x = 0; bottom_ind_y = index
                if bottom_ind_x >=0 and bottom_ind_y >= 0 and bottom_ind_y < len(level_list):
                    fusion_list.append(self.across_skip(bottom_ind_x, bottom_ind_y, dim_reduced))

                fusion_feature = merge_sum(fusion_list, name='sum_P%s_%s' % (level, stage))
                feature_grid = reluconvbn(fusion_feature, dim_reduced, init, norm, name='P%s_%s'%(level, stage), prefix=prefix)
                self.feature_grids[stage].append(feature_grid)

    def across_down(self, stage_num, level_num, dim_reduced):
        feature = self.feature_grids[stage_num][level_num]
        upsampling_feature = mx.sym.UpSampling(
            feature,
            scale=2,
            sample_type='nearest',
            name='P%s_%s_acrossdown_upsample' % (level_num, stage_num),
            num_args=1,
        )
        output_feature = X.conv(
            data=upsampling_feature,
            kernel=3,
            pad=1,
            stride=1,
            filter=dim_reduced,
            name='P%s_%s_acrossdown_conv' % (level_num, stage_num),
        )
        p0_feature = self.feature_grids[0][level_num-1]
        output_feature = mx.sym.slice_like(output_feature, p0_feature, name='P%s_%s_slicelike'%(stage_num, level_num))
        return output_feature
    
    def across_same(self, stage_num, level_num, dim_reduced):
        feature = self.feature_grids[stage_num][level_num]
        across_same_feature = X.conv(
            data=feature,
            kernel=1,
            pad=0,
            stride=1,
            filter=dim_reduced,
            name='P%s_%s_acrosssame_conv' % (level_num, stage_num),
        )
        return across_same_feature

    def across_up(self, stage_num, level_num, dim_reduced):
        feature = self.feature_grids[stage_num][level_num]
        across_up_feature = X.conv(
            data=feature,
            kernel=3,
            stride=2,
            pad=1,
            filter=dim_reduced,
            name='P%s_%s_acrossup_conv' % (level_num, stage_num),
        )
        return across_up_feature

    def same_up(self, stage_num, level_num, dim_reduced):
        feature = self.feature_grids[stage_num][level_num]
        same_up_feature = X.conv(
            data=feature,
            kernel=3,
            stride=2,
            pad=1,
            filter=dim_reduced,
            name='P%s_%s_sameup_conv' % (level_num, stage_num)
        )
        return same_up_feature

    def across_skip(self, stage_num, level_num, dim_reduced):
        feature = self.feature_grids[stage_num][level_num]
        across_skip_feature = X.conv(
            data=feature,
            kernel=1,
            pad=0,
            stride=1,
            filter=dim_reduced,
            name='P%s_%s_acrossskip_conv' % (level_num, stage_num),
        )
        return across_skip_feature

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
