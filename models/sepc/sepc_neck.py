import mxnet as mx
import mxnext as X 
from mxnext import conv, relu
from functools import partial
from models.sepc.sepc_dconv import sepc_conv
from mxnext.complicate import normalizer_factory
import numpy as np


def SEPCFPN(inputs, out_channels=256, pconv_deform=False, lcconv_deform=None, ibn=None, Pconv_num=4,
            start_level=1, norm=None, bilinear_upsample=None, feat_sizes=None):
    assert feat_sizes is not None
    Pconvs_list = []
    for i in range(Pconv_num):
        Pconvs_list.append(partial(
            PConvModule, out_channels=out_channels, ibn=ibn, part_deform=pconv_deform, 
            PConv_idx=i, start_level=start_level, norm=norm, bilinear_upsample=bilinear_upsample, feat_sizes=feat_sizes))
    
    if lcconv_deform is not None:
        assert lcconv_deform in [False, True]
        lconv_weight, lconv_bias = X.var(name='LConv_weight', init=X.gauss(std=0.01)), X.var(name='LConv_bias',init=X.zero_init())
        cconv_weight, cconv_bias = X.var(name='CConv_weight', init=X.gauss(std=0.01)), X.var(name='CConv_bias',init=X.zero_init())
        lconv_offset_weight, lconv_offset_bias = None, None
        cconv_offset_weight, cconv_offset_bias = None, None
        if lcconv_deform:
            lconv_offset_weight, lconv_offset_bias=X.var(name='LConv_offset_weight', init=X.zero_init()), X.var(name='LConv_offset_bias', init=X.zero_init())
            cconv_offset_weight, cconv_offset_bias=X.var(name='CConv_offset_weight', init=X.zero_init()), X.var(name='CConv_offset_bias', init=X.zero_init())
        lconv_func = partial(sepc_conv, name='LConv{}_',out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                dilation=1, groups=1, deformable_groups=1, part_deform=lcconv_deform, start_level=start_level,
                weight=lconv_weight, bias=lconv_bias, weight_offset=lconv_offset_weight, bias_offset=lconv_offset_bias)
        cconv_func = partial(sepc_conv, name='CConv{}_', out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                dilation=1, groups=1, deformable_groups=1, part_deform=lcconv_deform, start_level=start_level,
                weight=cconv_weight, bias=cconv_bias, weight_offset=cconv_offset_weight, bias_offset=cconv_offset_bias)

        if ibn:
            assert norm is not None
            lbn = partial(norm, name='lconv_ibn')
            cbn = partial(norm, name='cconv_ibn')

    x = inputs
    for pconv in Pconvs_list:
        x = pconv(x)
    if lcconv_deform is None:
        return x
    cls_outs = [cconv_func(i=level, x=item) for level, item in enumerate(x)]
    loc_outs = [lconv_func(i=level, x=item) for level, item in enumerate(x)]
    if ibn:
        cls_outs = ibn_func(cls_outs, cbn, feat_sizes)
        loc_outs = ibn_func(loc_outs, lbn, feat_sizes)
    outs = [mx.sym.Concat(*[relu(s), relu(l)], num_args=2, dim=1) for s, l in zip(cls_outs, loc_outs)]
    return outs
    

def PConvModule(x, out_channels=256, kernel_size=[3, 3, 3], dilation=[1, 1, 1], groups=[1, 1, 1], ibn=None,
                part_deform=False, PConv_idx=-1, start_level=1, norm=None, bilinear_upsample=None, feat_sizes=None):
    assert PConv_idx > -1 and feat_sizes is not None
    name_pref = 'PConv{}_sepc'.format(PConv_idx)
    sepc0_weight, sepc0_bias = X.var(name=name_pref+'0_weight', init=X.gauss(std=0.01)), X.var(name=name_pref+'0_bias', init=X.zero_init())
    sepc1_weight, sepc1_bias = X.var(name=name_pref+'1_weight', init=X.gauss(std=0.01)), X.var(name=name_pref+'1_bias', init=X.zero_init())
    sepc2_weight, sepc2_bias = X.var(name=name_pref+'2_weight', init=X.gauss(std=0.01)), X.var(name=name_pref+'2_bias', init=X.zero_init())
    sepc0_offset_weight, sepc0_offset_bias = None, None
    sepc1_offset_weight, sepc1_offset_bias = None, None
    sepc2_offset_weight, sepc2_offset_bias = None, None
    if part_deform:
        # NOTE zero_init for offset's weight and bias
        sepc0_offset_weight, sepc0_offset_bias = X.var(name=name_pref+'0_offset_weight', init=X.zero_init()), X.var(name=name_pref+'0_offset_bias', init=X.zero_init())
        sepc1_offset_weight, sepc1_offset_bias = X.var(name=name_pref+'1_offset_weight', init=X.zero_init()), X.var(name=name_pref+'1_offset_bias', init=X.zero_init())
        sepc2_offset_weight, sepc2_offset_bias = X.var(name=name_pref+'2_offset_weight', init=X.zero_init()), X.var(name=name_pref+'2_offset_bias', init=X.zero_init())
    norm_func = []
    if ibn:
        assert norm is not None
        norm_func = partial(norm, name=name_pref+'_ibn')

    sepc_conv0_func = partial(
                sepc_conv, name='PConv{}_sepc0_'.format(PConv_idx), out_channels=out_channels,
                kernel_size=kernel_size[0], stride=1, padding=(kernel_size[0]+(dilation[0]-1)*2)//2,
                dilation=dilation[0], groups=groups[0], deformable_groups=1, part_deform=part_deform, start_level=start_level,
                weight=sepc0_weight, bias=sepc0_bias, weight_offset=sepc0_offset_weight, bias_offset=sepc0_offset_bias)
    sepc_conv1_func = partial(
                sepc_conv, name='PConv{}_sepc1_'.format(PConv_idx), out_channels=out_channels,
                kernel_size=kernel_size[1], stride=1, padding=(kernel_size[1]+(dilation[1]-1)*2)//2,
                dilation=dilation[1], groups=groups[1], deformable_groups=1, part_deform=part_deform, start_level=start_level,
                weight=sepc1_weight, bias=sepc1_bias, weight_offset=sepc1_offset_weight, bias_offset=sepc1_offset_bias)
    sepc_conv2_func = partial(
                sepc_conv, name='PConv{}_sepc2_'.format(PConv_idx), out_channels=out_channels,
                kernel_size=kernel_size[2], stride=2, padding=(kernel_size[2]+(dilation[2]-1)*2)//2,
                dilation=dilation[2], groups=groups[2], deformable_groups=1, part_deform=part_deform, start_level=start_level,
                weight=sepc2_weight, bias=sepc2_bias, weight_offset=sepc2_offset_weight, bias_offset=sepc2_offset_bias)
    next_x = []
    for level, feature in enumerate(x):
        temp_fea = sepc_conv1_func(i=level, x=feature)
        if level > 0:
            tmp = sepc_conv2_func(i=level, x=x[level - 1])
            temp_fea = temp_fea + tmp
        if level < len(x) - 1:
            tmp_x = sepc_conv0_func(i=level,x=x[level+1])
            if bilinear_upsample:
                tmp_x = mx.contrib.symbol.BilinearResize2D(tmp_x, scale_height=2, scale_width=2,
                    name='PConv{}_upsampling_level{}'.format(PConv_idx,level))
            else:
                tmp_x = mx.sym.UpSampling(tmp_x, scale=2, sample_type='nearest', num_args=1,
                    name='PConv{}_upsampling_level{}'.format(PConv_idx,level))
            tmp_x = mx.sym.slice_like(tmp_x, temp_fea)
            temp_fea = temp_fea + tmp_x
        next_x.append(temp_fea)
    if ibn:
        next_x = ibn_func(next_x, norm_func, feat_sizes)
    next_x = [relu(item, name='PConv{}_level{}_relu'.format(PConv_idx, level)) for level,item in enumerate(next_x)]
    return next_x


def ibn_func(fms, bn, feat_sizes):
    sizes = feat_sizes
    sizes_accu = np.cumsum([e0*e1 for (e0,e1) in sizes]).tolist()
    fm = mx.sym.Concat(*[mx.sym.reshape(p, shape=(0,0,1,-1)) for p in fms], num_args=len(fms), dim=-1)
    fm = bn(fm)
    fm_splited = [mx.sym.slice_axis(fm, axis=-1, begin=sizes_accu[i-1] if i>=1 else 0, end=sizes_accu[i]) for i in range(len(sizes))]
    fm_outs = [mx.sym.reshape_like(p, fms[i]) for i,p in enumerate(fm_splited)]
    return fm_outs