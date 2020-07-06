import mxnext as X 
import mxnet as mx
from mxnext import conv, relu


def DeformConv(x, offset, name, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
               groups=1, deformable_groups=1, no_bias=False, weight=None, bias=None):
    assert weight is not None
    if not no_bias:
        assert bias is not None
    assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
    out = mx.sym.contrib.DeformableConvolution(
            x, offset, weight=weight, bias=bias if not no_bias else None, kernel=(kernel_size,kernel_size), 
            stride=(stride,stride), dilate=(dilation,dilation), pad=(padding,padding), num_filter=out_channels, 
            num_group=groups, num_deformable_group=deformable_groups, no_bias=no_bias, name=name)
    return out


def sepc_conv(x, name, out_channels, kernel_size, i, stride=1, padding=0, dilation=1, 
                groups=1, deformable_groups=1, part_deform=False, start_level=1,
                weight=None, bias=None, weight_offset=None, bias_offset=None):
    assert weight is not None and bias is not None
    if part_deform:
        assert weight_offset is not None and bias_offset is not None
    if i < start_level or not part_deform:
        return conv(x, name, filter=out_channels, kernel=kernel_size, stride=stride, pad=kernel_size//2,
                    dilate=dilation, num_group=groups, no_bias=False, weight=weight, bias=bias)
    offset = conv(x, name+'offset', filter=deformable_groups*2*kernel_size*kernel_size, kernel=kernel_size, stride=stride, 
                  pad=kernel_size//2, dilate=dilation, num_group=groups, no_bias=False, weight=weight_offset, bias=bias_offset)
    return DeformConv(x, offset, name, out_channels, kernel_size, stride, padding=padding, dilation=dilation,
                      groups=groups, deformable_groups=deformable_groups, no_bias=False, weight=weight, bias=bias)