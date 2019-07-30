import mxnet as mx
import mxnext as X
from mxnext import dwconv, conv, relu6, add, global_avg_pool, sigmoid, to_fp16, to_fp32
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone


def _make_divisible(dividend, divisor):
    if dividend % divisor == 0:
        return dividend
    else:
        return (dividend // divisor + 1) * divisor

round32 = lambda dividend: _make_divisible(dividend, 32)


def se(input, prefix, f_down, f_up):
    with mx.name.Prefix(prefix + "_"):
        gap = mx.sym.mean(input, axis=-1, keepdims=True)
        gap = mx.sym.mean(gap, axis=-2, keepdims=True)
        fc1 = conv(gap, name="fc1", filter=f_down)
        fc1 = relu6(fc1, name="fc1_relu")
        fc2 = conv(fc1, name="fc2", filter=f_up)
        att = sigmoid(fc2, name="sigmoid")
        input = mx.sym.broadcast_mul(input, att, name="mul")

    return input


def convnormrelu(input, prefix, kernel, f_in, f_out, stride, proj, norm, **kwargs):
    with mx.name.Prefix(prefix + "_"):
        conv1 = conv(input, name="conv1", filter=f_out, kernel=kernel, stride=stride, no_bias=False)
        bn1 = norm(conv1, name="bn1")
        relu1 = relu6(bn1, name="relu1")
    return relu1


def mbconv(input, prefix, kernel, f_in, f_out, stride, proj, bottleneck_ratio, norm, **kwargs):
    with mx.name.Prefix(prefix + "_"):
        if bottleneck_ratio != 1:
            conv1 = conv(input, name="conv1", filter=f_in * bottleneck_ratio, no_bias=False)
            bn1 = norm(conv1, name="bn1")
            relu1 = relu6(bn1, name="relu1")
        else:
            relu1 = input

        conv2 = dwconv(relu1, name="conv2", filter=f_in * bottleneck_ratio,
            kernel=kernel, stride=stride, no_bias=False)
        bn2 = norm(conv2, name="bn2")
        relu2 = relu6(bn2, name="relu2")
        relu2 = se(relu2, prefix=prefix + "_se2", f_down=f_in//4, f_up=f_in * bottleneck_ratio)

        conv3 = conv(relu2, name="conv3", filter=f_out, no_bias=False)
        bn3 = norm(conv3, name="bn3")

        if proj:
            return bn3
        else:
            return bn3 + input


mbc1 = lambda input, prefix, kernel, f_in, f_out, stride, proj, norm, **kwargs: \
    mbconv(input, prefix, kernel, f_in, f_out, stride, proj, 1, norm, **kwargs)
mbc6 = lambda input, prefix, kernel, f_in, f_out, stride, proj, norm, **kwargs: \
    mbconv(input, prefix, kernel, f_in, f_out, stride, proj, 6, norm, **kwargs)


def efficientnet_helper(data, norm, us, fos, fis, ss, ks, cs):
    stages = []
    for i, (u, fo, fi, s, k, c) in enumerate(zip(us, fos, fis, ss, ks, cs), start=1):
        for j in range(1, u + 1):
            s = s if j == 1 else 1
            proj = True if j == 1 else False
            fi = fi if j == 1 else fo
            data = c(data, prefix="stage%s_unit%s" % (i, j), f_in=fi, f_out=fo,
                kernel=k, stride=s, proj=proj, norm=norm)
        stages.append(data)
    return stages


def efficientnet_b4(data, norm, **kwargs):
    # 1.5 GFLOPs
    us = [1, 2, 4, 4, 6, 6, 8, 2, 1]
    fos = [48, 24, 32, 56, 112, 160, 272, 448, 1792]
    fis = [0] + fos[:-1]
    ss = [2, 1, 2, 2, 2, 1, 2, 1, 1]
    ks = [3, 3, 3, 5, 3, 5, 5, 3, 1]
    cs = [convnormrelu, mbc1, mbc6, mbc6, mbc6, mbc6, mbc6, mbc6, convnormrelu]
    return efficientnet_helper(data, norm, us, fos, fis, ss, ks, cs)


def efficientnet_b5(data, norm, **kwargs):
    # 2.3 GFLOPs
    us = [1, 3, 5, 5, 7, 7, 9, 3, 1]
    fos = [48, 24, 40, 64, 128, 172, 304, 512, 2048]
    fis = [0] + fos[:-1]
    ss = [2, 1, 2, 2, 2, 1, 2, 1, 1]
    ks = [3, 3, 3, 5, 3, 5, 5, 3, 1]
    # ks = [3, 5, 5, 5, 5, 5, 5, 5, 1]
    cs = [convnormrelu, mbc1, mbc6, mbc6, mbc6, mbc6, mbc6, mbc6, convnormrelu]
    return efficientnet_helper(data, norm, us, fos, fis, ss, ks, cs)


def efficientnet_b6(data, norm, **kwargs):
    # 3.3 GFLOPs
    us = [1, 3, 6, 6, 8, 8, 11, 3, 1]
    fos = [56, 32, 40, 72, 144, 200, 344, 576, 2304]
    fis = [0] + fos[:-1]
    ss = [2, 1, 2, 2, 2, 1, 2, 1, 1]
    ks = [3, 3, 3, 5, 3, 5, 5, 3, 1]
    cs = [convnormrelu, mbc1, mbc6, mbc6, mbc6, mbc6, mbc6, mbc6, convnormrelu]
    return efficientnet_helper(data, norm, us, fos, fis, ss, ks, cs)


def efficientnet_b7(data, norm, **kwargs):
    # 5.1 GFLOPs
    us = [1, 4, 7, 7, 10, 10, 13, 4, 1]
    fos = [64, 32, 48, 80, 160, 224, 384, 640, 2560]
    fis = [0] + fos[:-1]
    ss = [2, 1, 2, 2, 2, 1, 2, 1, 1]
    ks = [3, 3, 3, 5, 3, 5, 5, 3, 1]
    cs = [convnormrelu, mbc1, mbc6, mbc6, mbc6, mbc6, mbc6, mbc6, convnormrelu]
    return efficientnet_helper(data, norm, us, fos, fis, ss, ks, cs)


def efficientnet_fpn_builder(efficientnet):
    class EfficientNetFPN(Backbone):
        def __init__(self, pBackbone):
            super().__init__(pBackbone)
            p = self.p
            data = X.var("data")
            if p.fp16:
                data = data.astype("float16")
            stages = efficientnet(data, p.normalizer, params=p)
            self.symbol = (stages[2], stages[3], stages[5], stages[8])

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol
    return EfficientNetFPN


EfficientNetB4FPN = efficientnet_fpn_builder(efficientnet_b4)
EfficientNetB5FPN = efficientnet_fpn_builder(efficientnet_b5)
EfficientNetB6FPN = efficientnet_fpn_builder(efficientnet_b6)
EfficientNetB7FPN = efficientnet_fpn_builder(efficientnet_b7)


if __name__ == "__main__":
    data = X.var("data")
    norm = X.normalizer_factory()
    *_, last = efficientnet_b4(data, norm)
    mx.viz.print_summary(last, shape={"data": (1, 3, 224, 224)})
