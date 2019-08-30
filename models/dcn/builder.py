import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone


def dcn_resnet_unit(input, name, filter, stride, dilate, proj, norm, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    # conv2 filter router
    conv2_offset = conv(relu1, name=name + "_conv2_offset", filter=72, kernel=3, stride=stride, dilate=dilate)
    conv2 = mx.sym.contrib.DeformableConvolution(relu1, conv2_offset, kernel=(3, 3),
        stride=(stride, stride), dilate=(dilate, dilate), pad=(1, 1), num_filter=filter // 4,
        num_deformable_group=4, no_bias=True, name=name + "_conv2")
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


def hybrid_resnet_stage(data, name, num_block, num_special_block, special_res_unit, filter,
    stride, dilate, norm, **kwargs):
    s, d = stride, dilate

    for i in range(1, num_block + 1 - num_special_block):
        proj = True if i == 1 else False
        s = stride if i == 1 else 1
        d = dilate
        data = resnet_unit(data, "{}_unit{}".format(name, i), filter, s, d, proj, norm)

    for i in range(num_block + 1 - num_special_block, num_block + 1):
        proj = True if i == 1 else False
        s = stride if i == 1 else 1
        d = dilate
        data = special_res_unit(data, "{}_unit{}".format(name, i), filter, s, d, proj, norm, **kwargs)

    return data


def hybrid_resnet_c4_builder(special_resnet_unit):
    class ResNetC4(Backbone):
        def __init__(self, pBackbone):
            super().__init__(pBackbone)
            p = self.p

            import mxnext.backbone.resnet_v1b_helper as helper
            num_c2, num_c3, num_c4, _ = helper.depth_config[p.depth]

            data = X.var("data")
            if p.fp16:
                data = data.astype("float16")
            c1 = helper.resnet_c1(data, p.normalizer)
            c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
            c3 = hybrid_resnet_stage(c2, "stage2", num_c3, p.num_c3_block or 0, special_resnet_unit, 512, 2, 1,
                p.normalizer, params=p)
            c4 = hybrid_resnet_stage(c3, "stage3", num_c4, p.num_c4_block or 0, special_resnet_unit, 1024, 2, 1,
                p.normalizer, params=p)

            self.symbol = c4

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol

    return ResNetC4


def hybrid_resnet_fpn_builder(special_resnet_unit):
    class ResNetFPN(Backbone):
        def __init__(self, pBackbone):
            super().__init__(pBackbone)
            p = self.p

            import mxnext.backbone.resnet_v1b_helper as helper
            num_c2, num_c3, num_c4, num_c5 = helper.depth_config[p.depth]

            data = X.var("data")
            if p.fp16:
                data = data.astype("float16")
            c1 = helper.resnet_c1(data, p.normalizer)
            c2 = hybrid_resnet_stage(c1, "stage1", num_c2, p.num_c2_block or 0, special_resnet_unit, 256, 1, 1,
                p.normalizer, params=p)
            c3 = hybrid_resnet_stage(c2, "stage2", num_c3, p.num_c3_block or 0, special_resnet_unit, 512, 2, 1,
                p.normalizer, params=p)
            c4 = hybrid_resnet_stage(c3, "stage3", num_c4, p.num_c4_block or 0, special_resnet_unit, 1024, 2, 1,
                p.normalizer, params=p)
            c5 = hybrid_resnet_stage(c4, "stage4", num_c5, p.num_c5_block or 0, special_resnet_unit, 2048, 2, 1,
                p.normalizer, params=p)

            self.symbol = (c2, c3, c4, c5)

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol

    return ResNetFPN


DCNResNetC4 = hybrid_resnet_c4_builder(dcn_resnet_unit)
