import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone


def dr_resnet_unit(input, name, filter, stride, dilate, proj, norm, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    # conv2 filter banks
    conv2_weight = X.var(name + "_conv2_weight")
    dil1, dil2, dil3 = kwargs.get("dilates", (1, 2, 3))
    conv2_1 = conv(relu1, name=name + "_conv2_1", weight=conv2_weight,
        filter=filter // 4, kernel=3, stride=stride, dilate=dil1)
    conv2_2 = conv(relu1, name=name + "_conv2_2", weight=conv2_weight,
        filter=filter // 4, kernel=3, stride=stride, dilate=dil2)
    conv2_3 = conv(relu1, name=name + "_conv2_3", weight=conv2_weight,
        filter=filter // 4, kernel=3, stride=stride, dilate=dil3)

    # construct
    if kwargs.get("baseline", False):
        conv2 = mx.sym.add_n(conv2_1, conv2_2, conv2_3)
    else:
        conv2_bank = mx.sym.stack(conv2_1, conv2_2, conv2_3, axis=1)  # n x 3 x c x h x w
        conv2_router = conv(relu1, name=name + "_conv2_router", filter=3, kernel=3, stride=stride, dilate=1)
        conv2_router = X.softmax(conv2_router, name=name + "_conv2_router_softmax")  # n x 3 x h x w
        conv2_router = mx.sym.expand_dims(conv2_router, axis=2)  # n x 3 x 1 x h x w
        conv2 = mx.sym.broadcast_mul(conv2_bank, conv2_router).sum(axis=1)  # n x c x h x w

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

def dr_resnet_stage(data, name, num_block, filter, stride, dilate, norm, **kwargs):
    s, d = stride, dilate

    data = resnet_unit(data, "{}_unit1".format(name), filter, s, d, True, norm)
    for i in range(2, num_block + 1 - 3):
        data = resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm)
    for i in range(num_block + 1 - 3, num_block + 1):
        data = dr_resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm, **kwargs)

    return data


class DRResNetC4(Backbone):
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
        c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
        c4 = dr_resnet_stage(c3, "stage3", num_c4, 1024, 2, 1, p.normalizer,
            dilates=p.dilates, baseline=p.baseline)

        self.symbol = c4

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


def dcn_resnet_unit(input, name, filter, stride, dilate, proj, norm):
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

def dcn_resnet_stage(data, name, num_block, filter, stride, dilate, norm):
    s, d = stride, dilate

    data = resnet_unit(data, "{}_unit1".format(name), filter, s, d, True, norm)
    for i in range(2, num_block + 1 - 3):
        data = resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm)
    for i in range(num_block + 1 - 3, num_block + 1):
        data = dcn_resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm)

    return data


class DCNResNetC4(Backbone):
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
        c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
        c4 = dcn_resnet_stage(c3, "stage3", num_c4, 1024, 2, 1, p.normalizer)

        self.symbol = c4

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol
