import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add
from mxnext.backbone import resnet_v1_helper, resnet_v1b_helper
from symbol.builder import Backbone


def trident_resnet_v1_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

    ######################### prepare names #########################
    if id is not None:
        conv_postfix = ("_shared%s" if share_conv else "_branch%s") % id
        bn_postfix = ("_shared%s" if share_bn else "_branch%s") % id
        other_postfix = "_branch%s" % id
    else:
        conv_postfix = ""
        bn_postfix = ""
        other_postfix = ""

    ######################### prepare parameters #########################
    conv_params = lambda x: dict(
        weight=X.shared_var(name + "_%s_weight" % x) if share_conv else None,
        name=name + "_%s" % x + conv_postfix
    )

    bn_params = lambda x: dict(
        gamma=X.shared_var(name + "_%s_gamma" % x) if share_bn else None,
        beta=X.shared_var(name + "_%s_beta" % x) if share_bn else None,
        moving_mean=X.shared_var(name + "_%s_moving_mean" % x) if share_bn else None,
        moving_var=X.shared_var(name + "_%s_moving_var" % x) if share_bn else None,
        name=name + "_%s" % x + bn_postfix
    )

    ######################### construct graph #########################
    conv1 = conv(input, filter=filter // 4, stride=stride, **conv_params("conv1"))
    bn1 = norm(conv1, **bn_params("bn1"))
    relu1 = relu(bn1, name=name + other_postfix)

    conv2 = conv(relu1, filter=filter // 4, kernel=3, dilate=dilate, **conv_params("conv2"))
    bn2 = norm(conv2, **bn_params("bn2"))
    relu2 = relu(bn2, name=name + other_postfix)

    conv3 = conv(relu2, filter=filter, **conv_params("conv3"))
    bn3 = norm(conv3, **bn_params("bn3"))

    if proj:
        shortcut = conv(input, filter=filter, stride=stride, **conv_params("sc"))
        shortcut = norm(shortcut, **bn_params("sc_bn"))
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus" + other_postfix)

    return relu(eltwise, name=name + "_relu" + other_postfix)


def trident_resnet_v1b_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    """
    Compared with v1, v1b moves stride=2 to the 3x3 conv instead of the 1x1 conv and use std in pre-processing
    This is also known as the facebook re-implementation of ResNet(a.k.a. the torch ResNet)
    """
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

    ######################### prepare names #########################
    if id is not None:
        conv_postfix = ("_shared%s" if share_conv else "_branch%s") % id
        bn_postfix = ("_shared%s" if share_bn else "_branch%s") % id
        other_postfix = "_branch%s" % id
    else:
        conv_postfix = ""
        bn_postfix = ""
        other_postfix = ""

    ######################### prepare parameters #########################
    conv_params = lambda x: dict(
        weight=X.shared_var(name + "_%s_weight" % x) if share_conv else None,
        name=name + "_%s" % x + conv_postfix
    )

    def bn_params(x):
        ret = dict(
            gamma=X.shared_var(name + "_%s_gamma" % x) if share_bn else None,
            beta=X.shared_var(name + "_%s_beta" % x) if share_bn else None,
            moving_mean=X.shared_var(name + "_%s_moving_mean" % x) if share_bn else None,
            moving_var=X.shared_var(name + "_%s_moving_var" % x) if share_bn else None,
            name=name + "_%s" % x + bn_postfix
        )
        if norm.__name__ == "gn":
            del ret["moving_mean"], ret["moving_var"]
        return ret

    ######################### construct graph #########################
    conv1 = conv(input, filter=filter // 4, **conv_params("conv1"))
    bn1 = norm(conv1, **bn_params("bn1"))
    relu1 = relu(bn1, name=name + other_postfix)

    conv2 = conv(relu1, filter=filter // 4, kernel=3, stride=stride, dilate=dilate, **conv_params("conv2"))
    bn2 = norm(conv2, **bn_params("bn2"))
    relu2 = relu(bn2, name=name + other_postfix)

    conv3 = conv(relu2, filter=filter, **conv_params("conv3"))
    bn3 = norm(conv3, **bn_params("bn3"))

    if proj:
        shortcut = conv(input, filter=filter, stride=stride, **conv_params("sc"))
        shortcut = norm(shortcut, **bn_params("sc_bn"))
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus" + other_postfix)

    return relu(eltwise, name=name + "_relu" + other_postfix)


def get_trident_resnet_backbone(unit, helper):
    class TridentResNetC4(Backbone):
        def __init__(self, pBackbone):
            super().__init__(pBackbone)
            p = self.p

            num_c2, num_c3, num_c4, _ = helper.depth_config[p.depth]
            num_tri = p.num_c4_block or (num_c4 - 1)

            ################### construct symbolic graph ###################
            data = X.var("data")
            if p.fp16:
                data = data.astype("float16")
            c1 = helper.resnet_c1(data, p.normalizer)
            c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
            c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)

            # construct leading res blocks
            data = c3
            for i in range(1, num_c4 - num_tri + 1):
                data = unit(
                    input=data,
                    name="stage3_unit%s" % i,
                    id=None,
                    filter=1024,
                    stride=2 if i == 1 else 1,
                    proj=True if i == 1 else False,
                    dilate=1,
                    params=p)

            # construct parallel branches
            c4s = []
            for dil, id in zip(p.branch_dilates, p.branch_ids):
                c4 = data  # reset c4 to the output of last stage
                for i in range(num_c4 - num_tri + 1, num_c4 + 1):
                    c4 = trident_resnet_v1b_unit(
                        input=c4,
                        name="stage3_unit%s" % i,
                        id=id,
                        filter=1024,
                        stride=2 if i == 1 else 1,
                        proj=True if i == 1 else False,
                        dilate=dil,
                        params=p)
                c4s.append(c4)

            # stack branch outputs on the batch dimension
            c4 = mx.sym.stack(*c4s, axis=1)
            c4 = mx.sym.reshape(c4, shape=(-3, -2))

            self.symbol = c4

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol

    return TridentResNetC4


TridentResNetV1C4 = get_trident_resnet_backbone(trident_resnet_v1_unit, resnet_v1_helper)
TridentResNetV1bC4 = get_trident_resnet_backbone(trident_resnet_v1b_unit, resnet_v1b_helper)
