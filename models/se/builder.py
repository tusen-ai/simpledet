import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add, sigmoid
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone
from models.efficientnet.builder import se
from models.dcn.builder import hybrid_resnet_fpn_builder
from models.maskrcnn.builder import MaskFasterRcnnHead


def se_resnet_v1b_unit(input, name, filter, stride, dilate, proj, norm, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    conv2 = conv(relu1, name=name + "_conv2", stride=stride, filter=filter // 4, kernel=3)
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(conv3, name=name + "_bn3")
    bn3 = se(bn3, prefix=name + "_se3", f_down=filter // 16, f_up=filter)


    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


def se_v2_resnet_v1b_unit(input, name, filter, stride, dilate, proj, norm, **kwargs):
    """
    diff with v1: move the SE module to 3x3 conv
    """
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    conv2 = conv(relu1, name=name + "_conv2", stride=stride, filter=filter // 4, kernel=3)
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")
    relu2 = se(relu2, prefix=name + "_se2", f_down=filter // 16, f_up=filter // 4)

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


SEResNetV1bFPN = hybrid_resnet_fpn_builder(se_resnet_v1b_unit)
SEv2ResNetV1bFPN = hybrid_resnet_fpn_builder(se_v2_resnet_v1b_unit)


class MaskRcnnSe4convHead(MaskFasterRcnnHead):
    def __init__(self, pBbox, pMask, pMaskRoi):
        super().__init__(pBbox, pMask, pMaskRoi)

    def _get_mask_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        up_stride = int(self.pMask.resolution // self.pMaskRoi.out_size)
        dim_reduced = self.pMask.dim_reduced

        msra_init = mx.init.Xavier(rnd_type="gaussian", factor_type="out", magnitude=2)

        current = conv_feat
        for i in range(4):
            current = X.conv(
                current,
                name="mask_fcn_conv{}".format(i + 1),
                filter=dim_reduced,
                kernel=3,
                no_bias=False,
                init=msra_init
            )
            current = self.add_norm(current)
            current = X.relu(current)
            current = se(current, "mask_fcn_se{}".format(i + 1), f_down=dim_reduced // 4, f_up=dim_reduced)

        mask_up = current
        for i in range(up_stride // 2):
            weight = X.var(
                name="mask_up{}_weight".format(i),
                init=msra_init,
                lr_mult=1,
                wd_mult=1)
            mask_up = mx.sym.Deconvolution(
                mask_up,
                kernel=(2, 2),
                stride=(2, 2),
                num_filter=dim_reduced,
                no_bias=False,
                weight=weight,
                name="mask_up{}".format(i)
                )
            mask_up = X.relu(
                mask_up,
                name="mask_up{}_relu".format(i))

        mask_up = X.to_fp32(mask_up, name='mask_up_to_fp32')
        self._head_feat = mask_up

        return self._head_feat