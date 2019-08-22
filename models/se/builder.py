import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add, sigmoid
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone
from models.efficientnet.builder import se
from models.dcn.builder import hybrid_resnet_fpn_builder


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


SEResNetV1bFPN = hybrid_resnet_fpn_builder(se_resnet_v1b_unit)
