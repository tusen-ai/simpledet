"""
Convert model from Detectron to SimpleDet
@author: Chenxia Han
@usage: Follow following steps to support new model.
    1. Implement functions listed below.
        - get_[part]_regex: regular expression to match parameter name
        - get_[part]_map: mapping relations
        - parse_[part]: parse parameter name into arbitrary parts
    2. Update get_map_str_list to match different types of parameter.
       The front elements have high priority for matching.
    3. Add new choice for model_type of arguments.
    4. Note that this script transposes weight of first convolution from RGB to BGR.
    5. Update get_ignore_name_list to handle some unmatching name
"""

import mxnet as mx
import numpy as np
import argparse
import pickle
import re


def get_res_regex():
    return r'^(res|conv|fc)(b[^n]|[^b])*(w|b)$'


def get_resbn_regex():
    return r'^(res|conv|fc)\w*_bn_(b|s)$'


def get_retnet_regex():
    return r'^(retnet)\w*_(w|b)$'


def get_fpn_regex():
    return r'^(fpn)\w*_(w|b)$'


def get_mask_regex():
    return r'^(mask_fcn_logits|_\[mask\]_fcn|conv5_mask)\w*_(w|b)$'


def get_rpn_head_regex():
    return r'^(conv_rpn|rpn_cls_logits|rpn_bbox_pred)\w*_(w|b)$'


def get_bbox_head_regex():
    return r'^(fc6|fc7|cls_score|bbox_pred)\w*_(w|b)$'


def _get_res_map(is_bn, max_units=50):
    stage_map = {
        'conv1': 'conv0',
        'res': 'bn0',
        'res2': 'stage1',
        'res3': 'stage2',
        'res4': 'stage3',
        'res5': 'stage4',
        'fc1000': 'fc1',
        'pred': 'fc1'
    }

    unit_map = dict([(str(item), 'unit{}'.format(item+1)) for item in range(max_units)])
    unit_map['conv1'] = ''

    branch_map = {
        'branch1': 'sc',
        'branch2a': 'conv1',
        'branch2b': 'conv2',
        'branch2c': 'conv3'
    }

    branch_bn_map = {
        'branch1': 'sc_bn',
        'branch2a': 'bn1',
        'branch2b': 'bn2',
        'branch2c': 'bn3'
    }

    wtype_map = {
        'w': 'weight',
        'b': 'bias'
    }

    wtype_bn_map = {
        'b': 'beta',
        's': 'gamma'
    }

    if is_bn:
        return stage_map, unit_map, branch_bn_map, wtype_bn_map
    else:
        return stage_map, unit_map, branch_map, wtype_map


def get_res_map():
    return _get_res_map(is_bn=False)


def get_resbn_map():
    return _get_res_map(is_bn=True)


def get_retnet_map():
    stage_map = {
        'bbox': 'bbox',
        'cls': 'cls'
    }

    unit_map = {
        'conv_n0': 'conv1',
        'conv_n1': 'conv2',
        'conv_n2': 'conv3',
        'conv_n3': 'conv4',
        'pred': 'pred'
    }

    branch_map = {
        'fpn3': ''
    }

    wtype_map = {
        'w': 'weight',
        'b': 'bias'
    }

    return stage_map, unit_map, branch_map, wtype_map


def get_fpn_map():
    stage_map = {
        'fpn': '',
        'fpn_inner': ''
    }

    unit_map = {
        'res2_2': 'P2',
        'res3_3': 'P3',
        'res4_5': 'P4',
        'res5_2': 'P5',
        'res2_2_sum': 'P2_conv',
        'res3_3_sum': 'P3_conv',
        'res4_5_sum': 'P4_conv',
        'res5_2_sum': 'P5_conv',
        '6': 'P6_conv',
        '7': 'P7_conv'
    }

    branch_map = {
        'sum': 'lateral',
        'sum_lateral': 'lateral',
    }

    wtype_map = {
        'w': 'weight',
        'b': 'bias'
    }

    return stage_map, unit_map, branch_map, wtype_map


def get_mask_map():
    stage_map = {
        'mask': 'mask',
        'conv5': 'mask'
    }

    unit_map = {
        'fcn': 'fcn',
        '[mask]': 'mask',
        'mask': 'up0'
    }

    branch_map = {
        'logits': 'logit',
        'fcn1': 'fcn_conv0',
        'fcn2': 'fcn_conv1',
        'fcn3': 'fcn_conv2',
        'fcn4': 'fcn_conv3'
    }

    wtype_map = {
        'w': 'weight',
        'b': 'bias'
    }

    return stage_map, unit_map, branch_map, wtype_map


def get_rpn_head_map():
    head_map = {
        'conv_rpn_w': 'rpn_conv_weight',
        'conv_rpn_b': 'rpn_conv_bias',
        'conv_rpn_fpn2_w': 'rpn_conv_weight',
        'conv_rpn_fpn2_b': 'rpn_conv_bias',
        'rpn_cls_logits_w': 'rpn_conv_cls_weight',
        'rpn_cls_logits_b': 'rpn_conv_cls_bias',
        'rpn_bbox_pred_w': 'rpn_conv_bbox_weight',
        'rpn_bbox_pred_b': 'rpn_conv_bbox_bias',
        'rpn_cls_logits_fpn2_w': 'rpn_conv_cls_weight',
        'rpn_cls_logits_fpn2_b': 'rpn_conv_cls_bias',
        'rpn_bbox_pred_fpn2_w': 'rpn_conv_bbox_weight',
        'rpn_bbox_pred_fpn2_b': 'rpn_conv_bbox_bias'
    }

    return head_map


def get_bbox_head_map():
    head_map = {
        'fc6_w': 'bbox_fc1_weight',
        'fc6_b': 'bbox_fc1_bias',
        'fc7_w': 'bbox_fc2_weight',
        'fc7_b': 'bbox_fc2_bias',
        'cls_score_w': 'bbox_cls_logit_weight',
        'cls_score_b': 'bbox_cls_logit_bias',
        'bbox_pred_w': 'bbox_reg_delta_weight',
        'bbox_pred_b': 'bbox_reg_delta_bias'
    }

    return head_map


def _parse_res(item, is_bn):
    name_list = item.split('_')
    stage = unit = branch = wtype = ''

    if len(name_list) == 2:
        stage, wtype = name_list
    elif len(name_list) == 4:
        if is_bn:
            stage, unit, _, wtype = name_list
        else:
            stage, unit, branch, wtype = name_list
    elif len(name_list) == 5:
        stage, unit, branch, _, wtype = name_list
    else:
        print(item)
        raise NotImplementedError

    return stage, unit, branch, wtype


def parse_res(item):
    return _parse_res(item, is_bn=False)


def parse_resbn(item):
    return _parse_res(item, is_bn=True)


def parse_retnet(item):
    name_list = item.split('_')
    stage = unit = branch = wtype = ''

    if len(name_list) == 5:
        """
        retnet_bbox_pred_fpn3_b
        => [bbox, pred, fpn3, b]
        => [bbox_pred_bias]
        """
        _, stage, unit, branch, wtype = name_list
    elif len(name_list) == 6:
        """
        retnet_bbox_conv_n1_fpn3_b
        => [bbox, conv_n1, fpn3, b]
        => [bbox_conv2_bias]
        """
        _, stage, unit_a, unit_b, branch, wtype = name_list
        unit = unit_a + '_' + unit_b
    else:
        print(item)
        raise NotImplementedError

    return stage, unit, branch, wtype


def parse_fpn(item):
    name_list = item.split('_')
    stage = unit = branch = wtype = ''

    if item.startswith('fpn_inner'):
        """
        fpn_inner_res3_3_sum_lateral_w
        => [fpn_inner, res3_3, sum_lateral, w]
        => P3_lateral_weight
        fpn_inner_res5_2_sum_w
        => [fpn_inner, res5_2, sum, w]
        => P5_lateral_weight
        """
        if len(name_list) == 6:
            stage_a, stage_b, unit_a, unit_b, branch, wtype = name_list
            stage = stage_a + '_' + stage_b
            unit = unit_a + '_' + unit_b
        elif len(name_list) == 7:
            stage_a, stage_b, unit_a, unit_b, branch_a, branch_b, wtype = name_list
            stage = stage_a + '_' + stage_b
            unit = unit_a + '_' + unit_b
            branch = branch_a + '_' + branch_b
    elif item.startswith('fpn_res'):
        """
        fpn_res3_3_sum_w
        => [fpn, res3_3_sum, w]
        => P3_conv_weight
        """
        assert len(name_list) == 5
        stage, unit_a, unit_b, unit_c, wtype = name_list
        unit = unit_a + '_' + unit_b + '_' + unit_c
    elif item.startswith('fpn_6') or item.startswith('fpn_7'):
        """
        fpn_6_w
        => [fpn, 6, w]
        => P6_conv_weight
        """
        assert len(name_list) == 3
        stage, unit, wtype = name_list
    else:
        print(item)
        raise NotImplementedError

    return stage, unit, branch, wtype


def parse_mask(item):
    name_list = item.split('_')
    stage = unit = branch = wtype = ''

    if len(name_list) == 3:
        stage, unit, wtype = name_list
    elif len(name_list) == 4:
        stage, unit, branch, wtype = name_list
    else:
        print(item)
        raise NotImplementedError

    return stage, unit, branch, wtype


def parse_rpn_head(item):
    return item


def parse_bbox_head(item):
    return item


def get_map_str_list(model_type):
    # candidates for map: [res, resbn, fpn, retnet, rpn_head, bbox_head, mask]
    if model_type == 'resnet':
        map_str_list = ['res', 'resbn']
    elif model_type == 'resnext':
        map_str_list = ['res', 'resbn']
    elif model_type == 'retina':
        map_str_list = ['res', 'resbn', 'fpn', 'retnet']
    elif model_type == 'fpn':
        # bbox_head and rpn_head should be put before res to own higher priority
        map_str_list = ['bbox_head', 'rpn_head', 'res', 'resbn', 'fpn']
    elif model_type == 'mask':
        # mask is the same as above
        map_str_list = ['bbox_head', 'rpn_head', 'mask', 'res', 'resbn', 'fpn']
    else:
        print(model_type)
        raise NotImplementedError

    return map_str_list


def get_ignore_name_list():
    ignore_name_list = [
        'pred_w', 'pred_b',  # resnext-101-32x8d
        'lr', 'weight_order'  # resnext-101-64x4d
    ]

    return ignore_name_list


def main(args):
    data = pickle.load(open(args.detectron_model, 'rb'), encoding='latin1')
    if 'blobs' in data:
        data = data['blobs']

    # get list of ignored names
    ignore_name_list = get_ignore_name_list()

    # get map_str_list
    map_str_list = get_map_str_list(args.model_type)

    # create map_regex_dict
    map_regex_dict = dict()
    for map_str in map_str_list:
        map_regex = eval('get_%s_regex' % map_str)()
        map_regex_dict[map_str] = re.compile(map_regex)

    # regex for batch norm
    bn_regex = re.compile(r'^\w+(_bn_(b|s))$')

    # init parameters
    arg_params = dict()
    aux_params = dict()

    for item in data:
        # ignore momentum
        if item.endswith('momentum'):
            print('ignore %s' % item)
            continue

        # get each splitted part and map
        regex_matched = False
        for map_str in map_str_list:
            map_regex = map_regex_dict[map_str]
            if map_regex.search(item):
                regex_matched = True
                part_list = eval('parse_%s' % map_str)(item)
                part_map_list = eval('get_%s_map' % map_str)()
                if not isinstance(part_list, list) and not isinstance(part_list, tuple):
                    part_list = [part_list]
                if not isinstance(part_map_list, list) and not isinstance(part_list, tuple):
                    part_map_list = [part_map_list]
                assert len(part_list) == len(part_map_list), \
                    'Numebr of parts should be equals to its maps. (%d vs. %d)' % (len(part_list), len(part_map_list))
                break

        # raise error if not match to any regex
        if not regex_matched:
            if item in ignore_name_list:
                continue
            print(item)
            raise NotImplementedError

        # handle absence of part
        for part_map in part_map_list:
            part_map[''] = ''

        # map each part
        mapped_part_list = [part_map[part] for part, part_map in zip(part_list, part_map_list)]

        # construct mapped name
        mapped_part_filtered = [part for part in mapped_part_list if part != '']
        mapped_name = '_'.join(mapped_part_filtered)

        print('converting %s => %s' % (item, mapped_name))

        # transpose first convolution weight, from RGB to BGR
        if mapped_name == 'conv0_weight':
            print('\033[94m' + ('transpose %s weight from RGB to BGR' % mapped_name) + '\033[0m')
            arg_params[mapped_name] = mx.nd.array(data[item][:, [2, 1, 0], :, :], dtype=np.float32)
        else:
            arg_params[mapped_name] = mx.nd.array(data[item], dtype=np.float32)

        # set default value for moving_mean and moving_var
        # since which are merged into conv in Detectron
        if bn_regex.search(item):
            bn_shape = data[item].shape
            if 'gamma' in mapped_name:
                moving_var_name = mapped_name.replace('gamma', 'moving_var')
                aux_params[moving_var_name] = mx.nd.ones(bn_shape)
                print('\033[94m' + ('init %s to one' % moving_var_name) + '\033[0m')
            elif 'beta' in mapped_name:
                moving_mean_name = mapped_name.replace('beta', 'moving_mean')
                aux_params[moving_mean_name] = mx.nd.zeros(bn_shape)
                print('\033[94m' + ('init %s to zero' % moving_mean_name) + '\033[0m')

    mx.model.save_checkpoint(args.simpledet_model_prefix, 0, None, arg_params, aux_params)
    print('\033[92m' + 'Successful convert to simpledet model, saved to %s-%04d.params.' % (args.simpledet_model_prefix, 0) + '\033[0m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Model from Detectron')
    parser.add_argument('--detectron_model',            help='path to detectron model',         required=True)
    parser.add_argument('--simpledet_model_prefix',     help='path to simpledet model prefix',  required=True)
    parser.add_argument('--model_type',                 help='model type to convert',           required=True,
                        choices=['resnet', 'resnext', 'retina', 'fpn', 'mask'])
    args = parser.parse_args()

    main(args)
