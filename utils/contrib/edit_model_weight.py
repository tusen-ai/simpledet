# -*- coding: utf-8 -*-
"""
    This script allows you to edit the model weight from https://github.com/TuSimple/simpledet/blob/master/MODEL_ZOO.md
    for your own training. In this script, it assumes you don't change the layer of model, but only edit the content in 
    the layer that models of MODEL_ZOO have. For example, you just change the training classes but not add or delete a 
    unit of resnet.
    
    Example: Edit the weight from 80 + 1 classes to 3 + 1 classes training.
        - Train by your own configuration for one epoch, the configuration should have edited for 3 + 1 classes training.
        - Edit the constant in this file.
            - SIMPLEDET_WEIGHT_FOLDER   the path to the weight folder you download
            - TRAINED_WEIGHT_FOLDER     the path to the weight folder you need the shape
            - EDIT_KEY                  the key of layer which you want to edit the weight, you can show the key by 
                                        print(arg_params_src), in this example, the key names are 
                                        "bbox_cls_logit_weight", "bbox_cls_logit_bias"
        - Run the code!
    
    Note: The new generated model weight file will cover your original downloaded weight file, if you don't want like this,
    you can edit the last line of the code.
    
    TODO: Before you run the code, you should train a model for one epoch and edit the code as instructions above.
"""

import mxnet as mx
import numpy as np
import os

# TODO: Edit the path.
SIMPLEDET_WEIGHT_FOLDER = " "
TRAINED_WEIGHT_FOLDER = " "

# TODO: Edit the key names which you want to modify.
EDIT_KEY = ["bbox_cls_logit_weight", "bbox_cls_logit_bias"]

def change_weight_by_copy_from_right_weight(arg_params_src, arg_params_dst):
    for key in EDIT_KEY:
        arg_params_src[key] = arg_params_dst[key]
    return arg_params_src

if __name__ == "__main__":
    sym, arg_params_src, aux_params = \
        mx.model.load_checkpoint(os.path.join(SIMPLEDET_WEIGHT_FOLDER, "checkpoint"), 6)
    _, arg_params_dst, _ = \
        mx.model.load_checkpoint(os.path.join(TRAINED_WEIGHT_FOLDER, "checkpoint"), 1)

    # print(arg_params_src) to show the key name.
    # arg_params_src means the weight you want to change which is downloaded from simpledet, 
    # arg_params_src means the weight you need the shape.
    arg_params = change_weight_by_copy_from_right_weight(arg_params_src, arg_params_dst)

    mx.model.save_checkpoint(os.path.join(SIMPLEDET_WEIGHT_FOLDER, "checkpoint"), 1, sym, arg_params, aux_params)
