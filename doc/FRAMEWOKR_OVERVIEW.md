## Introduction

### Scripts


### Top-level Singletons (Detectors)
Top-level singletons generally represent an unique kind of detection method (detector).
By unique we mean that the detector has a pipeline that can not be easily adapted from any existing detector.
Detectors now include
- RPN
- RetinaNet
- YOLOv3 (WIP)
- Fast R-CNN
- Faster R-CNN
- Mask R-CNN
- Cascade R-CNN

Every detector have a `get_train_symbol` method.
Each may have one or more of `get_bbox_test_symbol`, `get_mask_test_symbol`, `get_rpn_test_symbol`, and `get_kp_test_symbol` methods.

Here we leave the API design of `get_xxx_symbol` to user due to
- Detector design should not be bounded by the framework.
- The user is responsible for constructing the `train_sym` used in `detection_train.py` in the config.

We provide a walkthrough of the Mask R-CNN as an example.
```
```