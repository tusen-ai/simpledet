## Scale-Aware Trident Networks for Object Detection

Yanghao Li\*, Yuntao Chen\*, Naiyan Wang, Zhaoxiang Zhang

<p align="center"> <img src="../../doc/image/trident_block.png" width="480"> </p>

### Introduction

This repository implements [TridentNet](https://arxiv.org/abs/1901.01892) in the SimpleDet framework. 

Trident Network (TridentNet) aims to generate scale-specific feature maps with a uniform representational power. We construct a parallel multi-branch architecture in which each branch shares the same transformation parameters but with different receptive fields. Then, we propose a scale-aware training scheme to specialize each branch by sampling object instances of proper scales for training. As a bonus, a fast approximation version of TridentNet could achieve significant improvements without any additional parameters and computational cost. On the COCO dataset, our TridentNet with ResNet-101 backbone achieves state-of-the-art single-model results by obtaining an mAP of 48.4.

#### Trident Blocks

- Dilated convolution for efficient scale enumeration
- Weight sharing between convs for uniform representation

<p align="center"> <img src="../../doc/image/trident_block_details.png" width="480"> </p>

The above figure shows how to convert bottleneck residual blocks to 3-branch Trident Blocks. The dilation rate of three branches are set as 1, 2 and 3, respectively.

### Use TridentNet

Please setup SimpleDet following [README](../../README.md) and [INSTALL](../../doc/INSTALL.md) and use the TridentNet configuration files in the `config` folder.

### Results on MS-COCO

|                             | Backbone   | Test data | mAP@[0.5:0.95] | Link |
| --------------------------- | ---------- | --------- | :------------: | -----|
| Faster R-CNN, 1x            | ResNet-101 | minival   |      37.6      |[model](https://1dv.alarge.space/faster_r101v2c4_c5_256roi_1x.zip)|
| TridentNet, 1x              | ResNet-101 | minival   |      40.6      |[model](https://1dv.alarge.space/tridentnet_r101v2c4_c5_1x.zip)|
| TridentNet, 1x, Fast Approx | ResNet-101 | minival   |      39.9      |[model](https://1dv.alarge.space/tridentnet_r101v2c4_c5_fastapprox_1x.zip)|
| TridentNet, 2x              | ResNet-101 | test-dev  |      42.8      |[model](https://1dv.alarge.space/tridentnet_r101v2c4_c5_addminival_2x.zip)|
| TridentNet*, 3x             | ResNet-101 | test-dev  |      48.4      |[model](https://1dv.alarge.space/tridentnet_r101v2c4_c5_multiscale_addminival_3x_fp16.zip)|

Note: 
1. These models are not trained in SimpleDet. Re-training these models in SimpleDet gives a slightly better result.
2. TridentNet* - TridentNet = extended training + softNMS + multi-scale training/testing + syncBN + DCNv1.

### Results on MS-COCO with stronger baselines
All config files are available in [config/resnet_v1b](../../config/resnet_v1b).

|Model|Backbone|Head|Train Schedule|AP|AP50|AP75|APs|APm|APl|
|-----|--------|----|--------------|--|----|----|---|---|---|
|Faster|R50v1b-C4|C5-512ROI|2X|36.9|57.9|39.3|19.9|41.4|50.2|
|Trident|R50v1b-C4|C5-128ROI|2X|39.6|60.9|42.9|22.5|44.5|53.9|
|TridentFast|R50v1b-C4|C5-128ROI|2X|39.0|60.2|41.8|20.8|43.6|53.8|
|Faster|R101v1b-C4|C5-512ROI|2X|40.5|61.2|43.8|22.5|44.8|55.4|
|Trident|R101v1b-C4|C5-128ROI|2X|43.0|64.3|46.3|25.3|47.9|58.4|
|TridentFast|R101v1b-C4|C5-128ROI|2X|42.5|63.7|46.0|23.3|46.7|59.3|
|Faster|R152v1b-C4|C5-512ROI|2X|41.8|62.4|45.2|23.2|46.0|56.9|
|Trident|R152v1b-C4|C5-128ROI|2X|44.4|65.4|48.3|26.4|49.4|59.6|
|TridentFast|R152v1b-C4|C5-128ROI|2X|43.9|65.1|47.0|25.1|48.1|60.4|

### Citing TridentNet

```
@article{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={ICCV 2019},
  year={2019}
}
```
