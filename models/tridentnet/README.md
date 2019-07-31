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

Please setup SimpleDet following [README](../../README.md)  and use the TridentNet configuration files in the `config` folder.

### Results on MS-COCO

|                             | Backbone   | Test data | mAP@[0.5:0.95] | Link |
| --------------------------- | ---------- | --------- | :------------: | -----|
| Faster R-CNN, 1x            | ResNet-101 | minival   |      37.6      |[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v2c4_c5_256roi_1x.zip)|
| TridentNet, 1x              | ResNet-101 | minival   |      40.6      |[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_1x.zip)|
| TridentNet, 1x, Fast Approx | ResNet-101 | minival   |      39.9      |[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_fastapprox_1x.zip)|
| TridentNet, 2x              | ResNet-101 | test-dev  |      42.8      |[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_addminival_2x.zip)|
| TridentNet*, 3x             | ResNet-101 | test-dev  |      48.4      |[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_multiscale_addminival_3x_fp16.zip)|

Note: 
1. These models are not trained in SimpleDet. Re-training these models in SimpleDet gives a slightly better result.

### Results on MS-COCO with stronger baselines
All config files are available in [config/resnet_v1b](../../config/resnet_v1b).

|Model|Backbone|Head|Train Schedule|AP|AP50|AP75|APs|APm|APl|
|-----|--------|----|--------------|--|----|----|---|---|---|
|Faster|R50v1b-C4|C5-512ROI|1X|35.7|56.7|37.9|18.6|40.4|48.1|
|Faster|R50v1b-C4|C5-512ROI|2X|36.9|57.9|39.3|19.9|41.4|50.2|
|Faster|R101v1b-C4|C5-512ROI|1X|40.0|61.3|43.1|21.5|44.8|54.3|
|Faster|R101v1b-C4|C5-512ROI|2X|40.5|61.2|43.8|22.5|44.8|55.4|
|Faster|R152v1b-C4|C5-512ROI|1X|41.3|62.6|44.6|23.4|46.2|55.6|
|Faster|R152v1b-C4|C5-512ROI|2X|41.8|62.4|45.2|23.2|46.0|56.9|
|Trident|R50v1b-C4|C5-128ROI|1X|38.4|59.7|41.5|21.4|43.6|52.4|
|Trident|R50v1b-C4|C5-128ROI|2X|39.6|60.9|42.9|22.5|44.5|53.9|
|Trident|R101v1b-C4|C5-128ROI|1X|42.2|63.6|45.3|24.5|47.2|57.7|
|Trident|R101v1b-C4|C5-128ROI|2X|43.0|64.3|46.3|25.3|47.9|58.4|
|Trident|R152v1b-C4|C5-128ROI|1X|43.7|64.1|48.0|26.9|47.9|58.9|
|Trident|R152v1b-C4|C5-128ROI|2X|44.4|65.4|48.3|26.4|49.4|59.6|
|TridentFast|R50v1b-C4|C5-128ROI|1X|37.7|58.7|40.3|19.5|42.4|52.7|
|TridentFast|R50v1b-C4|C5-128ROI|2X|39.0|60.2|41.8|20.8|43.6|53.8|
|TridentFast|R101v1b-C4|C5-128ROI|1X|41.1|62.5|43.9|22.1|45.7|57.7|
|TridentFast|R101v1b-C4|C5-128ROI|2X|42.5|63.7|46.0|23.3|46.7|59.3|
|TridentFast|R152v1b-C4|C5-128ROI|1X|42.7|64.0|45.6|23.4|47.5|59.1|
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
