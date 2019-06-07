# SimpleDet Model Zoo

## Introduction

This file documents a large collection of baselines trained with SimpleDet.

#### Common Settings
- All models were trained on ```train2014+valminusminival2014```, and tested on ```minival2014```.
- We adopt the same training schedules as Detectron. 1x indicates 6 epochs and 2x indicates 12 epochs since we append flipped images into training data.
- We report the training GPU memory as what ```nvidia-smi``` shows.

#### ImageNet Pretrained Models

We provide the ImageNet pretrained models used by SimpleDet. Unless otherwise noted, these models are trained on the standard ImageNet-1k dataset.

- [resnet-v1-50](https://simpledet-model.oss-cn-beijing.aliyuncs.com/resnet-v1-50-0000.params): converted copy of MARA's original ResNet-50 model.
- [resnet-v1-101](https://simpledet-model.oss-cn-beijing.aliyuncs.com/resnet-v1-101-0000.params): converted copy of MARA's original ResNet-101 model.
- [resnet-50](https://simpledet-model.oss-cn-beijing.aliyuncs.com/resnet-50-0000.params): ResNet-v2-50 model provided by MXNet Model Gallery.
- [resnet-101](https://simpledet-model.oss-cn-beijing.aliyuncs.com/resnet-101-0000.params): ResNet-v2-101 model provided by MXNet Model Gallery.
- [resnet50_v1b](https://simpledet-model.oss-cn-beijing.aliyuncs.com/resnet50_v1b-0000.params):
converted from gluoncv.
- [resnet101_v1b](https://simpledet-model.oss-cn-beijing.aliyuncs.com/resnet101_v1b-0000.params):
converted from gluoncv.
- [resnet152_v1b](https://simpledet-model.oss-cn-beijing.aliyuncs.com/resnet152_v1b-0000.params):
converted from gluoncv.

## ResNetV1b Baselines
All config files can be found in config/resnet_v1b.
Pretrains are converted from GluonCV. 
All AP results are reported on minival2014 of the [COCO dataset](http://cocodataset.org).

|Model|Backbone|Head|Train Schedule|AP|AP50|AP75|APs|APm|APl|
|-----|--------|----|--------------|--|----|----|---|---|---|
|Faster|R50v1b-C4|C5-512ROI|1X|35.7|56.7|37.9|18.6|40.4|48.1|
|Faster|R50v1b-C4|C5-512ROI|2X|36.9|57.9|39.3|19.9|41.4|50.2|
|Faster|R101v1b-C4|C5-512ROI|1X|40.0|61.3|43.1|21.5|44.8|54.3|
|Faster|R101v1b-C4|C5-512ROI|2X|40.5|61.2|43.8|22.5|44.8|55.4|
|Faster|R152v1b-C4|C5-512ROI|1X|41.3|62.6|44.6|23.4|46.2|55.6|
|Faster|R152v1b-C4|C5-512ROI|2X|41.8|62.4|45.2|23.2|46.0|56.9|
|Faster|R50v1b-FPN|2MLP|1X|37.2|59.4|40.4|22.3|41.3|47.6|
|Faster|R50v1b-FPN|2MLP|2X|38.0|59.7|41.5|22.2|41.6|48.8|
|Faster|R101v1b-FPN|2MLP|1X|39.9|62.1|43.5|23.1|44.4|51.1|
|Faster|R101v1b-FPN|2MLP|2X|40.4|62.1|44.0|23.2|44.4|52.7|
|Faster|R152v1b-FPN|2MLP|1X|41.5|63.5|45.7|24.7|46.0|53.3|
|Faster|R152v1b-FPN|2MLP|2X|42.0|63.6|45.9|24.8|45.9|55.0|

## Box, and Mask Detection Baselines
All AP results are reported on minival2014 of the [COCO dataset](http://cocodataset.org).

|Model|Backbone|Head|Train Schedule|GPU|Image/GPU|FP16|Train MEM|Train Speed|Box AP(Mask AP)|Link|
|-----|--------|----|--------------|---|---------|----|---------|-----------|---------------|----|
|Faster|R50v1-C4|C5-512ROI|1X|8X 1080Ti|2|no|8.4G|20 img/s|34.2|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v1c4_c5_512roi_1x.zip)|
|Faster|R50v1-C4|C5-512ROI|1X|8X TitanV|2|yes|6.1G|49 img/s|34.4|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v1c4_c5_512roi_1x_fp16.zip)|
|Faster|R50v2-C4|C5-256ROI|1X|8X 1080Ti|2|no|5.1G|33 img/s|32.8|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v2c4_c5_256roi_1x.zip)|
|Cascade|R50v2-C5|2MLP|1X|8X 1080Ti|2|no|5.3G|27 img/s|37.5|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/cascade_r50v2_c5_red_1x.zip)|
|Faster|R50v1-FPN|2MLP|1X|8X 1080Ti|2|no|5.2G|36 img/s|36.5|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v1_fpn_1x.zip)|
|Mask|R50v1-FPN|2MLP+4CONV|1X|8X 1080Ti|2|no|6.7G|19 img/s|37.1(33.7)|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/mask_r50v1_fpn_1x.zip)|
|Retina|R50v1-FPN|4Conv|1X|8X 1080Ti|2|no|5.1G|44 img/s|35.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/retina_r50v1_fpn_1x.zip)|
|Trident|R50v2-C4|C5-128ROI|1X|8X 1080Ti|2|no|7.2G|19 img/s|36.4|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r50v2c4_c5_1x.zip)|
|Faster|R101v2-C4|C5-256ROI|1X|8X 1080Ti|2|no|6.7G|25 img/s|37.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v2c4_c5_256roi_1x.zip)|
|Faster-SyncBN|R101v2-C4|C5-256ROI|1X|8X 1080Ti|2|no|7.8G|17 img/s|38.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v2c4_c5_256roi_syncbn_1x.zip)|
|Faster|R101v1-C4|C5-512ROI|1X|8X 1080Ti|2|no|10.2G|16 img/s|38.3|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v1c4_c5_512roi_1x.zip)|
|Faster|R101v1-C4|C5-512ROI|1X|8X TitanV|2|yes|7.0G|35 img/s|38.1|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v1c4_c5_512roi_1x_fp16.zip)|
|Faster|R101v1-FPN|2MLP|1X|8X 1080Ti|2|no|7.5G|24 img/s|38.7|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v1_fpn_1x.zip)|
|Cascade|R101v2-C5|2MLP|1X|8X 1080Ti|2|no|7.1G|23 img/s|40.0|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/cascade_r101v2_c5_red_1x.zip)|
|Trident|R101v2-C4|C5-128ROI|1X|8X 1080Ti|1|no|6.6G|9 img/s|40.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_1x.zip)|
|Trident-Fast|R101v2-C4|C5-128ROI|1X|8X 1080Ti|1|no|6.6G|9 img/s|39.9|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_fastapprox_1x.zip)|
|Retina|R101v1-FPN|4Conv|1X|8X 1080Ti|2|no|7.1G|31 img/s|37.8|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/retina_r101v1_fpn_1x.zip)|

## FP16 Speed Benchmark
Here we provide the FP16 speeed benchmark results of several models.

|Model|Backbone|Head|Train Schedule|GPU|Image/GPU|FP16|Train MEM|Train Speed|
|-----|--------|----|--------------|---|---------|----|---------|-----------|
|Faster|R50v1-C4|C5-512ROI|1X|8X 1080Ti|2|no|8.4G|20 img/s|
|Faster|R50v1-C4|C5-512ROI|1X|8X TitanV|2|yes|6.1G|49 img/s|
|Faster|R50v1-C4|C5-512ROI|1X|8X TitanV|4|yes|11.2G|55 img/s|
|Faster|R50v2-C4|C5-256ROI|1X|8X 1080Ti|2|no|5.1G|33 img/s|
|Faster|R50v2-C4|C5-256ROI|1X|8X TitanV|2|yes|3.8G|61 img/s|
|Faster|R50v2-C4|C5-256ROI|1X|8X TitanV|4|yes|6.6G|73 img/s|
|Faster|R101v1-C4|C5-512ROI|1X|8X 1080Ti|2|no|10.2G|16 img/s|
|Faster|R101v1-C4|C5-512ROI|1X|8X TitanV|2|yes|7.0G|35 img/s|
