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
|Retina|R50v1-FPN|4Conv|1X|36.6|56.9|39.0|20.3|40.7|47.2|
|Retina|R101v1-FPN|4Conv|1X|39.2|59.5|42.2|22.8|44.0|51.1|
|Retina|R152v1-FPN|4Conv|1X|40.4|61.1|43.4|23.6|45.0|52.3|


## Box, and Mask Detection Baselines
All AP results are reported on minival2014 of the [COCO dataset](http://cocodataset.org).

|Model|Backbone|Head|Train Schedule|GPU|Image/GPU|FP16|Train MEM|Train Speed|Box AP(Mask AP)|Link|
|-----|--------|----|--------------|---|---------|----|---------|-----------|---------------|----|
|Faster|R50v1-C4|C5-512ROI|1X|8X 1080Ti|2|no|5.9G(4.5G)|20 img/s|34.2|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v1c4_c5_512roi_1x.zip)|
|Faster|R50v1-C4|C5-512ROI|1X|8X TitanV|2|yes|6.1G|49 img/s|34.4|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v1c4_c5_512roi_1x_fp16.zip)|
|Faster|R50v2-C4|C5-256ROI|1X|8X 1080Ti|2|no|5.1G|33 img/s|32.8|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v2c4_c5_256roi_1x.zip)|
|Cascade|R50v2-C5|2MLP|1X|8X 1080Ti|2|no|5.9G|25 img/s|38.8|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/cascade_r50v2_c5_red_1x.zip)|
|Cascade|R50v1-FPN|2MLP|1X|8X 1080Ti|2|no|6.6G|21 img/s|40.3|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/cascade_r50v1_fpn_1x.zip)|
|Faster|R50v1-FPN|2MLP|1X|8X 1080Ti|2|no|4.2G(2.6G)|43 img/s|36.5|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r50v1_fpn_1x.zip)|
|Mask|R50v1-FPN|2MLP+4CONV|1X|8X 1080Ti|2|no|5.7G(3.6G)|35 img/s|37.1(33.7)|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/mask_r50v1_fpn_1x.zip)|
|Retina|R50v1-FPN|4Conv|1X|8X 1080Ti|2|no|4.7G(2.2G)|44 img/s|35.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/retina_r50v1_fpn_1x.zip)|
|Trident|R50v2-C4|C5-128ROI|1X|8X 1080Ti|2|no|7.0G(5.3G)|20 img/s|37.1|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r50v2c4_c5_1x.zip)|
|Faster|R101v2-C4|C5-256ROI|1X|8X 1080Ti|2|no|6.7G|25 img/s|37.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v2c4_c5_256roi_1x.zip)|
|Faster-SyncBN|R101v2-C4|C5-256ROI|1X|8X 1080Ti|2|no|7.8G|17 img/s|38.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v2c4_c5_256roi_syncbn_1x.zip)|
|Faster|R101v1-C4|C5-512ROI|1X|8X 1080Ti|2|no|10.2G|16 img/s|38.3|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v1c4_c5_512roi_1x.zip)|
|Faster|R101v1-C4|C5-512ROI|1X|8X TitanV|2|yes|7.0G|35 img/s|38.1|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v1c4_c5_512roi_1x_fp16.zip)|
|Faster|R101v1-FPN|2MLP|1X|8X 1080Ti|2|no|5.3G(3.4G)|31 img/s|38.7|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/faster_r101v1_fpn_1x.zip)|
|Cascade|R101v2-C5|2MLP|1X|8X 1080Ti|2|no|7.6G|22 img/s|41.0|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/cascade_r101v2_c5_red_1x.zip)|
|Cascade|R101v1-FPN|2MLP|1X|8X 1080Ti|2|no|8.7G|19 img/s|42.3|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/cascade_r101v1_fpn_1x.zip)|
|Trident|R101v2-C4|C5-128ROI|1X|8X 1080Ti|1|no|6.6G|9 img/s|40.6|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_1x.zip)|
|Trident-Fast|R101v2-C4|C5-128ROI|1X|8X 1080Ti|1|no|6.6G|9 img/s|39.9|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/tridentnet_r101v2c4_c5_fastapprox_1x.zip)|
|Retina|R101v1-FPN|4Conv|1X|8X 1080Ti|2|no|5.9G(3.0G)|31 img/s|37.8|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/retina_r101v1_fpn_1x.zip)|

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
|Faster|R50v1-FPN|2MLP|1X|8X 1080Ti|2|no|4.2G(2.6G)|43 img/s|
|Faster|R50v1-FPN|2MLP|1X|8X 2080Ti|2|yes|3.7G(3.1G)|65 img/s|
|Faster|R50v1-FPN|2MLP|1X|8X 2080Ti|4|yes|6.2G(6.4G)|77 img/s|
