# TSD
This repository implements the CVPR 2020 paper [\<\<Revisiting the Sibling Head in Object Detector\>\>](https://openaccess.thecvf.com/content_CVPR_2020/papers/Song_Revisiting_the_Sibling_Head_in_Object_Detector_CVPR_2020_paper.pdf).

## Quick Start
```
# train
python detection_train.py --config config/TSD/tsd_r50_rpn_1x.py

# test
python detection_test.py --config config/TSD/tsd_r50_rpn_1x.py
```

## COCO minival Performance

All results are reported using ResNet-50 and 1x schedule training.

TSD: Task-aware Spatial Disentanglement 

PC: Progressive Constraint 

|Method|AP|AP_50|AP_75|AP_s|AP_m|AP_l|
|------|--|-----|-----|----|----|----|
|Baseline Faster RCNN|36.3|58.2|39.0|21.3|39.8|46.9|
|+TSD|39.3|60.6|42.8|22.2|42.8|52.0|	
|+TSD and PC|38.9|60.2|42.2|22.0|42.4|51.6|

## Citation
```
@InProceedings{Song_2020_CVPR,
author = {Song, Guanglu and Liu, Yu and Wang, Xiaogang},
title = {Revisiting the Sibling Head in Object Detector},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```