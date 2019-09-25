## EfficientNet for object detection
This repository implements [**EfficientNet**](https://arxiv.org/abs/1905.11946) in the SimpleDet framework. Efficient B5 achives the same mAP with **~1/10 FLOPs** compared with ResNet-50.

### Qucik Start
```bash
# train faster r-cnn with efficientnet fpn backbone
python3 detection_train.py --config config/efficientnet/efficientnet_b5_fpn_bn_scratch_400_6x.py
```

### Results and Models
All AP results are reported on minival of the [COCO dataset](http://cocodataset.org).

|Model|InputSize|Backbone|Train Schedule|GPU|Image/GPU|FP16|Train MEM|Train Speed|Box AP|Link|
|-----|-----|--------|--------------|---|---------|----|---------|-----------|---------------|----|
|Faster|400x600|B5-FPN|36 epoch(6X)|8X 1080Ti|8|yes|-|75 img/s|37.2|[model](https://1dv.alarge.space/efficientnet_b5_fpn_bn_scratch_400_6x.zip)|
|Faster|400x600|B5-FPN|54 epoch(9X)|8X 1080Ti|8|yes|-|75 img/s|37.9|-|
|Faster|400x600|B5-FPN|72 epoch(12X)|8X 1080Ti|8|yes|-|75 img/s|38.3|-|

### Reference
```
@inproceedings{tan2019,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  year={2019}
}
```
