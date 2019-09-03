## Introduction
This document describes the process of funetuning an existing model for your custom dataset. Here we take a model trained on the COCO dataset and finetune it on the PASCAL VOC dataset.

### Preparing Data
SimpleDet requires groundtruth annotation organized as following format
```python
# example.json
[
    {
        "gt_class": (nBox, ),
        "gt_bbox": (nBox, 4),  # in xyxy format
        "flipped": bool,
        "h": int,
        "w": int,
        "image_url": str,
        "im_id": int
    },
    ...
]
```

The default directory structure of VOC looks like
```bash
$ tree -d
VOCdevkit
├── VOC2007
│   ├── Annotations
│   ├── ImageSets
│   │   ├── Layout
│   │   ├── Main
│   │   └── Segmentation
│   ├── JPEGImages
│   ├── SegmentationClass
│   └── SegmentationObject
└── VOC2012
    ├── Annotations
    ├── ImageSets
    │   ├── Action
    │   ├── Layout
    │   ├── Main
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject
```

The original annotation provided by VOC looks like
```xml
<annotation>
        <folder>VOC2007</folder>
        <filename>000001.jpg</filename>
        <source>
                <database>The VOC2007 Database</database>
                <annotation>PASCAL VOC2007</annotation>
                <image>flickr</image>
                <flickrid>341012865</flickrid>
        </source>
        <owner>
                <flickrid>Fried Camels</flickrid>
                <name>Jinky the Fruit Bat</name>
        </owner>
        <size>
                <width>353</width>
                <height>500</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>dog</name>
                <pose>Left</pose>
                <truncated>1</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>48</xmin>
                        <ymin>240</ymin>
                        <xmax>195</xmax>
                        <ymax>371</ymax>
                </bndbox>
        </object>
        <object>
                <name>person</name>
                <pose>Left</pose>
                <truncated>1</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>8</xmin>
                        <ymin>12</ymin>
                        <xmax>352</xmax>
                        <ymax>498</ymax>
                </bndbox>
        </object>
</annotation>
```

The `gt_class` and `gt_bbox` can be read from `<object>`. `gt_class` start with 1, as 0 is reserved for the background class.
`filp` is always set to `False` as `detection_train.py` will do the flip for you.
`h` and `w` can be read from `<size>`.
`image_url` can be constrcuted from `<folder>` and `<filename>`.
`im_id` is set as a monotonic incresing sequence starting from 0. Note that the value of `im_id` shall not exceed 16M due to float32 overflow. `im_id` is used to uniquely identify images during test only. **Since combining several subset for testing is quite rare, so one may reuse the `im_id` across subsets.**

```json
# example.json
[
    {
        "gt_class": [1, 5],
        "gt_bbox": [[48, 240, 195, 371], [8, 12, 352, 498]],
        "flipped": false,
        "h": 500,
        "w": 353,
        "image_url": "/absolute/path/to/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
        "im_id": 1
    },
    ...
]
```
Refer to `utils/create_voc_roidb.py` for more details.

After prepared the json file, you can use `utils/json_to_roidb.py` to convert it into a valid roidb file for SimpleDet
```bash
python3 utils/json_to_roidb.py --json path/to/your.json
```


### Tune from COCO pretrain
We first train a Faster R-CNN FPN R-50 detector as the baseline.
```
python detection_train.py --config config/faster_r50v1_fpn_voc07_1x.py
python detection_test.py --config config/faster_r50v1_fpn_voc07_1x.py
```
This gives a mAP@50 of 76.3

We then use the Mask R-CNN R-50 FPN pretrained on COCO for initialization.
1. Download and rename the weight
```
mv checkpoint-0006.params pretrain_model/r50v1-maskrcnn-coco-0000.params
```
2. Remove the class-aware logit parameters.
```python
import mxnet as mx
params = mx.nd.load("r50v1-maskrcnn-coco-0000.params")
del params["arg:bbox_cls_logit_weight"]
del params["arg:bbox_cls_logit_bias"]
del params["arg:bbox_reg_delta_weight"]
del params["arg:bbox_reg_delta_bias"]
mx.nd.save("r50v1-maskrcnn-coco-0000.params", params)
```

3. Train the FPN R50 from MaskRCNN initialization. This gives a mAP@50 of 82.5, a 6.2 mAP gain compared with the ImageNet pretrain.
```
python detection_train.py --config config/faster_r50v1_fpn_voc07_finetune_1x.py
python detection_test.py --config config/faster_r50v1_fpn_voc07_finetune_1x.py
```

From the diff we can see that the pretrain model is changed, the init lr is divided by 10 without stepping, no warmup is empolyed, and the training epochs is halved.

```bash
$ diff config/finetune/faster_r50v1_fpn_voc07_1x.py config/finetune/faster_r50v1_fpn_voc07_finetune_1x.py
145c145
<             prefix = "pretrain_model/resnet-v1-50"
---
>             prefix = "pretrain_model/r50v1-maskrcnn-coco"
159c159
<             lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
---
>             lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 10
166,172c166,167
<             end_epoch = 6
<             lr_iter = [10000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]
<
<         class warmup:
<             type = "gradual"
<             lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3.0
<             iter = 100
---
>             end_epoch = 3
>             lr_iter = []

```