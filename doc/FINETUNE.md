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
[
    {
        "gt_class": [1, 5],
        "gt_bbox": [[48, 240, 195, 371], [8, 12, 352, 498]],
        "flipped": False,
        "h": 500,
        "w": 353,
        "image_url": "/absolute/path/to/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
        "im_id": 1
    },
    ...
]
```

Refer to `utils/create_voc_roidb.py` for more details.

###