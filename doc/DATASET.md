## Introduction
This document describes the process of creating roidb from COCO-format, VOC-format or JSON-format annotations.

### COCO format
In this section, we create roidb from coco-format annotaions of PASCAL VOC dataset. 

```bash
# enter simpledet main directory
cd simpledet

# create data dir
mkdir -p data/src
pushd data/src

# download and extract VOC2007 trainval
wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar 
tar xf data/src/VOCtrainval_06-Nov-2007.tar

# download and extract VOC annotaitons provided by COCO
wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip
unzip PASCAL_VOC.zip
popd

# create soft links
mkdir -p data/pascal_voc/annotations
ln -s data/src/PASCAL_VOC/pascal_train2007.json data/pascal_voc/annotations/instances_train2007.json
ln -s data/src/PASCAL_VOC/pascal_val2007.json data/pascal_voc/annotations/instances_val2007.json

mkdir -p data/pascal_voc/images
ln -s data/src/VOCdevkit/VOC2007/JPEGImages data/pascal_voc/images/train2007
ln -s data/src/VOCdevkit/VOC2007/JPEGImages data/pascal_voc/images/val2007

# annotations/instances_split.json should correspond with images/split
pascal_voc
├── annotations
│   ├── instances_train2007.json -> data/src/PASCAL_VOC/pascal_train2007.json
│   └── instances_val2007.json -> data/src/PASCAL_VOC/pascal_val2007.json
└── images
    ├── train2007 -> data/src/VOCdevkit/VOC2007/JPEGImages
    └── val2007 -> data/src/VOCdevkit/VOC2007/JPEGImages

# generate roidbs
python3 utils/create_coco_roidb.py --dataset pascal_voc --dataset-split train2007
python3 utils/create_coco_roidb.py --dataset pascal_voc --dataset-split val2007
```


### VOC format
In this section, we create roidb from voc-format annotaions of clipart dataset. 
```bash
# enter simpledet main directory
cd simpledet

# create data dir
mkdir -p data/src
pushd data/src

# download and extract clipart.zip
# courtesy to "Towards Universal Object Detection by Domain Attention"
wget https://1dv.alarge.space/clipart.zip -O clipart.zip
unzip clipart.zip
popd

# generate roidbs
python3 utils/create_voc_roidb.py --data-dir data/src/clipart --split train
```

### JSON format
In this section, we create roidb from json-format annotaions of clipart dataset. 

Prepare your own data like the example
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

```bash
python3 utils/json_to_roidb.py --json path/to/your.json
```

### Existing Annotations
- Cityscapes (coco format)
Check [this](https://github.com/facebookresearch/Detectron/blob/master/tools/convert_cityscapes_to_coco.py) script
- COCO (coco format)
http://cocodataset.org/#download
- DeepLesion (voc format)
Check [here](https://drive.google.com/drive/folders/1Uwnhg0qZ5k-3VZ7uSDt3WyETcnSdeA3M)
- DOTA (voc format)
Check [here](https://drive.google.com/drive/folders/1Uwnhg0qZ5k-3VZ7uSDt3WyETcnSdeA3M)
- Kitchen (voc format)
Check [here](https://drive.google.com/drive/folders/1Uwnhg0qZ5k-3VZ7uSDt3WyETcnSdeA3M)
- KITTI (voc format)
Check [here](https://drive.google.com/drive/folders/1Uwnhg0qZ5k-3VZ7uSDt3WyETcnSdeA3M)
- VOC (voc format)
Check [here](https://drive.google.com/drive/folders/1Uwnhg0qZ5k-3VZ7uSDt3WyETcnSdeA3M)
- WiderFace (voc format)
Check [here](https://drive.google.com/drive/folders/1Uwnhg0qZ5k-3VZ7uSDt3WyETcnSdeA3M)
