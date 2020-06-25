## Crowdhuman Dataset

This repository implements Faster-RCNN and [**Double Pred**](https://arxiv.org/abs/2003.09163) on [**CrowdHuman**](https://arxiv.org/abs/1805.00123) dataset in the SimpleDet framework.

### Quick Start

#### 1. Prepare Crowdhuman Format Dataset
```bash
# Ensure that the directory of crowdhuman dataset looks like:
# data/crowdhuman
#       ---------/images/xxxx.jpg
#       ---------/annotations/xxxx.ogdt
python utils/create_crowdhuman_roidb.py --dataset crowdhuman --dataset-split train --num-threads 45
```

#### 2. Train Model
```bash
# train
python detection_train.py --config config/crowdhuman/faster_r50v1b_fpn_1x.py
python detection_train.py --config config/doublepred_r50v1b_fpn_1x.py
python detection_train.py --config config/doublepred_r50v1b_fpn_1x_refine.py

# test
python detection_test.py --config config/crowdhuman/faster_r50v1b_fpn_1x.py
python detection_test.py --config config/doublepred_r50v1b_fpn_1x.py
python detection_test.py --config config/doublepred_r50v1b_fpn_1x_refine.py
```

### Results on CrowdHuman

| Detector | AP | MR |
|----------|---------|----|
| Faster R50v1b | 84.77 | 46.72 |
| DoublePred R50v1b | 88.64 | 45.52 |
| DoublePred R50v1b + Refine | 88.81 | 45.02 |

Note that crowdhuman is different from COCO-like dataset, since it contains **ignore region**. We followed the procedure shared by Zheng Ge([Talk Link](https://www.bilibili.com/video/av455989666/)) by ignoring anchors in RPN and adding BN in FPN. A simple Toolkit to evaluate AP and MR with ignore region can refer to [here](https://github.com/Purkialo/CrowdDet/tree/master/lib/evaluate).