## FreeAnchor

This repository implements [**FreeAnchor**](https://arxiv.org/abs/1909.02466) in the SimpleDet framework.
FreeAnchor assigns anchors for ground-truth objects with a maximum likelihood estimation procedure. On the basis of RetinaNet, this method achieves a significant improvement on performance.

### Qucik Start
```bash
# train
python3 detection_train.py --config config/FreeAnchor/free_anchor_r50v1_fpn_1x.py
# test
python3 detection_test.py --config config/FreeAnchor/free_anchor_r50v1_fpn_1x.py
```

### Models
All AP results are reported on minival2014 of the [COCO dataset](http://cocodataset.org).

|Method|Backbone|Schedule|AP|Link|
|------|--------|--------|--|----|
|FreeAnchor|R50v1-FPN|1x|38.3|[model](https://drive.google.com/open?id=1k043sSZa-sa6qeHuDG21OFOrze1SF364)|
|FreeAnchor|R101v1-FPN|1x|40.4|[model](https://drive.google.com/open?id=1Rki-hZFsuMHleYJpoXFJQMplCFxkDfW-)|

### Reference
```
@inproceedings{zhang2019freeanchor,
  title={{FreeAnchor}: Learning to Match Anchors for Visual Object Detection},
  author={Zhang, Xiaosong and Wan, Fang and Liu, Chang and Ji, Rongrong and Ye, Qixiang},
  booktitle={Neural Information Processing Systems},
  year={2019}
}
```
