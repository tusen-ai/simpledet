## RepPoints

This repository implements [**RepPoints**](https://arxiv.org/abs/1904.11490) in the SimpleDet framework.
RPDet is a state-of-the-art anchor-free detector, utilizing a point set as the representation of objects in localization and recognition.

### Qucik Start
```bash
# train
python3 detection_train.py --config config/reppoints_moment_r50v1_fpn_1x.py

# test
python3 detection_test.py --config config/reppoints_moment_r50v1_fpn_1x.py
```

### Models
All AP results are reported on minival2014 of the [COCO dataset](http://cocodataset.org).

|Method|Backbone|Transform|Schedule|AP (paper)|AP (re-impl)|Link|
|------|--------|---------|--------|----------|------------|----|
|RepPoints|R50v1-FPN|MinMax|1x|38.2|38.0|[model](https://drive.google.com/open?id=1BNF7cLJDLgOUpSgQ3bcXm2iSHop5G3Rp)|
|RepPoints|R50v1-FPN|Moment|1x|38.3|38.3|[model](https://drive.google.com/open?id=1q0mFJl0qG22Y6AlRQ95HSIFT0GKuQRLS)|
|RepPoints|R101v1-FPN|Moment|2x|40.3|40.7|[model](https://drive.google.com/open?id=1dslqEcvlPh-8NoRhU--7ypan7XnAP_S5)|
|RepPoints|R101v1b-FPN-DCNv1|Moment|2x, multi-scale training & testing|-|46.4|[model](https://drive.google.com/open?id=1SreAuNE7ILXcBx8_-NHyftZTgS94kzO6)|
|RepPoints|R101v1b-FPN-DCNv2|Moment|2x, multi-scale training & testing|-|47.0|[model](https://drive.google.com/open?id=14GFKGeXU9FVBFDQUS-4jlH2raSLzt8Zd)|

### Reference
```
@inproceedings{yang2019reppoints,
  title={RepPoints: Point Set Representation for Object Detection},
  author={Yang, Ze and Liu, Shaohui and Hu, Han and Wang, Liwei and Lin, Stephen},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019}
}
```
