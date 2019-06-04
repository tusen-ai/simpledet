## NAS-FPN

This repository implements [**NAS-FPN**](https://arxiv.org/abs/1904.07392) in the SimpleDet framework.

### Qucik Start
Set workspace=1400 in [RetinaNet Builder](https://github.com/TuSimple/simpledet/blob/master/models/retinanet/builder.py#L295).

```bash
# train baseline retinanet following the setting of NAS-FPN
python3 detection_train.py --config config/NASFPN/retina_r50v1b_fpn_640640_25epoch.py

# train NAS-FPN
python3 detection_train.py --config config/NASFPN/retina_r50v1b_nasfpn_640640_25epoch.py
```

### Results and Models
All AP results are reported on test-dev of the [COCO dataset](http://cocodataset.org).

|Model|Backbone|Head|Train Schedule|GPU|Image/GPU|FP16|Train MEM|Train Speed|Box AP(Mask AP)|Link|
|-----|--------|----|--------------|---|---------|----|---------|-----------|---------------|----|
|RetinaNet|R50v1b-FPN|4Conv|25 epoch|8X 1080Ti|8|yes|6.6G|85 img/s|37.4|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/retina_r50v1b_fpn_640640_25epoch.zip)|
|NAS-FPN|R50v1b-FPN|4Conv|25 epoch|8X 1080Ti|8|yes|7.8G|66 img/s|40.1|[model](https://simpledet-model.oss-cn-beijing.aliyuncs.com/retina_r50v1b_nasfpn_640640_25epoch.zip)|

### Reference
```
@inproceedings{ghiasi2019fpn,
  title={NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection},
  author={Ghiasi, Golnaz and Lin, Tsung-Yi and Pang, Ruoming and Le, Quoc V},
  booktitle={CVPR},
  year={2019}
}
```
