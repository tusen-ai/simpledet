## NAS-FPN

This repository implements [**NAS-FPN**](https://arxiv.org/abs/1904.07392) in the SimpleDet framework.

### Qucik Start
```bash
# train baseline retinanet following the setting of NAS-FPN
python3 detection_train.py --config config/NASFPN/retina_r50v1b_fpn_640_1@256_25epoch.py

# train NAS-FPN
python3 detection_train.py --config config/NASFPN/retina_r50v1b_nasfpn_640_7@256_25epoch.py
python3 detection_train.py --config config/NASFPN/retina_r50v1b_nasfpn_1024_7@256_25epoch.py
python3 detection_train.py --config config/NASFPN/retina_r50v1b_nasfpn_1280_7@384_25epoch.py

# train hand-crafted neck
python3 detection_train.py --config config/NASFPN/retina_r50v1b_tdbu_1280_3@384_25epoch.py
```

### Results and Models
All AP results are reported on test-dev of the [COCO dataset](http://cocodataset.org).

|Model|InputSize|Backbone|Neck|Train Schedule|GPU|Image/GPU|FP16|Train MEM|Train Speed|Box AP(Mask AP)|Link|
|-----|-----|--------|----|--------------|---|---------|----|---------|-----------|---------------|----|
|RetinaNet|640|R50v1b-FPN|1@256|25 epoch|8X 1080Ti|8|yes|6.6G|85 img/s|37.4|[model](https://1dv.alarge.space/retina_r50v1b_fpn_640640_25epoch.zip)|
|NAS-FPN|640|R50v1b-FPN|7@256|25 epoch|8X 1080Ti|8|yes|7.8G|66 img/s|40.1|[model](https://1dv.alarge.space/retina_r50v1b_nasfpn_640640_25epoch.zip)|
|NAS-FPN|1024|R50v1b-FPN|7@256|25 epoch|8X 1080Ti|4|yes|9.1G|17 img/s|44.2|[model](https://1dv.alarge.space/retina_r50v1b_nasfpn_1024_7%40256_25epoch.zip)|
|NAS-FPN|1280|R50v1b-FPN|7@384|25 epoch|8X 1080Ti|2|yes|8.9G|10 img/s|45.3|[model](https://1dv.alarge.space/retina_r50v1b_nasfpn_1280_7%40384_25epoch.zip)|
|TD-BU*|1280|R50v1b-FPN|3@384|25 epoch|8X 1080Ti|3|yes|10.5G|12 img/s|44.7|[model](https://1dv.alarge.space/retina_r50v1b_tdbu_1280_3%40384_25epoch.zip)|

\* Short for TopDown-BottomUp neck which is highly symmetric proposed by Zehao.
### Reference
```
@inproceedings{ghiasi2019fpn,
  title={NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection},
  author={Ghiasi, Golnaz and Lin, Tsung-Yi and Pang, Ruoming and Le, Quoc V},
  booktitle={CVPR},
  year={2019}
}
```
