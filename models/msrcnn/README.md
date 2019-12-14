## Mask Scoring RCNN

This repository implements [**Mask Scoring RCNN**](https://arxiv.org/abs/1903.00241) in the SimpleDet framework.

### Set Up
You need newer [mxnet-cu100-20191214](https://1dv.alarge.space/mxnet_cu100-1.6.0b20191214-py2.py3-none-manylinux1_x86_64.whl) or [mxnet-cu101-20191214](https://1dv.alarge.space/mxnet_cu101-1.6.0b20191214-py2.py3-none-manylinux1_x86_64.whl)

### Qucik Start
```bash
# train
python3 detection_train.py --config config/ms_r50v1_fpn_1x.py

# test
python3 ms_test.py --config config/ms_r50v1_fpn_1x.py
```

### Performance
|Model|Backbone|Head|Train Schedule|GPU|FP16|Train MEM|Train Speed|Image/GPU|Box AP(Mask AP)|Link|
|-----|--------|----|--------------|---|---------|----|---------|-----------|---------------|----|
|Mask Scoring|R50v1-FPN|2MLP+4CONV|1X|8X 1080Ti|2|no|8.1G(3.6G)|23 img/s|37.2(35.0)|[model](https://1dv.alarge.space/ms_r50v1_fpn_1x.zip)
