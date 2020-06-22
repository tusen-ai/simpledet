## Scale-Equalizing Pyramid Convolution for Object Detection
This repository implements [Scale-Equalizing Pyramid Convolution for Object Detection](https://arxiv.org/abs/2005.03101) in the SimpleDet framework.

### Qucik Start
 
```python
# train
python detection_train.py --config config/sepc/retina_r50v1b_fpn_sepc_1x.py

# test
python detection_test.py --config config/sepc/retina_r50v1b_fpn_sepc_1x.py
```

### Performance and Model
All AP results are reported on COCO val2017:

Model | Backbone | Train Schedule | GPU | Image/GPU| Train MEM| Train Speed|	FP16| Box AP | link |
---------- | --------- | --------- | ---------- | ---------| ----------| ----------| ---------| -----------| -----------
retinanet (baseline) | res50v1b | 1x | 8X 2080Ti |4|8653M | 44 img/s| yes|35.9 | [model](https://1drv.ms/u/s!AhNcLYzCx6CCjGJfW59R3IEelhxv?e=Ob9y4W)|
retinanet_pconv | res50v1b | 1x | 8X 2080Ti |4| 9111M|43 img/s | yes|37.2 | [model](https://1drv.ms/u/s!AhNcLYzCx6CCjGPiw3cfqOWZkUAB?e=PIHppA)|
retinanet_pconv+ibn | res50v1b | 1x | 8X 2080Ti|4 |9467M | 40 img/s| yes|37.6 | [model](https://1drv.ms/u/s!AhNcLYzCx6CCjGayQr_1Ew-dhRfA?e=W2AXi6)|
retinanet_sepclite | res50v1b | 1x | 8X 2080Ti |4| 9467M|36 img/s |yes|38.6 |[model](https://1drv.ms/u/s!AhNcLYzCx6CCjGTAbLT7_YXjq3GF?e=ZHfIqn) |
retinanet_sepc | res50v1b | 1x | 8X 2080Ti |4| 9471M|25 img/s | yes|**39.7** | [model](https://1drv.ms/u/s!AhNcLYzCx6CCjGWO02qo_adoy8km?e=30H1sl)|
