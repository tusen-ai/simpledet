## Mask Scoring RCNN

This repository implements [**Mask Scoring RCNN**](https://arxiv.org/abs/1903.00241) in the SimpleDet framework.

### Qucik Start
```bash
# train
python3 detection_train.py --config config/ms_r50v1_fpn_1x.py

# test
python3 ms_test.py --config config/ms_r50v1_fpn_1x.py
```