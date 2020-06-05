## Feature Pyramid Grids

This repository implements [**FPG**](https://arxiv.org/pdf/2004.03580.pdf) in the SimpleDet framework.

### Quick Start
```bash
# train
python detection_train.py --config config/FPG/faster_r50v1b_fpg6@128_syncbn_1x.py
# test
python detection_test.py --config config/FPG/faster_r50v1b_fpg6@128_syncbn_1x.py
```

### Results

| Detector | Pyramid | AP | AP50 | AP75 | APs | APm | APl |
|----------|---------|----|------|------|-----|-----|-----|
| Faster R50v1b | FPG 6@128 | 38.7 | 59.5 | 42.3 | 23.7 | 42.3 | 48.3|