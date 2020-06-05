## PAFPN

This repository implements [**PAFPN**](https://arxiv.org/abs/1803.01534) in the SimpleDet framework.

### Quick Start
```bash
# train
python detection_train.py --config config/pafpn/faster_r50v1b_pafpn3@256_syncbn_1x.py
# test
python detection_test.py --config config/pafpn/faster_r50v1b_pafpn3@256_syncbn_1x.py
```

### Results

| Detector | Pyramid | AP | AP50 | AP75 | APs | APm | APl |
|----------|---------|----|------|------|-----|-----|-----|
| Faster R50v1b | PAFPN 3@256 | 38.6 | 58.8 | 41.8 | 22.3 | 42.6 | 50.8 |