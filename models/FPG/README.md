## Feature Pyramid Grids & PAFPN

This repository implements [**FPG**](https://arxiv.org/pdf/2004.03580.pdf) and [**PAFPN**](https://arxiv.org/abs/1803.01534) in the SimpleDet framework.

### Quick Start
```bash
# train
python detection_train.py --config config/FPG/faster_r50v1b_fpg6@128_syncbn_1x.py
python detection_train.py --config config/pafpn/faster_r50v1b_pafpn3@256_syncbn_1x.py
python detection_train.py --config config/pafpn/faster_r50v1b_pafpn3@384_syncbn_1x.py
# test
python detection_test.py --config config/FPG/faster_r50v1b_fpg6@128_syncbn_1x.py
python detection_test.py --config config/pafpn/faster_r50v1b_pafpn3@256_syncbn_1x.py
python detection_test.py --config config/pafpn/faster_r50v1b_pafpn3@384_syncbn_1x.py
```

### Results

| Detector | Pyramid | AP | AP50 | AP75 | APs | APm | APl |
|----------|---------|----|------|------|-----|-----|-----|
| Faster R50v1b | FPG 6@128 | 38.7 | 59.5 | 42.3 | 23.7 | 42.3 | 48.3|
| Faster R50v1b | PAFPN 3@256 | 38.6 | 58.8 | 41.8 | 22.3 | 42.6 | 50.8 |
| Faster R50v1b | PAFPN 3@384 | 39.4 | 59.9 | 42.8 | 23.9 | 43.2 | 50.9 |

Note that SyncBN is only used in FPG neck but used in all BN layers under PAFPN settings according to the original papers. Besides, TDBU Neck in NASFPN folder is a special case of PAFPN with 3 stages and 384 channels, thus this setting is also appended in this config.