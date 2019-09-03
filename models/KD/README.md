## KD

This repository implements [**Knowledge Distillation**](https://arxiv.org/abs/1503.02531) in the SimpleDet framework.

### Qucik Start
```bash
python3 detection_train.py --config config/kd/retina_r50v1b_fpn_1x_fitnet_g10.py
python3 detection_test.py --config config/kd/retina_r50v1b_fpn_1x_fitnet_g10.py
```

### Results and Models
All AP results are reported on the minival2014 split of the [COCO](http://cocodataset.org) dataset.

|Model|Backbone|Head|Train Schedule|AP|AP50|AP75|APs|APm|APl|
|-----|--------|----|--------------|--|----|----|---|---|---|
|Retina|R50v1b-FPN|4Conv|1X|36.6|56.9|39.0|20.3|40.7|47.2|
|Retina|R50v1b-FPN-TR152v1b1X|4Conv|1X|38.9|59.0|41.6|21.4|43.3|52.1|
|Retina|R50v1b-FPN-TR152v1b1X|4Conv|2X|40.1|60.6|43.1|21.8|44.5|54.3|
|Faster|R50v1b-FPN|2MLP|1X|37.2|59.4|40.4|22.3|41.3|47.6|
|Faster|R50v1b-FPN|2MLP|2X|38.0|59.7|41.5|22.2|41.6|48.8|
|Faster|R50v1b-FPN-TR152v1b2X|2MLP|1X|39.9|61.3|43.6|22.7|44.2|52.7|
|Faster|R50v1b-FPN-TR152v1b2X|2MLP|2X|40.5|62.2|43.9|23.1|44.7|53.9|
