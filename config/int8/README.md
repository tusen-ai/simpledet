## quantization training

#### how to run int8 training

```shell
python detection_train.py --config config/faster_r50v1c4_c5_512roi_1x.py
```

#### quantization config

the quantization config is in the `ModelParam.QuantizeTrainingParam` class. 

**quantize_flag:**  To quantize the model or not.

**quantized_op:** the operators to quantize.

`WeightQuantizeParam` and `ActQuantizeParam` is attributes need by `Quantization_int8` operator for quantizing `weight` and `activation`.

#### details of quantization_int8' attributes

**delya_quant:** after delay_quant iters, the quantization working actually.

**ema_decay:**  the hyperparameter for activation threshold update.

**grad_mode:**  the mode for gradients pass. there are two mode: ste or clip. ste mean straightforward pass the out gradients to data, clip mean only pass the gradients whose value of data in the range of [-threshold, threshold], the gradients of outer is settting to 0.

**workspace:**  the temporary space used in grad_mode='clip'

**is_weight:** the tensor to be quantized is weight or not.

**is_weight_perchannel:** the granularity of quantization for weight : per tensor or per channel. Only used when the tensor is weight. Currently,  only support pertensor mode.

**quant_mode:**  the quantization methods: `minmax` or `power2`,  Currently, only support minmax mode.

### result

**dataset:** coco_2017

| model                        | fp32  | int8 training(quantize conv and fc) |
| ---------------------------- | ----- | ----------------------------------- |
| faster_r50v1bc4_c5_512roi_1x | 0.357 | 0.358                               |

#### how to Recurring result

1. To train a fp32 model with the default config.
2. setting quantization config. finetuning the trained fp32 model. Our finetuning epoch setting are:  `begin_epoch=6` and `end_epoch=12`.  all other configs are the same as fp32 training configs. 