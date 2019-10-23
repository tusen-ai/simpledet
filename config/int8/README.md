## quantization training

#### motivation
to speedup the inference application. For example, there is the time of model `faster_r50v1c4_c5_512roi_1x` in mxnet and tensorrt with `dtype=fp32` and `dtype=int8`.


| platform | dtype | time(ms) |
| -------- | ----- | -------- |
| mxnet    | fp32  | 337      |
| tensorrt | fp32  | 260      |
| tensorrt | int8  | 100      |

**detail configs**

```shell
batch size=1
device = GPU 1080
data shape = (1, 3, 800, 1200)
```



### the principle of our implementation

#### the quantization methods

**Weight:**  there is no learnable parameters. We quantize weight directly as below.

```shell
nbits = 8
QUANT_LEVEL = 2**(nbits -1) -1
threshold = max(abs(w_tensor))
quant_unit = threshold / QUANT_LEVEL
quantized_w = round(w_tensor / quant_unit) * quant_unit
```

**activation:** We learn it's threshold named `minmax` with EMA update method. [ref](<https://arxiv.org/pdf/1712.05877.pdf>)

```shell
nbits = 8
QUANT_LEVEL = 2**(nbits -1) -1
history_threshold;  # initialized by max(abs(act_tensor))
curr_max = max(abs(act_tensor))
threshold = 0.99 * history_threshold + 0.01 * curr_max
quant_unit = threshold / QUANT_LEVEL
quantized_act = round(w_tensor / quant_unit) * quant_unit
```

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


#### drawback

Tensorrt don't support quantizing Operator with custom setting. such as only quantize `Conv` or `fc`. And there is no API to setting `quantize scale` by user's own `scale` instead of   `scale`  calcuated by tensorrt.  So the  learned `threshold` can't deploy to tensorrt currently. 