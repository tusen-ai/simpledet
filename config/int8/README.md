## Quantization during Training

#### Motivation
Low precision weight and activation could greatly reduce the storage and memory footprint of detection models and improve the inference latency. We provide the inference time measured on TensorRT of INT8 and FP32 version of `faster_r50v1c4_c5_512roi_1x` as an example below.

| dtype | time(ms) | minival mAP|
| ----- | -------- | -----------|
| fp32  | 260      | 35.7       |
| int8  | 100      | 35.8       |

**detail configs**

```shell
batch size=1
device = GTX 1080
data shape = (1, 3, 800, 1200)
```

### Implementation Details

#### the Quantization Methods

**for model weight:**
```shell
nbits = 8
QUANT_LEVEL = 2 ** (nbits - 1) - 1
threshold = max(abs(w_tensor))
quant_unit = threshold / QUANT_LEVEL
quantized_w = round(w_tensor / quant_unit) * quant_unit
```

**for model activation:** The threshold is maintained with exponetial moving average of max absolute activation. [ref](<https://arxiv.org/pdf/1712.05877.pdf>)

```shell
nbits = 8
QUANT_LEVEL = 2**(nbits -1) -1
history_threshold;  # initialized by max(abs(act_tensor))
curr_max = max(abs(act_tensor))
threshold = 0.99 * history_threshold + 0.01 * curr_max
quant_unit = threshold / QUANT_LEVEL
quantized_act = round(w_tensor / quant_unit) * quant_unit
```

### Quick Start for int8 Training

```shell
python detection_train.py --config config/faster_r50v1c4_c5_512roi_1x.py
```

### Quantization Configs
The quantization configs are in the `ModelParam.QuantizeTrainingParam` class, which give users more flexibility during quantization.

**quantize_flag:**  to quantize the model or not.

**quantized_op:** the operators to quantize.

`WeightQuantizeParam` and `ActQuantizeParam` is attributes need by `Quantization_int8` operator for quantizing `weight` and `activation`.

### Attributes of the `quantization_int8` operator

**delya_quant:** after delay_quant iters, the quantization working actually.

**ema_decay:**  the hyperparameter for activation threshold update.

**grad_mode:**  the mode for gradients pass. there are two mode: ste or clip. ste mean straightforward pass the out gradients to data, clip mean only pass the gradients whose value of data in the range of [-threshold, threshold], the gradients of outer is settting to 0.

**workspace:**  the temporary space used in grad_mode='clip'

**is_weight:** the tensor to be quantized is weight or not.

**is_weight_perchannel:** the granularity of quantization for weight : per tensor or per channel. Only used when the tensor is weight. Currently,  only support pertensor mode.

**quant_mode:**  the quantization methods: `minmax` or `power2`,  Currently, only support minmax mode.


#### how to reproduce the result
1. To train a fp32 model with the default config.
2. setting quantization config. finetuning the trained fp32 model. Our finetuning epoch setting are:  `begin_epoch=6` and `end_epoch=12`.  all other configs are the same as fp32 training configs. 


#### drawback
Tensorrt don't support quantizing Operator with custom setting. such as only quantize `conv` or `fc`. And there is no API to setting `quantize scale` by user's own `scale` instead of   `scale`  calcuated by tensorrt.  So the  learned `threshold` can't deploy to tensorrt currently. 
