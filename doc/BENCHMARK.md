### Platform
SimpleDet features high efficiency on relatively cheap hardwares. P1 is a 10k USD platform and P2 can be build as a 20k USD platform with lower end CPUs. The hardwares used in the benchmark of `mmdetection` and `maskrcnn-benchmark` should cost around 100k USD and have far superior GPU inter-connection than P2.

P1:
- 2X E5-2682 v4
- 220G MEM
- 8X 1080Ti
- Single root PCI-E topology
- Ubuntu 16.04
- driver 390.48 + CUDA 9.0 + cuDNN 7.5.0 + NCCL 2.4.2
- pip environment

P2:
- 2X Platinum 8163
- 377G MEM
- 8X 2080Ti
- Dual root PCI-E topology(slower GPU inter-connection than E5V4)
- Ubuntu 16.04
- driver 418.39 + CUDA 10.0 + cuDNN 7.5.0 + NCCL 2.4.2
- conda environment

---

### Frameworks
simpledet:
`simpledet@2d8144` + `mxnext@896aa1` + `mxnet@24cce9e`

maskrcnn-benchmark:
`maskrcnn-benchmark@24c8c9` + `pytorch1.1` + `torchvision0.3.0`

mmdetection:
`mmdetection@cda5fd` + `pytorch1.1` + `torchvision0.3.0`

---

### Launching commands for benchmarking
#### simpledet:
P1:
```bash
# nnvm_rpn_target=Fasle, fp16=Fasle
python detection_train.py --config config/faster_r50v1_fpn_accel_1x.py

# nnvm_rpn_targe=True, fp16=False
python detection_train.py --config config/mask_r50v1_fpn_1x.py

# fp16=False
python detection_train.py --config config/retina_r50v1_fpn_1x.py
```

P2:
```bash
# fp16=False/True
python detection_train.py --config config/retina_r50v1_fpn_1x.py

# nnvm_rpn_target=True, fp16=Fasle/True
python detection_train.py --config config/faster_r50v1_fpn_accel_1x.py

# nnvm_rpn_targe=True, fp16=False/True
python detection_train.py --config config/mask_r50v1_fpn_1x.py
```

#### maskrcnn-benchmark
P1:
```bash
# IMS_PER_BATCH=16
export NGPUS=8; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/retinanet/retinanet_R-50-FPN_1x.yaml

export NGPUS=8; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_FPN_1x.yaml MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000

export NGPUS=8; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
```

P2:
```bash
# IMS_PER_BATCH=16
export NGPUS=8; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/retinanet/retinanet_R-50-FPN_1x.yaml

# IMS_PER_BATCH=16
export NGPUS=8; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/retinanet/retinanet_R-50-FPN_1x.yaml DTYPE "float16"

export NGPUS=8; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_FPN_1x.yaml MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000

export NGPUS=8; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_R_50_FPN_1x.yaml MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 DTYPE "float16"

# limit OMP_NUM_THREADS for the conda env, else each process takes 2000% CPU and causes severe contending
export NGPUS=8; export OMP_NUM_THREADS=1; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000

export NGPUS=8; export OMP_NUM_THREADS=1; python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 DTYPE "float16"
```

#### mmdetection
P1:
```bash
./tools/dist_train.sh configs/retinanet_r50_fpn_1x.py 8

./tools/dist_train.sh configs/faster_rcnn_r50_fpn_1x.py 8

./tools/dist_train.sh configs/mask_rcnn_r50_fpn_1x.py 8
```

P2:
```bash
./tools/dist_train.sh configs/retinanet_r50_fpn_1x.py 8
OMP_NUM_THREADS=1 ./tools/dist_train.sh configs/fp16/retinanet_r50_fpn_fp16_1x.py 8

./tools/dist_train.sh configs/faster_rcnn_r50_fpn_1x.py 8
OMP_NUM_THREADS=1 ./tools/dist_train.sh configs/fp16/faster_rcnn_r50_fpn_fp16_1x.py 8

./tools/dist_train.sh configs/mask_rcnn_r50_fpn_1x.py 8
OMP_NUM_THREADS=1 ./tools/dist_train.sh configs/fp16/mask_rcnn_r50_fpn_fp16_1x.py 8
```

---

### Throughput Results
ResNet-50 backbone is adopted. Smaller models put larger stress on the data loader and parameter synchronization system, which better reflects the limit of a framework. Unless otherwise specified, a total batch size of 16 over 8 GPUs is adpoted.

#### P1-FP32
Throughputs are measured in **images/s** and iteration time is reported in the parens.

|model|simpledet|mmdetection|maskrcnn-benchmark|
|-----|------|-----|-------|
|RetinaNet      | 43.4(369ms) | 36.2(442ms) | 44.2(362ms) |
|Faster R-CNN w/ FPN | 43.0(372ms) | 32.8(488ms) | 41.9(382ms) |
|Mask R-CNN w/ FPN   | 35.1(456ms) | 24.7(648ms) | 30.2(530ms) |

#### P2-FP32
Throughputs are measured in **images/s** and iteration time is reported in the parens.

|model|simpledet|mmdetection|maskrcnn-benchmark|
|-----|------|-----|-------|
|RetinaNet      | 55.5(288ms) | 40.1(399ms) | 49.7(322ms) |
|Faster R-CNN w/ FPN | 54.0(296ms) | 38.0(421ms) | 48.3(331ms) |
|Mask R-CNN w/ FPN   | 45.5(352ms) | 29.1(550ms) | 36.0(445ms) |

#### P2-FP16(Mixed Precision)
Throughputs are measured in **images/s** and iteration time is reported in the parens.

|model|simpledet|mmdetection|maskrcnn-benchmark|
|-----|------|-----|-------|
|RetinaNet      | 72.5(221ms) | 46.8(342ms) | NA |
|Faster R-CNN w/ FPN | 70.2(228ms) | 41.8(383ms) | 47.5(337ms) |
|Mask R-CNN w/ FPN   | 58.1(275ms) | 31.1(515ms) | 34.8(460ms) |

#### P2-FP32-1card
Here we provide the single card **iteration time** baseline and the **data time** is reported in the parens.
For a single card, all frameworks give a similar network speed as the underlying cuDNN is the same.
The communication cost and CPU contending is also hidden for a single card.

|model|simpledet|mmdetection|maskrcnn-benchmark|
|-----|------|-----|-------|
|Mask R-CNN w/ FPN   | 313ms(0ms) | 360ms(40ms) | 331ms(12ms) |
