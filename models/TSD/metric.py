'''
Loss metric for QueryDet training

Author: Chenhongyi Yang 
'''
import numpy as np
import mxnet as mx
from core.detection_metric import EvalMetricWithSummary


class LossMetric(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)

    def update(self, labels, preds):
        self.sum_metric += preds[0].asnumpy().sum()
        self.num_inst += 1

