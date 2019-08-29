import numpy as np
import mxnet as mx

from core.detection_metric import EvalMetricWithSummary


class SigmoidCELossMetric(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)

    def update(self, labels, preds):
        self.sum_metric += preds[0].mean().asscalar()
        self.num_inst += 1