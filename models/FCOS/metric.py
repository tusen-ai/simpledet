import mxnet as mx
import numpy as np

class LossWithIgnore(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names, ignore_label=-1):
        super().__init__(name, output_names, label_names)
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        raise NotImplementedError

class ClsAccWithIgnore(LossWithIgnore):
    def __init__(self, stride, name, output_names, label_names, ignore_label=-1):
        super().__init__(name, output_names, label_names, ignore_label)
        self.stride = stride

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0

    def update(self, labels, preds):
        pred = preds[1]
        label = labels[len(stride):len(stride)*2]

        for stride in self.stride:
            self.sum_metric += mx.nd.sum(mx.nd.logical_and(pred[stride]>0.5, label[stride]))
            self.num_inst += mx.nd.sum(label[stride])

class LossMeter(mx.metric.EvalMetric):
    def __init__(self, stride, pred_id_start, pred_id_end, name='LossMeter'):
        self.stride = stride
        self.pred_id_start = pred_id_start
        self.pred_id_end = pred_id_end
        super(LossMeter, self).__init__(name=name)

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0

    def update(self, labels, preds):
        for i, pred in enumerate(preds[self.pred_id_start:self.pred_id_end]):
            if len(pred.shape) > 1:
                valid_pred = pred.mean().asnumpy()
            else:
                valid_pred = pred.asnumpy()

            self.sum_metric += valid_pred
            self.num_inst += +1
