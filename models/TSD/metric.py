'''
Loss metric for QueryDet training

Author: Chenhongyi Yang 
'''
import numpy as np
import mxnet as mx
import os
from core.detection_metric import EvalMetricWithSummary


class FGAccMetric(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, threshold=0, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)
        self.thr = threshold

    def update(self, labels, preds):
        if len(preds) == 1 and len(labels) == 1:
            pred = preds[0]
            label = labels[0]
        elif len(preds) == 2:
            pred = preds[0]
            label = preds[1]
        else:
            raise Exception(
                "unknown loss output: len(preds): {}, len(labels): {}".format(
                    len(preds), len(labels)
                )
            )

        label = label.asnumpy().astype('int32')
        keep_inds = np.where(label >= 1)

        # treat as foreground if score larger than threshold
        # select class with maximum score as prediction
        pred_score = pred.max(axis=-1)
        pred_label = pred.argmax(axis=-1) + 1

        print(label.shape, pred_label.shape)
        if self.thr != 0:
            pred_label *= pred_score > self.thr

        pred_label = pred_label.asnumpy().astype('int32')

        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class FGAccMetricBinary(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names, threshold=0.5, n_class=80, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)
        self.thr = threshold
        self.n_class = n_class

    def update(self, labels, preds):
        if len(preds) == 2:
            pred = preds[0]
            label = preds[1]

        label = label.asnumpy().astype('int32')
        pred = pred.asnumpy()

        label = label.reshape(-1, self.n_class)
        pred = pred.reshape(-1, self.n_class)
        
        pred_class = np.argmax(pred, axis=1)
        gt_class = np.argmax(label, axis=1)
        pred_score = pred[:, pred_class]

        ind = np.where(label[:,  gt_class] == 1)
        pred_class_fg = pred_class[ind]
        gt_class_fg = gt_class[ind]
        pred_score_fg = pred_score[ind]
        acc = np.sum((pred_class_fg == gt_class_fg) & (pred_score_fg > self.thr)) / (gt_class_fg.shape[0] + 1)

        self.sum_metric += acc
        self.num_inst += 1


class ScalarLossMetric(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names):
        self.__print_name = name
        super(ScalarLossMetric, self).__init__(name, output_names, label_names)
    
    def update(self, labels, preds):
        loss_np = preds[0].asnumpy()
        loss_sum = loss_np.sum()
        # print(self.__print_name, self.num_inst, os.getpid(), loss_sum)
        self.num_inst += 1
        self.sum_metric += loss_sum


class LossMetric(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)

    def update(self, labels, preds):
        self.sum_metric += preds[0].asnumpy().sum()
        self.num_inst += 1


class TensorMetric(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names):
        self.__print_name = name
        super(TensorMetric, self).__init__(name, output_names, label_names)
    
    def update(self, labels, preds):
        loss_np = preds[0].asnumpy()
        loss_mean = loss_np.mean()
        # print(self.__print_name, loss_np.shape, loss_np.reshape(1024, 81, -1).mean(axis=(0,1)))
        print(self.__print_name, loss_np.shape, loss_np.mean(axis=(0)))
        
        self.num_inst += 1
        self.sum_metric += loss_mean
