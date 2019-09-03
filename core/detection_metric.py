import mxnet as mx
import numpy as np


class EvalMetricWithSummary(mx.metric.EvalMetric):
    def __init__(self, name, output_names=None, label_names=None, summary=None, **kwargs):
        super().__init__(name, output_names=output_names, label_names=label_names, **kwargs)
        self.summary = summary
        self.global_step = 0

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            self.global_step += 1
            if self.summary:
                self.summary.add_scalar(tag=self.name, 
                    value=self.sum_metric / self.num_inst, global_step=self.global_step)
            return (self.name, self.sum_metric / self.num_inst)


class LossWithIgnore(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        raise NotImplementedError


class FgLossWithIgnore(LossWithIgnore):
    def __init__(self, name, output_names, label_names, bg_label=0, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)
        self.bg_label = bg_label

    def update(self, labels, preds):
        raise NotImplementedError


class AccWithIgnore(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)

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

        pred_label = mx.ndarray.argmax_channel(pred).astype('int32').asnumpy().reshape(-1)
        label = label.astype('int32').asnumpy().reshape(-1)

        keep_inds = np.where(label != self.ignore_label)[0]
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label == label)
        self.num_inst += len(pred_label)


class FgAccWithIgnore(FgLossWithIgnore):
    def __init__(self, name, output_names, label_names, bg_label=0, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, bg_label, ignore_label, **kwargs)

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        pred_label = mx.ndarray.argmax_channel(pred).astype('int32').asnumpy().reshape(-1)
        label = label.astype('int32').asnumpy().reshape(-1)

        keep_inds = np.where((label != self.bg_label) & (label != self.ignore_label))[0]
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label == label)
        self.num_inst += len(pred_label)


class CeWithIgnore(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        label = label.astype('int32').asnumpy().reshape(-1)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))  # -1 x c

        keep_inds = np.where(label != self.ignore_label)[0]
        label = label[keep_inds]
        prob = pred[keep_inds, label]

        prob += 1e-14
        ce_loss = -1 * np.log(prob)
        ce_loss = np.sum(ce_loss)
        self.sum_metric += ce_loss
        self.num_inst += label.shape[0]


class FgCeWithIgnore(FgLossWithIgnore):
    def __init__(self, name, output_names, label_names, bg_label=0, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, bg_label, ignore_label, **kwargs)

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        label = label.astype('int32').asnumpy().reshape(-1)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))  # -1 x c

        keep_inds = np.where((label != self.ignore_label) & (label != self.bg_label))[0]
        label = label[keep_inds]
        prob = pred[keep_inds, label]

        prob += 1e-14
        ce_loss = -1 * np.log(prob)
        ce_loss = np.sum(ce_loss)
        self.sum_metric += ce_loss
        self.num_inst += label.shape[0]


class L1(FgLossWithIgnore):
    def __init__(self, name, output_names, label_names, bg_label=0, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, bg_label, ignore_label, **kwargs)

    def update(self, labels, preds):
        if len(preds) == 1 and len(labels) == 1:
            pred = preds[0].asnumpy()
            label = labels[0].asnumpy()
        elif len(preds) == 2:
            pred = preds[0].asnumpy()
            label = preds[1].asnumpy()
        else:
            raise Exception(
                "unknown loss output: len(preds): {}, len(labels): {}".format(
                    len(preds), len(labels)
                )
            )

        label = label.reshape(-1)
        num_inst = len(np.where((label != self.bg_label) & (label != self.ignore_label))[0])

        self.sum_metric += np.sum(pred)
        self.num_inst += num_inst


class SigmoidCrossEntropy(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)

    def update(self, labels, preds):
        x = preds[0].reshape(-1)  # logit
        z = preds[1].reshape(-1)  # label
        l = mx.nd.relu(x) - x * z + mx.nd.log1p(mx.nd.exp(-mx.nd.abs(x)))
        l = l.mean().asnumpy()

        self.num_inst += 1
        self.sum_metric += l


class ScalarLoss(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names):
        super().__init__(name, output_names, label_names)

    def update(self, labels, preds):

        self.num_inst += 1
        self.sum_metric += preds[0].asscalar()