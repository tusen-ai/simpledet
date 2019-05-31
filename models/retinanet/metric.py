import numpy as np
import mxnet as mx


class FGAccMetric(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names, threshold=0):
        super().__init__(name, output_names, label_names)
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
        pred_label *= pred_score > self.thr

        pred_label = pred_label.asnumpy().astype('int32')

        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)
