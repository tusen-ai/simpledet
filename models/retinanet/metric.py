import numpy as np
import mxnet as mx


class FGAccMetric(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names):
        super(FGAccMetric, self).__init__(name, output_names, label_names)

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

        # label (b, p)
        label = label.asnumpy().astype('int32')
        keep_inds = np.where(label >= 1)

        if pred.shape[-1] == 1:
            # treat as foreground if score larger than 0.05
            pred_label = pred[:, :, 0] > 0.05
            label = label[keep_inds]
        else:
            # pred (b, p, c) , c is 0 ~ CLASSNUM-1
            pred_label = mx.nd.argmax(pred, axis=2)
            # label is 1 ~ CLASSNUM , so label = label - 1
            label = label[keep_inds] - 1

        pred_label = pred_label.asnumpy().astype('int32')

        pred_label = pred_label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)
