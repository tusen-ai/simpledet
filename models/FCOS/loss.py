import mxnet as mx
import mxnext as X

# Use symbol for internal variables computation for better parallelization
# Use custom python op to control loss/gradient flow

""" ---Sigmoid Focal Loss---
    def forward(self, is_train, req, in_data, out_data, aux):
        logits = in_data[0]
        labels = in_data[1]
        nonignore_mask = in_data[2]

        p = 1 / (1 + mx.nd.exp(-logits))
        mask_logits_GE_zero = mx.nd.broadcast_greater_equal(lhs=logits, rhs=mx.nd.zeros((1,1)))
        minus_abs_logits = logits - 2*logits*mask_logits_GE_zero  #mx.nd.abs(logits)

        term1 = self.alpha * \
                (1-p)**self.gamma * \
                mx.nd.log(mx.nd.clip(p, a_min=1e-5, a_max=1)) * \
                labels
        term2 = (1 - self.alpha) * \
                p**self.gamma * \
                ( -1. * logits * mask_logits_GE_zero - mx.nd.log(1. + mx.nd.exp(minus_abs_logits)) ) * \
                (1 - labels)

        loss = -1 * (term1 + term2) * nonignore_mask / (mx.nd.sum(nonignore_mask) + in_data[0].shape[0])

        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        logits = in_data[0]
        labels = in_data[1]
        nonignore_mask = in_data[2]

        p = 1 / (1 + mx.nd.exp(-logits))
        mask_logits_GE_zero = mx.nd.broadcast_greater_equal(lhs=logits, rhs=mx.nd.zeros((1,1)))
        minus_abs_logits = logits - 2*logits*mask_logits_GE_zero  #mx.nd.abs(logits)

        term1 = self.alpha * \
                (1-p)**self.gamma * \
                ( 1 - p - p * self.gamma * mx.nd.log(mx.nd.clip(p, a_min=1e-5, a_max=1)) ) * \
                labels
        term2 = (1 - self.alpha) * \
                p**self.gamma * \
                (( -1. * logits * mask_logits_GE_zero - mx.nd.log(1. + mx.nd.exp(minus_abs_logits)) ) * (1 - p) * self.gamma - p) * \
                (1 - labels)

        grad = -1 * (term1 + term2) * nonignore_mask / (mx.nd.sum(nonignore_mask) + in_data[0].shape[0])

        self.assign(in_grad[0], req[0], grad)"""


# the formula is better demonstrated above
class ComputeSigmoidFocalLoss(mx.operator.CustomOp):
    def __init__(self):
        super(ComputeSigmoidFocalLoss, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        loss = in_data[1]
        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = in_data[2]
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("compute_focal_loss")
class ComputeSigmoidFocalLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ComputeSigmoidFocalLossProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['logits', 'loss', 'grad']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        return in_shape, [[1]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return ComputeSigmoidFocalLoss()

def make_sigmoid_focal_loss(gamma, alpha, logits, labels, nonignore_mask):
    p = 1 / (1 + mx.sym.exp(-logits))					# sigmoid
    mask_logits_GE_zero = mx.sym.broadcast_greater_equal(lhs=logits, rhs=mx.sym.zeros((1,1)))
									# logits>=0
    minus_logits_mask = -1. * logits * mask_logits_GE_zero		# -1 * logits * [logits>=0]
    negative_abs_logits = logits - 2*logits*mask_logits_GE_zero		# logtis - 2 * logits * [logits>=0]
    log_one_exp_minus_abs = mx.sym.log(1. + mx.sym.exp(negative_abs_logits))
    minus_log = minus_logits_mask - log_one_exp_minus_abs

    alpha_one_p_gamma_labels = alpha * (1-p)**gamma * labels
    log_p_clip = mx.sym.log(mx.sym.clip(p, a_min=1e-5, a_max=1)) 
    one_alpha_p_gamma_one_labels = (1 - alpha) * p**gamma * (1 - labels)
    norm = mx.sym.sum(labels*nonignore_mask) + 1

    forward_term1 = alpha_one_p_gamma_labels * log_p_clip
    forward_term2 = one_alpha_p_gamma_one_labels * minus_log
    loss = mx.sym.sum(-1 * (forward_term1 + forward_term2) * nonignore_mask) / norm

    backward_term1 = alpha_one_p_gamma_labels * ( 1 - p - p * gamma * log_p_clip )
    backward_term2 = one_alpha_p_gamma_one_labels * (minus_log  * (1 - p) * gamma - p)
    grad = mx.sym.broadcast_div( lhs=-1 * (backward_term1 + backward_term2) * nonignore_mask, rhs=norm.reshape((1,1)) )

    loss = X.block_grad(loss)						# symbols are only used for computation
    grad = X.block_grad(grad)						# use custom op to control gradient flow instead

    loss = mx.sym.Custom(logits=logits, loss=loss, grad=grad, op_type='compute_focal_loss', name='focal_loss')
    return loss

# -------------------------------------------------------
class ComputeBCELoss(mx.operator.CustomOp):
    def __init__(self):
        super(ComputeBCELoss, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        loss = in_data[1]
        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = in_data[2]
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("compute_bce_loss")
class ComputeBCELossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ComputeBCELossProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['logits', 'loss', 'grad']

    def list_outputs(self):
        return ['bce_loss']

    def infer_shape(self, in_shape):
        return in_shape, [[1]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return ComputeBCELoss()

def make_binary_cross_entropy_loss(logits, labels, nonignore_mask):
    p = 1 / (1 + mx.sym.exp(-logits))
    loss = -labels * mx.sym.log(mx.sym.clip(p, a_min=1e-5, a_max=1)) - (1 - labels) * mx.sym.log(mx.sym.clip(1 - p, a_min=1e-5, a_max=1))
    loss = mx.sym.sum(loss * nonignore_mask) / (mx.sym.sum(nonignore_mask) + 1e-30)
    grad = mx.sym.broadcast_div( lhs=(p - labels) * nonignore_mask, rhs=mx.sym.sum(nonignore_mask) + 1e-30 )

    loss = X.block_grad(loss)
    grad = X.block_grad(grad)

    return mx.sym.Custom(logits=logits, loss=loss, grad=grad, op_type='compute_bce_loss', name='sigmoid_bce_loss')

# -------------------------------------------------------
def IoULoss(x_box, y_box, ignore_offset, centerness_label, name='iouloss'):
    centerness_label = mx.sym.reshape(centerness_label, shape=(0,1,-1))
    y_box = X.block_grad(y_box)

    target_left = mx.sym.slice_axis(y_box, axis=1, begin=0, end=1)
    target_top = mx.sym.slice_axis(y_box, axis=1, begin=1, end=2)
    target_right = mx.sym.slice_axis(y_box, axis=1, begin=2, end=3)
    target_bottom = mx.sym.slice_axis(y_box, axis=1, begin=3, end=4)

    # filter out out-of-bbox area, loss is only computed inside bboxes
    nonignore_mask = mx.sym.broadcast_logical_and(lhs = mx.sym.broadcast_not_equal(lhs=target_left, rhs=ignore_offset),
                                              rhs = mx.sym.broadcast_greater( lhs=centerness_label, rhs=mx.sym.full((1,1,1), 0) )
                                             )
    nonignore_mask = X.block_grad(nonignore_mask)
    x_box = mx.sym.clip(x_box, a_min=0, a_max=1e4)
    x_box = mx.sym.broadcast_mul(lhs=x_box, rhs=nonignore_mask)
    centerness_label = centerness_label * nonignore_mask

    pred_left = mx.sym.slice_axis(x_box, axis=1, begin=0, end=1)
    pred_top = mx.sym.slice_axis(x_box, axis=1, begin=1, end=2)
    pred_right = mx.sym.slice_axis(x_box, axis=1, begin=2, end=3)
    pred_bottom = mx.sym.slice_axis(x_box, axis=1, begin=3, end=4)

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = mx.sym.min(mx.sym.stack(pred_left, target_left, axis=0), axis=0) + mx.sym.min(mx.sym.stack(pred_right, target_right, axis=0), axis=0)
    h_intersect = mx.sym.min(mx.sym.stack(pred_bottom, target_bottom, axis=0), axis=0) + mx.sym.min(mx.sym.stack(pred_top, target_top, axis=0), axis=0)

    area_intersect = w_intersect * h_intersect
    area_union = (target_area + pred_area - area_intersect)

    loss = -mx.sym.log((area_intersect + 1.0) / (area_union + 1.0))

    loss = mx.sym.broadcast_mul(lhs=loss, rhs=centerness_label)
    loss = mx.sym.sum(loss) / (mx.sym.sum(centerness_label) + 1e-30)

    return X.loss(loss, grad_scale=1, name=name)
