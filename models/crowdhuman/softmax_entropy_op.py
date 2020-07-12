import numpy as np
import mxnet as mx

class SoftmaxEntropyOperator(mx.operator.CustomOp):
    def __init__(self):
        super().__init__()
    
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        label = in_data[1]
        
        num_reg_class = data.shape[-1]
        label = mx.nd.one_hot(label, depth=num_reg_class)
        
        data = mx.nd.softmax(data, axis=-1)
        loss = - label * mx.nd.log(data + 1e-10)
        self.assign(out_data[0], req[0], loss)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0]
        label = in_data[1]
        
        batch_roi, num_reg_class = data.shape
        onehot_label = mx.nd.one_hot(label, depth=num_reg_class)

        d_grad = mx.nd.softmax(data, axis=-1) - onehot_label
        # since we directly backward grad from here, we need to normalize gradient right!
        d_grad *= out_grad[0]
        
        self.assign(in_grad[0], req[0], d_grad)
        self.assign(in_grad[1], req[1], mx.nd.zeros_like(label))
        

@mx.operator.register('softmax_entropy')
class SoftmaxEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super().__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data', 'label']
    
    def list_outputs(self):
        return ['output']
    
    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[1]], [in_shape[0]]
    
    def create_operator(self, ctx, shapes, dtypes):
        return SoftmaxEntropyOperator()