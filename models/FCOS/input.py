from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import mxnet as mx
import time

class DetectionAugmentation(object):
    def __init__(self):
        pass

    def apply(self, input_record):
        pass


class PreMakeFCOSgt(mx.operator.CustomOp):

    def __init__(self, fcos_gt_setting):
        super(PreMakeFCOSgt, self).__init__()
        self.p = fcos_gt_setting
        self.stride = self.p.stride
        self.stages = self.p.stages			# type: FCOSFPNAssignParam

        # make locations
        self.data_size = self.p.data_size
        h, w = self.data_size
        self.loc_x = []
        self.loc_y = []
        self.loc_x_T = []
        self.loc_y_T = []
        self.stage_lowerbound = [-1e-5, 64, 128, 256, 512]
        self.stage_upperbound = [64, 128, 256, 512, 1e5]
        for idx, stride in enumerate(self.stride):
            x = np.array(range(0,w,stride), dtype=np.float32) + stride/2.
            y = np.array(range(0,h,stride), dtype=np.float32) + stride/2.
            x, y = np.meshgrid(x, y)
            x = mx.nd.from_numpy(x)
            y = mx.nd.from_numpy(y)
            self.loc_x.append(x.reshape(-1))
            self.loc_y.append(y.reshape(-1))
            self.loc_x_T.append(y.T.reshape(-1))
            self.loc_y_T.append(x.T.reshape(-1))
            self.stage_lowerbound[idx] = mx.nd.full(self.loc_x[-1].shape, self.stage_lowerbound[idx])
            self.stage_upperbound[idx] = mx.nd.full(self.loc_x[-1].shape, self.stage_upperbound[idx])
        self.loc_x = mx.nd.concat(*(self.loc_x), dim=0)
        self.loc_y = mx.nd.concat(*(self.loc_y), dim=0)
        self.loc_x_T = mx.nd.concat(*(self.loc_x_T), dim=0)
        self.loc_y_T = mx.nd.concat(*(self.loc_y_T), dim=0)
        self.stage_lowerbound = mx.nd.concat(*(self.stage_lowerbound), dim=0)
        self.stage_upperbound = mx.nd.concat(*(self.stage_upperbound), dim=0)

    def forward(self, is_train, req, in_data, out_data, aux):
        context = in_data[0].context
        if self.loc_x.context != context:		# execute only once
            self.loc_x = self.loc_x.as_in_context(context)
            self.loc_y = self.loc_y.as_in_context(context)
            self.loc_x_T = self.loc_x_T.as_in_context(context)
            self.loc_y_T = self.loc_y_T.as_in_context(context)
            self.stage_lowerbound = self.stage_lowerbound.as_in_context(context)
            self.stage_upperbound = self.stage_upperbound.as_in_context(context)

        ori_h = in_data[1][0,0]				# aspect_ratio_grouping ensures all aspect ratios within a batch are same
        ori_w = in_data[1][0,1]

        if ori_h < ori_w:
            self.assign(out_data[0], req[0], self.loc_x)
            self.assign(out_data[1], req[1], self.loc_y)
            nonignore_area = mx.nd.logical_and(lhs=(self.loc_x<ori_w), rhs=(self.loc_y<ori_h))
        else:
            self.assign(out_data[0], req[0], self.loc_x_T)
            self.assign(out_data[1], req[1], self.loc_y_T)
            nonignore_area = mx.nd.logical_and(lhs=(self.loc_x_T<ori_w), rhs=(self.loc_y_T<ori_h))

        self.assign(out_data[2], req[2], self.stage_lowerbound)
        self.assign(out_data[3], req[3], self.stage_upperbound)
        self.assign(out_data[4], req[4], nonignore_area)

        return

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register("make_fcos_gt_preparation")
class PreMakeFCOSGTProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(PreMakeFCOSGTProp, self).__init__(need_top_grad=False)
        from config.fcos_r50v1_fpn_1x import throwout_param
        self.p = throwout_param
        self.stride = self.p.stride
        self.data_size = self.p.data_size

    def list_arguments(self):
        return ['gt_bbox', 'im_info']

    def list_outputs(self):
        return ['loc_x', 'loc_y', 'stage_lowerbound', 'stage_upperbound', 'nonignore_area']

    def infer_shape(self, in_shape):
        n = in_shape[0][0]
        h, w = self.data_size
        hw = 0
        for stride in self.stride:
            width = len(range(0,w,stride))
            height = len(range(0,h,stride))
            hw += height * width
        return in_shape, [[hw], [hw], [hw], [hw], [hw]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0] for _ in range(5)], []

    def create_operator(self, ctx, shapes, dtypes):
        return PreMakeFCOSgt(self.p)


def make_fcos_gt(gt_bbox, im_info, ignore_offset, ignore_label, num_classifier):
    loc_x, loc_y, stage_lowerbound, stage_upperbound, nonignore_area = mx.sym.Custom(gt_bbox=gt_bbox, im_info=im_info, op_type='make_fcos_gt_preparation', name='pre_fcos_gt')

    bboxes = gt_bbox

    # compute offset
    #bboxes_ = mx.sym.expand_dims(bboxes, axis=-1)
    l = mx.sym.broadcast_sub( lhs=loc_x, rhs=mx.sym.slice(bboxes, begin=(None,None,0), end=(None,None,1)) )
    t = mx.sym.broadcast_sub( lhs=loc_y, rhs=mx.sym.slice(bboxes, begin=(None,None,1), end=(None,None,2)) )
    r = mx.sym.broadcast_sub( lhs=mx.sym.slice(bboxes, begin=(None,None,2), end=(None,None,3)), rhs=loc_x )
    b = mx.sym.broadcast_sub( lhs=mx.sym.slice(bboxes, begin=(None,None,3), end=(None,None,4)), rhs=loc_y )
    offset_gt = mx.sym.stack(l,t,r,b, axis=1)					# (N, 4, M, HW)
    # clean non-box area
    in_box_area = mx.sym.min(offset_gt, axis=1, keepdims=True) >= 0		# (N, 1, M, HW)
    offset_gt = mx.sym.broadcast_add( lhs=mx.sym.broadcast_mul(lhs=offset_gt, rhs=in_box_area), 
                                      rhs=(1 - in_box_area) * ignore_offset
                                    )						# offset_gt[!in_box_area] = self.ignore_offset
    # assign stage
    longest_side = mx.sym.max(offset_gt, axis=1, keepdims=True)			# (N, 1, M, HW)
    stage_assign_mask = mx.sym.broadcast_logical_and( lhs=mx.sym.broadcast_greater_equal(lhs=longest_side, rhs=stage_lowerbound), 
                                                      rhs=mx.sym.broadcast_lesser(lhs=longest_side, rhs=stage_upperbound)
                                                    )					# (N, 1, M, HW)
    offset_gt = mx.sym.broadcast_add( lhs=mx.sym.broadcast_mul(lhs=offset_gt, rhs=stage_assign_mask), 
                                      rhs=(1 - stage_assign_mask) * ignore_offset
                                    )						# offset[!stage_assign_mask] = self.ignore_offset
    # fuse box offsets based on box size
    box_size = ( mx.sym.slice(offset_gt, begin=(None,0,None,None), end=(None,1,None,None)) + \
                 mx.sym.slice(offset_gt, begin=(None,2,None,None), end=(None,3,None,None)) ) \
             * ( mx.sym.slice(offset_gt, begin=(None,1,None,None), end=(None,2,None,None)) + \
                 mx.sym.slice(offset_gt, begin=(None,3,None,None), end=(None,4,None,None)) )
										# (offset_gt[:,0,:,:] + offset_gt[:,2,:,:]) * (offset_gt[:,1,:,:] + offset_gt[:,3,:,:])
    #box_size = mx.sym.expand_dims(box_size, axis=1)
    box_size = mx.sym.broadcast_add( lhs=mx.sym.broadcast_mul(lhs=box_size, rhs=stage_assign_mask), 
                                     rhs=(1 - stage_assign_mask) * 1e10
                                   )						# box[!stage_assign_mask] = MAX_BBOX_SIZE
    smallest_box_id = mx.sym.argmin(box_size, axis=2)				# (N, 1, HW)
    smallest_box_ids = mx.sym.tile(smallest_box_id, reps=(1,4,1))		# (N, 4, HW)
    offset_gt = mx.sym.reshape_like( mx.sym.pick(offset_gt, smallest_box_ids, axis=2), smallest_box_ids )
                                                                                # (N, 4, HW)

    in_box_area = offset_gt != ignore_offset

    # centerness
    l_r_sorted = mx.sym.sort(mx.sym.slice(offset_gt, begin=(None,0,None), end=(None,3,None), step=(None,2,None)), axis=1)
										# mx.nd.sort(offset_gt[:,[0,2],:], axis=1)
    term1_min = mx.sym.reshape(mx.sym.slice(l_r_sorted, begin=(None,0,None), end=(None,1,None)), shape=(0,-1))
										# l_r_sorted[:,0,:]
    term1_max = mx.sym.reshape(mx.sym.slice(l_r_sorted, begin=(None,1,None), end=(None,2,None)), shape=(0,-1))
										# l_r_sorted[:,1,:]
    t_b_sorted = mx.sym.sort(mx.sym.slice(offset_gt, begin=(None,1,None), end=(None,4,None), step=(None,2,None)), axis=1)
										# mx.nd.sort(offset_gt[:,[1,3],:], axis=1)
    term2_min = mx.sym.reshape(mx.sym.slice(t_b_sorted, begin=(None,0,None), end=(None,1,None)), shape=(0,-1))
										# t_b_sorted[:,0,:]
    term2_max = mx.sym.reshape(mx.sym.slice(t_b_sorted, begin=(None,1,None), end=(None,2,None)), shape=(0,-1))
										# t_b_sorted[:,1,:]
    centerness_gt = mx.sym.sqrt( term1_min * term2_min / (term1_max * term2_max) )
    centerness_gt = centerness_gt * mx.sym.reshape(mx.sym.slice(in_box_area, begin=(None,0,None), end=(None,1,None)), shape=(0,-1))
										# (N, HW), centerness_gt*in_box_area[:,0,:]

    # cls
    smallest_box_id = smallest_box_id.reshape((0,-1))				# (N, HW)
    cls_gt = mx.sym.one_hot(smallest_box_id, num_classifier)			# (N, HW, num_cls)
    cls_gt = mx.sym.transpose(cls_gt, axes=(0,2,1))				# (N, num_cls, HW)
    cls_gt = mx.sym.broadcast_mul( lhs=cls_gt, rhs=mx.sym.slice(in_box_area, begin=(None,0,None), end=(None,1,None)) )
        
    # ignore label
    nonignore_area = mx.sym.reshape(nonignore_area, shape=(1,-1))
    centerness_gt = mx.sym.broadcast_add( lhs=mx.sym.broadcast_mul(lhs=centerness_gt, rhs=nonignore_area), 
                                          rhs=(1 - nonignore_area) * ignore_label
                                        )					# centerness_gt[!nonignore_area] = self.ignore_label
    nonignore_area = mx.sym.reshape(nonignore_area, shape=(1,1,-1))
    cls_gt = mx.sym.broadcast_add( lhs=mx.sym.broadcast_mul(lhs=cls_gt, rhs=nonignore_area), 
                                   rhs=(1 - nonignore_area) * ignore_label
                                 )		

    return centerness_gt, mx.sym.reshape(cls_gt, shape=(0,-1)), offset_gt
