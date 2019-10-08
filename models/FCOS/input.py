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


class MakeFCOSgt(mx.operator.CustomOp):

    def __init__(self, fcos_gt_setting):
        super(MakeFCOSgt, self).__init__()
        self.p = fcos_gt_setting
        self.stride = self.p.stride
        self.stages = self.p.stages			# type: FCOSFPNAssignParam
        self.ignore_offset = self.p.ignore_offset	# ignore label in IoULoss, as no box can overlap with ([INF, INF], [INF, INF])
        self.ignore_label = self.p.ignore_label

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
        # preparation
        context = in_data[0].context
        bboxes_np = in_data[0].asnumpy()
        if self.loc_x.context != context:
            self.loc_x = self.loc_x.as_in_context(context)
            self.loc_y = self.loc_y.as_in_context(context)
            self.loc_x_T = self.loc_x_T.as_in_context(context)
            self.loc_y_T = self.loc_y_T.as_in_context(context)
            self.stage_lowerbound = self.stage_lowerbound.as_in_context(context)
            self.stage_upperbound = self.stage_upperbound.as_in_context(context)

        bboxes = in_data[0]
        ori_h = in_data[1][0,0]		# aspect_ratio_grouping ensures all aspect ratios within a batch are sames
        ori_w = in_data[1][0,1]
        bs = bboxes.shape[0]

        # compute offset
        if ori_h < ori_w:
            l = mx.nd.broadcast_sub( lhs=self.loc_x.reshape((1,1,-1)), rhs=bboxes[:,:,0:1] )
            t = mx.nd.broadcast_sub( lhs=self.loc_y.reshape((1,1,-1)), rhs=bboxes[:,:,1:2] )
            r = mx.nd.broadcast_sub( lhs=bboxes[:,:,2:3], rhs=self.loc_x.reshape((1,1,-1)) )
            b = mx.nd.broadcast_sub( lhs=bboxes[:,:,3:4], rhs=self.loc_y.reshape((1,1,-1)) )
        else:
            l = mx.nd.broadcast_sub( lhs=self.loc_x_T.reshape((1,1,-1)), rhs=bboxes[:,:,0:1] )
            t = mx.nd.broadcast_sub( lhs=self.loc_y_T.reshape((1,1,-1)), rhs=bboxes[:,:,1:2] )
            r = mx.nd.broadcast_sub( lhs=bboxes[:,:,2:3], rhs=self.loc_x_T.reshape((1,1,-1)) )
            b = mx.nd.broadcast_sub( lhs=bboxes[:,:,3:4], rhs=self.loc_y_T.reshape((1,1,-1)) )
        offset_gt = mx.nd.stack(l,t,r,b, axis=1)				# (N, 4, M, HW)
        ignore_offset = mx.nd.full(offset_gt.shape, self.ignore_offset)
        # clean non-box area
        in_box_area = mx.nd.min(offset_gt, axis=1, keepdims=True) >= 0		# (N, 1, M, HW)
        in_box_area = mx.nd.tile(in_box_area, reps=(1,4,1,1))
        offset_gt = mx.nd.where(condition=in_box_area, x=offset_gt, y=ignore_offset)
        # assign stage
        longest_side = mx.nd.max(offset_gt, axis=1, keepdims=True)		# (N, 1, M, HW)
        stage_assign_mask = mx.nd.logical_and(
                                lhs=mx.nd.broadcast_greater_equal(lhs=longest_side, rhs=self.stage_lowerbound.reshape((1,1,1,-1))), 
                                rhs=mx.nd.broadcast_lesser(lhs=longest_side, rhs=self.stage_upperbound.reshape((1,1,1,-1)))
                              )
        stage_assign_mask = mx.nd.tile(stage_assign_mask, reps=(1,4,1,1))	# (N, 4, M, HW)
        offset_gt = mx.nd.where(condition=stage_assign_mask, x=offset_gt, y=ignore_offset)
        # fuse box offsets based on box size
        box_size = (offset_gt[:,0:1,:,:] + offset_gt[:,2:3,:,:]) * (offset_gt[:,1:2,:,:] + offset_gt[:,3:4,:,:])
        ignore_box_size = mx.nd.full(offset_gt[:,0:1,:,:].shape, 1e10)
        box_size = mx.nd.where(condition=stage_assign_mask[:,0:1,:,:], x=box_size, y=ignore_box_size)
        smallest_box_id = mx.nd.argmin(box_size, axis=2)			# (N, 1, HW)
        smallest_box_id = mx.nd.tile(smallest_box_id, reps=(1,4,1))		# (N, 4, HW)
        target_size = [offset_gt.shape[0], offset_gt.shape[1], offset_gt.shape[3]]
        offset_gt = mx.nd.pick(offset_gt, smallest_box_id, axis=2).reshape(target_size)
                                                                                # (N, 4, HW)

        in_box_area = offset_gt != mx.nd.full(offset_gt.shape, self.ignore_offset)

        # centerness
        l_r_sorted = mx.nd.sort(offset_gt[:,[0,2],:], axis=1)
        term1_min = l_r_sorted[:,0:1,:]
        term1_max = l_r_sorted[:,1:2,:]
        t_b_sorted = mx.nd.sort(offset_gt[:,[1,3],:], axis=1)
        term2_min = t_b_sorted[:,0:1,:]
        term2_max = t_b_sorted[:,1:2,:]
        centerness_gt = mx.nd.sqrt( term1_min * term2_min / (term1_max * term2_max) )
        centerness_gt = centerness_gt * in_box_area[:,0:1,:]

        # cls
        cls_gt = mx.nd.zeros((bs, self.p.num_classifier, self.loc_y.size), dtype=np.float32)
        bbox_cls = []
        batch_idx = []
        spatial_idx = []
        for n in range(bs):
            bbox_cls.append(bboxes[n,:,4].flatten()[smallest_box_id[n,0,:].astype(int)].astype(int))
                                                                               # (HW)
            batch_idx.append(mx.nd.full(smallest_box_id[n,0,:].size, n, dtype=int))
            spatial_idx.append(mx.nd.arange(smallest_box_id[n,0,:].size, dtype=int))
        bbox_cls = mx.nd.concat(*bbox_cls, dim=0).reshape(-1)
        bbox_cls = bbox_cls - 1
        batch_idx = mx.nd.concat(*batch_idx, dim=0)
        spatial_idx = mx.nd.concat(*spatial_idx, dim=0)
        cls_gt[batch_idx, bbox_cls, spatial_idx] = 1
        cls_gt = mx.nd.broadcast_mul(lhs=cls_gt, rhs=in_box_area[:,0:1,:])
        
        # ignore label
        if ori_h < ori_w:
            ignore_area = 1 - mx.nd.logical_or(lhs=(self.loc_x>=ori_w), rhs=(self.loc_y>=ori_h))
        else:
            ignore_area = 1 - mx.nd.logical_or(lhs=(self.loc_x_T>=ori_w), rhs=(self.loc_y_T>=ori_h))

        ignore_area = mx.nd.tile(ignore_area.reshape((1,1,-1)), reps=(bs,1,1))
        ignore_label = mx.nd.full(centerness_gt.shape, self.ignore_label)
        centerness_gt = mx.nd.where(condition=ignore_area, x=centerness_gt, y=ignore_label)

        ignore_area = mx.nd.tile(ignore_area, reps=(1,self.p.num_classifier,1))
        ignore_label = mx.nd.full(cls_gt.shape, self.ignore_label)
        cls_gt = mx.nd.where(condition=ignore_area, x=cls_gt, y=ignore_label)


        self.assign(out_data[0], req[0], centerness_gt.reshape((0,-1)))
        self.assign(out_data[1], req[1], cls_gt.reshape((0,-1)))
        self.assign(out_data[2], req[2], offset_gt)

        return

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register("make_fcos_gt")
class MakeFCOSGTProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MakeFCOSGTProp, self).__init__(need_top_grad=False)
        from config.fcos_r50v1_fpn_1x import throwout_param
        self.p = throwout_param
        self.stride = self.p.stride
        self.data_size = self.p.data_size

    def list_arguments(self):
        return ['gt_bbox', 'im_info']

    def list_outputs(self):
        return ['centerness_gt', 'cls_gt', 'offset_gt']

    def infer_shape(self, in_shape):
        n = in_shape[0][0]
        h, w = self.data_size
        hw = 0
        for stride in self.stride:
            width = len(range(0,w,stride))
            height = len(range(0,h,stride))
            hw += height * width
        return in_shape, [[n,hw], [n,self.p.num_classifier*hw], [n,4,hw]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0] for _ in range(3)], []

    def create_operator(self, ctx, shapes, dtypes):
        return MakeFCOSgt(self.p)
