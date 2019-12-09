import numpy as np
import mxnet as mx
import mxnext as X

# Convert classification/centerness prediction maps into proposal bbox
#   Used in only one FPN scale
class getProposalSingleStage(mx.operator.CustomOp):
    def __init__(self, top_n, stride, pre_nms_thresh):
        super(getProposalSingleStage, self).__init__()
        self.top_n = int(top_n)
        self.stride = int(stride)
        self.pre_nms_thresh = float(pre_nms_thresh)

    def forward(self, is_train, req, in_data, out_data, aux):
        # filter out low prob locations
        cls_logits = in_data[1]
        candidate_mask = mx.nd.broadcast_greater( lhs=cls_logits, rhs=mx.nd.full((1,1,1,1), float(self.pre_nms_thresh)) )
        # fuse centerness and classification prob
        cls_logits = mx.nd.broadcast_mul(lhs=cls_logits, rhs=in_data[0])

        n, c, h, w = in_data[1].shape
        res_bbox = mx.nd.full([n, self.top_n, 6], -1, ctx=cls_logits.context)	# 6 for [ch id + 1 (cls), score, tp_x, tp_y, rb_x, rb_y]

        for i in range(n):

            cls_logit = cls_logits[i,:,:,:]
            offset_logit = in_data[2][i,:,:,:]
            img_h, img_w, _ = in_data[3][i,:].asnumpy()

            # find topK activation locations
            # if the number of non-empty locations is less than K, then use all locations
            if mx.nd.sum(candidate_mask[i,:,:,:]) >= self.top_n:		# non-empty loc is more than K
                cls_score, idx = mx.nd.topk(mx.nd.reshape(cls_logit, shape=(c*h*w)), axis=0, ret_typ='both', k=self.top_n)
                loc_x = (idx % w).astype(int)
                loc_y = (idx / w % h).astype(int)
                cls = ((idx / w / h).astype(int) + 1).astype(int)
            else:
                candidate_np = candidate_mask[i,:,:,:].asnumpy()
                cls, loc_y, loc_x = np.nonzero(candidate_np)
                if cls.size == 0:						# for all-zero logit map, skip it
                    continue
                cls_score = cls_logit[cls, loc_y, loc_x]
                loc_y = mx.nd.array(loc_y, cls_logit.context).astype(int)
                loc_x = mx.nd.array(loc_x, cls_logit.context).astype(int)
                cls = mx.nd.array(cls, cls_logit.context).astype(int)
                cls += 1

            # transform coords of topK locations to image coord space
            loc_x_ori = loc_x.astype(np.float32) * self.stride + self.stride / 2
            loc_y_ori = loc_y.astype(np.float32) * self.stride + self.stride / 2
            # calculate bbox left-top and right-bottom coords, and clip to image boundary
            bbox = [mx.nd.clip(loc_x_ori - offset_logit[0,loc_y,loc_x], a_min=0, a_max=img_w),	# [tp_x, tp_y, rb_x, rb_y]
                    mx.nd.clip(loc_y_ori - offset_logit[1,loc_y,loc_x], a_min=0, a_max=img_h),
                    mx.nd.clip(loc_x_ori + offset_logit[2,loc_y,loc_x], a_min=0, a_max=img_w),
                    mx.nd.clip(loc_y_ori + offset_logit[3,loc_y,loc_x], a_min=0, a_max=img_h),
                   ]
            bbox = mx.nd.stack(*bbox, axis=1)
            bbox = mx.nd.concat(cls.astype(np.float32).reshape(shape=(cls.shape[0],1)), cls_score.reshape(shape=(cls_score.shape[0],1)), bbox, dim=1)

            # remove small bboxes
            small_bbox_mask = (bbox[:,0]>=bbox[:,2]) * (bbox[:,1]>=bbox[:,3])
            small_bbox_mask = small_bbox_mask.reshape((0,1))
            bbox = mx.nd.broadcast_mul(lhs=bbox, rhs=1-small_bbox_mask)
            bbox = mx.nd.broadcast_add(lhs=bbox, rhs=-small_bbox_mask)

            res_bbox[i,:bbox.shape[0],:] = bbox

        self.assign(out_data[0], req[0], res_bbox)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register("get_proposal_single_stage")
class GetProposalSingleStageProp(mx.operator.CustomOpProp):
    def __init__(self, pre_nms_top_n, stride, pre_nms_thresh):
        super(GetProposalSingleStageProp, self).__init__(need_top_grad=False)
        self.top_n = pre_nms_top_n
        self.stride = stride
        self.pre_nms_thresh = pre_nms_thresh

    def list_arguments(self):
        return ['centerness_logit','cls_logit','offset_logit','img_info']

    def list_outputs(self):
        return ['res_bbox']

    def infer_shape(self, in_shape):
        return in_shape, [[in_shape[1][0], int(self.top_n), 6]], []

    def infer_type(self, in_type):
        return in_type, [in_type[2]], []

    def create_operator(self, ctx, shapes, dtypes):
        return getProposalSingleStage(self.top_n, self.stride, self.pre_nms_thresh)


# Conduct bbox NMS within a batch
#   Use -1 as ignore flag
class GetBatchProposal(mx.operator.CustomOp):
    def __init__(self):
        super(GetBatchProposal, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        n = in_data[0].shape[0]
        res_bbox = in_data[0]

        # score + cls_id -> score vectors
        scores = np.zeros((n, res_bbox.shape[1], 81)).astype(np.float32)

        for i in range(n):
            # sort all bboxes by scores
            sorted_id = mx.nd.argsort(res_bbox[i,:,1], is_ascend=False).astype(int)
            res_bbox[i,:,:] = res_bbox[i,sorted_id,:]

            # filter out all 0 items
            cls = res_bbox[i,:,0]
            res_bbox = res_bbox.asnumpy()
            # sqrt(score)
            scores[i, np.array(range(res_bbox.shape[1])), res_bbox[i,:,0].astype(int)] = np.sqrt(np.clip(res_bbox[i,:,1], a_min=1e-20, a_max=1))
            res_bbox = mx.nd.array(res_bbox, cls.context)

        self.assign(out_data[0], req[0], res_bbox[:,:,2:])
        self.assign(out_data[1], req[1], mx.nd.array(scores, res_bbox.context))
        self.assign(out_data[2], req[2], res_bbox[:,:,0])


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register("get_batch_proposal")
class GetBatchProposalProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(GetBatchProposalProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['bbox']

    def list_outputs(self):
        return ['res_bbox', 'score', 'cls_id']

    def infer_shape(self, in_shape):
        a, b, c = in_shape[0]
        return in_shape, [[a,b,c-2], [a,b,81], [a,b,1]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0], in_type[0], in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return GetBatchProposal()
