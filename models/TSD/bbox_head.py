from __future__ import print_function

import mxnet as mx
import mxnext as X
from symbol.builder import BboxHead

from models.TSD.poolings import FPNRoIAlign_DeltaC, FPNRoIAlign_DeltaR

from shape_tool import infer_shape


def get_iou_mat(rois, gt_boxes, l1, l2):
    # compute IoU matrix between a set of boxes and gt_boxes
    rois = mx.sym.reshape(rois, (l1, 1, 4))
    gt_boxes = mx.sym.reshape(gt_boxes, (1, l2, 5))
    rois = mx.sym.tile(rois, (1, l2, 1))
    gt_boxes = mx.sym.tile(gt_boxes, (l1, 1, 1))

    xmin1, ymin1, xmax1, ymax1 = mx.sym.split(rois, axis=2, num_outputs=4, squeeze_axis=True)
    xmin2, ymin2, xmax2, ymax2, _ = mx.sym.split(gt_boxes, axis=2, num_outputs=5, squeeze_axis=True)

    x1 = mx.sym.maximum(xmin1, xmin2)
    y1 = mx.sym.maximum(ymin1, ymin2)
    x2 = mx.sym.minimum(xmax1, xmax2)
    y2 = mx.sym.minimum(ymax1, ymax2)
    
    w = mx.sym.maximum(x2-x1, 0)
    h = mx.sym.maximum(y2-y1, 0)

    iou = (w*h) / (((xmax1-xmin1)*(ymax1-ymin1)) + ((xmax2-xmin2)*(ymax2-ymin2)) - (w*h))
    return iou 


def get_iou(bboxs1, bboxes2):
    # compute per box IoU between two set of boxes
    xmin1, ymin1, xmax1, ymax1 = mx.sym.split(bboxs1, axis=-1, num_outputs=4, squeeze_axis=True)
    xmin2, ymin2, xmax2, ymax2 = mx.sym.split(bboxes2, axis=-1, num_outputs=4, squeeze_axis=True)

    x1 = mx.sym.maximum(xmin1, xmin2)
    y1 = mx.sym.maximum(ymin1, ymin2)
    x2 = mx.sym.minimum(xmax1, xmax2)
    y2 = mx.sym.minimum(ymax1, ymax2)

    w = mx.sym.maximum(x2-x1, 0)
    h = mx.sym.maximum(y2-y1, 0)

    iou = (w*h) / (((xmax1-xmin1)*(ymax1-ymin1)) + ((xmax2-xmin2)*(ymax2-ymin2)) - (w*h))
    return iou


class TSDConvFCBBoxHead(BboxHead):
    def __init__(self, pBbox, delta_c_pool: FPNRoIAlign_DeltaC, delta_r_pool: FPNRoIAlign_DeltaR):
        super().__init__(pBbox)
        self.delta_c_pool = delta_c_pool
        self.delta_r_pool = delta_r_pool       

    def get_output(self, fpn_conv_feats, roi_feat, rois, is_train):
        '''
        Args:
            fpn_conv_feats: dict of FPN features, each [batch_size, in_channels, fh, fw]
            roi_feat: [batch_size * image_roi, 256, roi_size, roi_size]
            rois: [batch_size, image_roi, 4]
            is_train: boolean
        Returns:
            cls_logit: [batch_size * image_roi, num_class]
            bbox_delta: [batch_size * image_roi, num_class * 4]
            tsd_cls_logit: [batch_size * image_roi, num_class]
            tsd_bbox_delta: [batch_size * image_roi, num_class * 4]
            delta_c: [batch_size * image_roi, 2*roi_size*roi_size, 1, 1]
            delta_r: [batch_size * image_roi, 2, 1, 1]
        '''
        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)
        # roi_feat: [batch_roi, 256, 7, 7]

        flatten = X.reshape(roi_feat, shape=(0, -1, 1, 1), name="bbox_feat_reshape") # [batch_roi, 256*7*7, 1, 1]
        
        x1 = flatten
        x2 = X.relu(X.conv(data=x1, kernel=1, filter=256, name="delta_shared_fc1", no_bias=False), name="delta_shared_fc1_relu") # [batch_roi, 256, 1, 1]

        delta_c = X.relu(X.conv(x2, filter=256, name="delta_c_fc1", init=X.gauss(0.01)), name="delta_c_fc1_relu") # [batch_roi, 256, 1, 1]
        delta_c = X.conv(delta_c, filter=2*self.p.roi_size**2, name="delta_c_fc2", init=X.gauss(0.01)) # [batch_roi, 2*7*7, 1, 1]

        delta_r = X.relu(X.conv(x2, filter=256, name="delta_r_fc1",init=X.gauss(0.01)), name="delta_r_fc1_relu")  # [batch_roi, 256, 1, 1]
        delta_r = X.conv(delta_r, filter=2, name="delta_r_fc2", init=X.gauss(0.01)) # [batch_roi, 2, 1, 1]

        image_roi = self.p.image_roi if is_train else 1000
        batch_size = self.p.batch_size
        
        TSD_cls_feats = self.delta_c_pool.get_roi_feature(fpn_conv_feats, rois, delta_c, image_rois=image_roi, batch_size=batch_size) # [batch_roi, 256, 7, 7]
        TSD_loc_feats = self.delta_r_pool.get_roi_feature(fpn_conv_feats, rois, delta_r, image_rois=image_roi, batch_size=batch_size) # [batch_roi, 256, 7, 7]
        
        TSD_x_cls = self._convs_and_fcs(TSD_cls_feats, self.p.TSD.num_shared_convs, self.p.TSD.num_shared_fcs, name='TSD_pc', conv_init=xavier_init, fc_init=X.gauss(0.01)) # [batch_roi, batch_roi, 1, 1]
        TSD_x_reg = self._convs_and_fcs(TSD_loc_feats, self.p.TSD.num_shared_convs, self.p.TSD.num_shared_fcs, name='TSD_pr', conv_init=xavier_init, fc_init=X.gauss(0.01)) # [batch_roi, batch_roi, 1, 1]

        TSD_x_cls = self._convs_and_fcs(TSD_x_cls, 0, self.p.TSD.num_cls_fcs, name='TSD_cls', conv_init=xavier_init, fc_init=X.gauss(0.01))  # [batch_roi, batch_roi, 1, 1]
        TSD_x_reg = self._convs_and_fcs(TSD_x_reg, 0, self.p.TSD.num_reg_fcs, name='TSD_reg', conv_init=xavier_init, fc_init=X.gauss(0.01)) # [batch_roi, batch_roi, 1, 1]

        num_class = self.p.num_class
        num_reg_class = 2 if self.p.regress_target.class_agnostic else num_class

        tsd_cls_logit = X.fc(TSD_x_cls, filter=num_class, name='tsd_cls_logit', init=X.gauss(0.01))
        tsd_bbox_delta = X.fc(TSD_x_reg, filter=4 * num_reg_class, name='tsd_reg_delta', init=X.gauss(0.01))

        x = self._convs_and_fcs(roi_feat, self.p.TSD.num_shared_convs, self.p.TSD.num_shared_fcs, name='shared_fc', conv_init=xavier_init, fc_init=X.gauss(0.01))
        x_cls = x
        x_reg = x
        x_cls = self._convs_and_fcs(x_cls, 0, self.p.TSD.num_cls_fcs, name='cls', conv_init=xavier_init, fc_init=X.gauss(0.01))
        x_reg = self._convs_and_fcs(x_reg, 0, self.p.TSD.num_reg_fcs, name='reg', conv_init=xavier_init, fc_init=X.gauss(0.01))
        cls_logit = X.fc(x_cls, filter=num_class, name='bbox_cls_logit', init=X.gauss(0.01))
        bbox_delta = X.fc(x_reg, filter=4 * num_reg_class, name='bbox_reg_delta', init=X.gauss(0.01))

        if self.p.fp16:
            cls_logit = X.to_fp32(cls_logit, name="cls_logits_fp32")
            bbox_delta = X.to_fp32(bbox_delta, name="bbox_delta_fp32")
            tsd_cls_logit = X.to_fp32(tsd_cls_logit, name="tsd_cls_logit_fp32")
            tsd_bbox_delta = X.to_fp32(tsd_bbox_delta, name="tsd_bbox_delta_fp32")
            delta_c = X.to_fp32(delta_c, name="delta_c_fp32")
            delta_r = X.to_fp32(delta_r, name="delta_r_fp32")

        return cls_logit, bbox_delta, tsd_cls_logit, tsd_bbox_delta, delta_c, delta_r
            
    def _convs_and_fcs(self, x, num_convs, num_fcs, name, conv_init, fc_init):
        '''
        Args:
            x: [N, C, H, W] feature maps
            num_convs: int
            num_fcs: int
            conv_init: mx initializer
        Returns:
            x: [N, C, H, W] or [N, C, 1, 1]
        '''
        if num_convs == 0 and num_fcs == 0:
            return x

        out_channels = self.p.TSD.conv_out_channels
        out_fc_channels = self.p.TSD.fc_out_channels

        if num_convs > 0:
            for i in range(num_convs):
                x = X.relu(X.conv(x, kernel=3, filter=out_channels, no_bias=False, name=name+'_conv%s'%i, init=conv_init))
        
        if num_fcs > 0:
            x = X.reshape(x, shape=(0, -1, 1, 1), name=name+'_conv_fc_flatten')
            for i in range(num_fcs):
                x = X.relu(X.conv(x, kernel=1, filter=out_fc_channels, no_bias=False, name=name+'_fc%s'%i, init=fc_init))
        return x
    
    def _get_delta_r_box(self, delta_r, rois, scale=0.1):
        '''
        Args:
            delta_r: [batch_size * image_roi, 2, 1, 1]
            rois: [batch_size, image_roi, in_channels]
        Returns:
            rois_r: # [batch_size, image_roi, 4]
        '''
        batch_size = self.p.batch_size

        delta_r = mx.sym.reshape(delta_r, (batch_size, -1, 2)) # [batch_size, image_roi, 2]
        dx, dy = mx.sym.split(delta_r, axis=2, num_outputs=2, squeeze_axis=True) # both [batch_size, image_roi]
        rx1, ry1, rx2, ry2 = mx.sym.split(rois, axis=2, num_outputs=4, squeeze_axis=True) # all [batch_size, image_roi]
        w = rx2 - rx1
        h = ry2 - ry1
        rrx1 = rx1 + dx * scale * w
        rry1 = ry1 + dy * scale * h
        rrx2 = rx2 + dx * scale * w
        rry2 = ry2 + dy * scale * h
        rois_r = mx.sym.stack(rrx1, rry1, rrx2, rry2, axis=2, name='delta_r_roi') # [batch_size, image_roi, 4]
        return rois_r

    def get_loss(self, rois, roi_feat, fpn_conv_feats, cls_label, bbox_target, bbox_weight, gt_bbox):
        '''
        Args:
            rois: [batch_size, image_roi, 4]
            roi_feat: [batch_size * image_roi, 256, roi_size, roi_size]
            fpn_conv_feats: dict of FPN features, each [batch_size, in_channels, fh, fw]
            cls_label: [batch_size * image_roi]
            bbox_target: [batch_size * image_roi, num_class * 4]
            bbox_weight: [batch_size * image_roi, num_class * 4]
            gt_bbox: [batch_size, max_gt_num, 4]
        Returns:
            cls_loss: [batch_size * image_roi, num_class]
            reg_loss: [batch_size * image_roi, num_class * 4]
            tsd_cls_loss: [batch_size * image_roi, num_class]
            tsd_reg_loss: [batch_size * image_roi, num_class * 4]
            tsd_cls_pc_loss: [batch_size * image_roi]
            tsd_reg_pc_loss: [batch_size * image_roi]
            cls_label: [batch_size, image_roi]
        '''
        p = self.p
        assert not p.regress_target.class_agnostic
        batch_size = p.batch_size
        image_roi = p.image_roi
        batch_roi = batch_size * image_roi
        smooth_l1_scalar = p.regress_target.smooth_l1_scalar or 1.0
        
        cls_logit, bbox_delta, tsd_cls_logit, tsd_bbox_delta, delta_c, delta_r = self.get_output(fpn_conv_feats, roi_feat, rois, is_train=True)

        rois_r = self._get_delta_r_box(delta_r, rois)
        tsd_reg_target = self.get_reg_target(rois_r, gt_bbox) # [batch_roi, num_class*4]

        scale_loss_shift = 128 if self.p.fp16 else 1.0
        # origin loss
        cls_loss = X.softmax_output(
            data=cls_logit,
            label=cls_label,
            normalization='batch',
            grad_scale=1.0 * scale_loss_shift,
            name='bbox_cls_loss'
        )
        reg_loss = X.smooth_l1(
            bbox_delta - bbox_target,
            scalar=smooth_l1_scalar,
            name='bbox_reg_l1'
        )
        reg_loss = bbox_weight * reg_loss
        reg_loss = X.loss(
            reg_loss,
            grad_scale=1.0 / batch_roi * scale_loss_shift,
            name='bbox_reg_loss',
        )
        # tsd loss
        tsd_cls_loss = X.softmax_output(
            data=tsd_cls_logit,
            label=cls_label,
            normalization='batch',
            grad_scale=1.0 * scale_loss_shift,
            name='tsd_bbox_cls_loss'
        )
        tsd_reg_loss = X.smooth_l1(
            tsd_bbox_delta - tsd_reg_target,
            scalar=smooth_l1_scalar,
            name='tsd_bbox_reg_l1'
        )
        tsd_reg_loss = bbox_weight * tsd_reg_loss
        tsd_reg_loss = X.loss(
            tsd_reg_loss,
            grad_scale=1.0 / batch_roi * scale_loss_shift,
            name='tsd_bbox_reg_loss',
        )  
        tsd_cls_pc_loss = self.cls_pc_loss(cls_logit, tsd_cls_logit, cls_label, scale_loss_shift)
        tsd_reg_pc_loss = self.reg_pc_loss(bbox_delta, tsd_bbox_delta, rois, rois_r, gt_bbox, cls_label, scale_loss_shift)
        
        # append label
        cls_label = X.reshape(
            cls_label,
            shape=(batch_size, -1),
            name='bbox_label_reshape'
        )
        cls_label = X.block_grad(cls_label, name='bbox_label_blockgrad')

        return cls_loss, reg_loss, tsd_cls_loss, tsd_reg_loss, tsd_cls_pc_loss, tsd_reg_pc_loss, cls_label

    def get_prediction(self, rois, roi_feat, fpn_conv_feats, im_info, play=False):
        '''
        Args:
            rois: [batch_size, image_roi, 4]
            roi_feat: [batch_size * image_roi, 256, roi_size, roi_size]
            fpn_conv_feats: dict of FPN features, each [batch_size, in_channels, fh, fw]
            im_info: ...
        Returns:
            cls_score: [batch_size, image_roi, num_class]
            bbox_xyxy: [batch_size, image_roi, num_class*4]
        '''
        p = self.p
        assert not p.regress_target.class_agnostic

        bbox_mean = p.regress_target.mean
        bbox_std = p.regress_target.std
        batch_image = p.batch_image
        num_class = p.num_class
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = num_class

        assert batch_image == 1 

        cls_logit, bbox_delta, tsd_cls_logit, tsd_bbox_delta, delta_c, delta_r = self.get_output(fpn_conv_feats, roi_feat, rois, is_train=False)
        rois_r = self._get_delta_r_box(delta_r, rois)

        bbox_xyxy = X.decode_bbox(
            rois=rois_r,
            bbox_pred=X.reshape(tsd_bbox_delta, (batch_image, -1, 4 * num_reg_class)),
            im_info=im_info,
            name='decode_bbox',
            bbox_mean=bbox_mean,
            bbox_std=bbox_std,
            class_agnostic=False
        )
        cls_score = X.reshape(
            X.softmax(tsd_cls_logit, axis=-1, name='bbox_cls_score'),
            shape=(batch_image, -1, num_class),
            name='bbox_cls_score_reshape'
        )
      
        if not play:
            return cls_score, bbox_xyxy
        else:
            return cls_score, bbox_xyxy, rois, rois_r

    def get_reg_target(self, rois, gt_bbox):
        '''
        Args:
            rois: [batch_size, image_roi, 4]
            gt_bbox: [batch_size, max_gt_num, 4]
        Returns:
            reg_target: [batch_size * image_roi, num_class * 4]
        '''
        def get_transform(rois, gt_boxes):
            bbox_mean = self.p.regress_target.mean
            bbox_std = self.p.regress_target.std

            xmin1, ymin1, xmax1, ymax1 = mx.sym.split(rois, axis=-1, num_outputs=4, squeeze_axis=True)
            xmin2, ymin2, xmax2, ymax2, _ = mx.sym.split(gt_boxes, axis=-1, num_outputs=5, squeeze_axis=True)
            
            w1 = xmax1 - xmin1 + 1.0
            h1 = ymax1 - ymin1 + 1.0 
            x1 = xmin1 + 0.5 * (w1 - 1.0)
            y1 = ymin1 + 0.5 * (h1 - 1.0)

            w2 = xmax2 - xmin2 + 1.0
            h2 = ymax2 - ymin2 + 1.0 
            x2 = xmin2 + 0.5 * (w2 - 1.0)
            y2 = ymin2 + 0.5 * (h2 - 1.0)

            dx = (x2 - x1) / (w1 + 1e-14)
            dy = (y2 - y1) / (h1 + 1e-14)
            dw = mx.sym.log(w2 / w1)
            dh = mx.sym.log(h2 / h1)

            dx = (dx - bbox_mean[0]) / bbox_std[0]
            dy = (dy - bbox_mean[1]) / bbox_std[1]
            dw = (dw - bbox_mean[2]) / bbox_std[2]
            dh = (dh - bbox_mean[3]) / bbox_std[3]

            return mx.sym.stack(dx, dy, dw, dh, axis=1, name='delta_r_roi_transform')

        p_rpn = self.p.rpn_params
        batch_size = self.p.batch_size
        image_roi = self.p.image_roi #image_roi
        num_class = self.p.num_class # num_class
        max_num_gt = self.p.max_num_gt # 100

        reg_target = []
        
        rois_group = mx.sym.split(rois, axis=0, num_outputs=batch_size, squeeze_axis=True)
        gt_group = mx.sym.split(gt_bbox, axis=0, num_outputs=batch_size, squeeze_axis=True)

        for i , (rois_i, gt_box_i) in enumerate(zip(rois_group, gt_group)):
            iou_mat = get_iou_mat(rois_i, gt_box_i, l1=image_roi, l2=max_num_gt) # [image_roi, 100]
            idxs = mx.sym.argmax(iou_mat, axis=1) # [image_roi]
            match_gt_boxes = mx.sym.gather_nd(gt_box_i, X.reshape(idxs, [1, -1])) # [image_roi, 4]
            delta_i = get_transform(rois_i, match_gt_boxes) # [image_roi, 4]
            delta_i = mx.sym.reshape(mx.sym.repeat(delta_i, repeats=num_class, axis=0), (image_roi, -1)) #[image_roi, num_class * 4]
            reg_target.append(delta_i) 
        
        reg_target = X.block_grad(mx.sym.reshape(mx.sym.stack(*reg_target, axis=0), [batch_size*image_roi, -1], name='TSD_reg_target')) # [batch_roi, num_class*4]
        return reg_target

    def cls_pc_loss(self, logits, tsd_logits, gt_label, scale_loss_shift):
        '''
        TSD classification progressive constraint
        Args:
            logits: [batch_size * image_roi, num_class]
            tsd_logits: [batch_size * image_roi, num_class]
            gt_label:  [batch_size * image_roi]
            scale_loss_shift: float
        Returns:
            loss: [batch_size * image_roi]
        '''
        p = self.p
        batch_size = p.batch_size
        image_roi = p.image_roi
        batch_roi = batch_size * image_roi
        margin = self.p.TSD.pc_cls_margin

        cls_prob = mx.sym.SoftmaxActivation(logits, mode='instance')
        tsd_prob = mx.sym.SoftmaxActivation(tsd_logits, mode='instance')
        
        cls_score = mx.sym.pick(cls_prob, gt_label, axis=1)
        tsd_score = mx.sym.pick(tsd_prob, gt_label, axis=1)

        cls_score = X.block_grad(cls_score)
        cls_pc_margin = mx.sym.minimum(1. - cls_score, margin)
        loss = mx.sym.relu(-(tsd_score - cls_score - cls_pc_margin))

        grad_scale = 1. / batch_roi if self.p.TSD.pc_cls else 0.
        grad_scale *= scale_loss_shift
        loss = X.loss(loss, grad_scale=grad_scale, name='cls_pc_loss')
        return loss

    def reg_pc_loss(self, bbox_delta, tsd_bbox_delta, rois, tsd_rois, gt_bbox, gt_label, scale_loss_shift):
        '''
        TSD regression progressive constraint
        Args:
            bbox_delta: [batch_size * image_roi, num_class*4]
            tsd_bbox_delta: [batch_size * image_roi, num_class*4]
            rois: [batch_size, image_roi, 4]
            rois_r: [batch_size, image_roi, 4]
            gt_bbox: [batch_size, max_gt_num, 4]
            gt_label:  [batch_size * image_roi]
            scale_loss_shift: float
        Returns:
            loss: [batch_size * image_roi]
        '''
        def _box_decode(rois, deltas, means, stds):
            rois = X.block_grad(rois)
            rois = mx.sym.reshape(rois, [-1, 4])
            deltas = mx.sym.reshape(deltas, [-1, 4])

            x1, y1, x2, y2 = mx.sym.split(rois, axis=-1, num_outputs=4, squeeze_axis=True)
            dx, dy, dw, dh = mx.sym.split(deltas, axis=-1, num_outputs=4, squeeze_axis=True)
            
            dx = dx * stds[0] + means[0]
            dy = dy * stds[1] + means[1]
            dw = dw * stds[2] + means[2]
            dh = dh * stds[3] + means[3]

            x = (x1 + x2) * 0.5
            y = (y1 + y2) * 0.5
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            nx = x + dx * w
            ny = y + dy * h
            nw = w * mx.sym.exp(dw)
            nh = h * mx.sym.exp(dh)

            nx1 = nx - 0.5 * nw
            ny1 = ny - 0.5 * nh
            nx2 = nx + 0.5 * nw
            ny2 = ny + 0.5 * nh

            return mx.sym.stack(nx1, ny1, nx2, ny2, axis=1, name='pc_reg_loss_decoded_roi')  
        
        def _gather_3d(data, indices, n):
            datas = mx.sym.split(data, axis=-1, num_outputs=n, squeeze_axis=True)
            outputs = []
            for d in datas:
                outputs.append(mx.sym.pick(d, indices, axis=1))
            return mx.sym.stack(*outputs, axis=1)
        
        batch_size = self.p.batch_size
        image_roi = self.p.image_roi
        batch_roi = batch_size * image_roi
        num_class = self.p.num_class
        max_num_gt = self.p.max_num_gt
        bbox_mean = self.p.regress_target.mean
        bbox_std = self.p.regress_target.std
        margin = self.p.TSD.pc_reg_margin

        gt_label = mx.sym.reshape(gt_label, (-1,))

        bbox_delta = mx.sym.reshape(bbox_delta, (batch_size*image_roi, num_class, 4))
        tsd_bbox_delta = mx.sym.reshape(tsd_bbox_delta, (batch_size*image_roi, num_class, 4))

        bbox_delta = _gather_3d(bbox_delta, gt_label, n=4)
        tsd_bbox_delta = _gather_3d(tsd_bbox_delta, gt_label, n=4)

        boxes = _box_decode(rois, bbox_delta, bbox_mean, bbox_std)
        tsd_bboxes = _box_decode(tsd_rois, tsd_bbox_delta, bbox_mean, bbox_std)
        
        rois = mx.sym.reshape(rois, [batch_size, -1, 4])
        tsd_rois = mx.sym.reshape(tsd_rois, [batch_size, -1, 4])

        boxes = mx.sym.reshape(boxes, [batch_size, -1, 4])
        tsd_bboxes = mx.sym.reshape(tsd_bboxes, [batch_size, -1, 4])

        rois_group = mx.sym.split(rois, axis=0, num_outputs=batch_size, squeeze_axis=True)
        tsd_rois_group = mx.sym.split(tsd_rois, axis=0, num_outputs=batch_size, squeeze_axis=True)
        boxes_group = mx.sym.split(boxes, axis=0, num_outputs=batch_size, squeeze_axis=True)
        tsd_bboxes_group = mx.sym.split(tsd_bboxes, axis=0, num_outputs=batch_size, squeeze_axis=True)
        gt_group = mx.sym.split(gt_bbox, axis=0, num_outputs=batch_size, squeeze_axis=True)

        ious = []
        tsd_ious = []
        for i, (rois_i, tsd_rois_i, boxes_i, tsd_boxes_i, gt_i) in \
            enumerate(zip(rois_group, tsd_rois_group, boxes_group, tsd_bboxes_group, gt_group)):

            iou_mat = get_iou_mat(rois_i, gt_i, l1=image_roi, l2=max_num_gt)
            tsd_iou_mat = get_iou_mat(tsd_rois_i, gt_i, l1=image_roi, l2=max_num_gt)

            matched_gt = mx.sym.gather_nd(gt_i, X.reshape(mx.sym.argmax(iou_mat, axis=1), [1, -1]))
            tsd_matched_gt = mx.sym.gather_nd(gt_i, X.reshape(mx.sym.argmax(tsd_iou_mat, axis=1), [1, -1]))

            matched_gt = mx.sym.slice_axis(matched_gt, axis=-1, begin=0, end=4)
            tsd_matched_gt = mx.sym.slice_axis(tsd_matched_gt, axis=-1, begin=0, end=4)

            ious.append(get_iou(boxes_i, matched_gt))
            tsd_ious.append(get_iou(tsd_boxes_i, tsd_matched_gt))
        
        iou = mx.sym.concat(*ious, dim=0)
        tsd_iou = mx.sym.concat(*tsd_ious, dim=0)

        weight = X.block_grad(gt_label != 0)
        iou = X.block_grad(iou)

        reg_pc_margin = mx.sym.minimum(1. - iou, margin)
        loss = mx.sym.relu(-(tsd_iou - iou - reg_pc_margin))
        
        grad_scale = 1. / batch_roi if self.p.TSD.pc_reg else 0.
        grad_scale *= scale_loss_shift
        loss = X.loss(weight * loss, grad_scale=grad_scale, name='reg_pc_loss')
        return loss



