import mxnext as X
import mxnet as mx

from symbol.builder import FasterRcnn, RpnHead
from models.FPN.builder import FPNRpnHead

from models.maskrcnn import bbox_post_processing
from models.msrcnn import maskiou_compute

from utils.patch_config import patch_config_as_nothrow
from utils.deprecated import deprecated

class MaskScoringFasterRcnn(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, roi_extractor, mask_roi_extractor, bbox_head, mask_head, maskiou_head):
        gt_bbox = X.var("gt_bbox")
        gt_poly = X.var("gt_poly")
        im_info = X.var("im_info")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_head.get_anchor()
        rpn_loss = rpn_head.get_loss(rpn_feat, gt_bbox, im_info)
        proposal, bbox_cls, bbox_target, bbox_weight, mask_proposal, mask_target, mask_ind, mask_ratio = \
            rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, gt_poly, im_info)

        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, mask_proposal)

        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)
        mask_loss, mask_pred_logits = mask_head.get_loss(mask_roi_feat, mask_target, mask_ind)

        iou_loss = maskiou_head.get_loss(mask_roi_feat, mask_pred_logits, mask_target, mask_ind, mask_ratio)
        return X.group(rpn_loss + bbox_loss + mask_loss + iou_loss)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head, roi_extractor, mask_roi_extractor, bbox_head, mask_head, maskiou_head, bbox_post_processor):
        rec_id, im_id, im_info, proposal, proposal_score = \
            MaskScoringFasterRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        post_cls_score, post_bbox_xyxy, post_cls = bbox_post_processor.get_post_processing(cls_score, bbox_xyxy)

        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, post_bbox_xyxy)
        mask = mask_head.get_prediction(mask_roi_feat)
        mask_score = maskiou_head.get_maskiou_prediction(mask, mask_roi_feat, post_cls, post_cls_score)

        return X.group([rec_id, im_id, im_info, post_cls_score, post_bbox_xyxy, post_cls, mask, mask_score])

    @staticmethod
    def get_rpn_test_symbol(backbone, neck, rpn_head):
        return FasterRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

class MaskIoUConvHead():
    def __init__(self, pTest, pBbox, pMask):
        self.pBbox = pBbox
        self.pMask = pMask
        self.pTest = pTest

    def get_maskiou_prediction(self, mask_pred_logits, conv_feat, post_cls, post_cls_score):
        # align cls index for MaskIoU prediction
        post_cls = mx.sym.reshape(post_cls, (-1, 1)) + 1
        post_cls_score = mx.sym.reshape(post_cls_score, (self.pBbox.batch_image, -1, 1))

        det_ind = mx.sym.arange(self.pTest.max_det_per_image)
        mask_inds = mx.sym.stack(det_ind, post_cls)

        mask_pred_logits = mx.sym.gather_nd(mask_pred_logits, mask_inds, axis=1)
        mask_pred_logits = self._get_output(mask_pred_logits, conv_feat)

        maskiou_pred = mx.sym.reshape(mask_pred_logits, (self.pBbox.batch_image, self.pTest.max_det_per_image, self.pBbox.num_class))

        batch_ind = mx.sym.arange(self.pBbox.batch_image)
        batch_ind = mx.sym.repeat(batch_ind, self.pTest.max_det_per_image)
        det_ind =mx.sym.tile(det_ind, self.pBbox.batch_image)
        post_cls_ind = mx.sym.tile(post_cls, self.pBbox.batch_image)
        maskiou_ind = mx.sym.stack(batch_ind, det_ind, post_cls_ind)

        maskiou_pred = mx.sym.gather_nd(maskiou_pred, maskiou_ind, axis=2)
        maskiou_pred = mx.sym.reshape(maskiou_pred, (self.pBbox.batch_image, self.pTest.max_det_per_image, 1))

        # align mask score by multiplying with classification score
        maskiou_pred = mx.sym.broadcast_mul(maskiou_pred, post_cls_score, name='maskiou_prediction')

        return maskiou_pred

    def _get_output(self, mask_pred_logits, conv_feat):
        num_class = self.pBbox.num_class

        msra_init = mx.init.Xavier(rnd_type="gaussian", factor_type="out", magnitude=2)
        normal_init = mx.init.Normal(0.01)
        kaiming_uniform = mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=3)

        mask_pred_logits = mx.sym.expand_dims(mask_pred_logits, axis=1)

        iou_head_maxpool_1 = X.pool(
            mask_pred_logits,
            name='iou_head_maxpool_1',
            kernel=2,
            stride=2,
            pad=0,
        )
        iou_head_input = X.concat([conv_feat, iou_head_maxpool_1], axis=1, name='iou_head_input')
        hi = iou_head_input
        for ii in range(3):
            hi = X.conv(
                hi,
                filter=256,
                kernel=3,
                stride=1,
                name='iou_head_conv_%d'%ii,
                no_bias=False,
                init=msra_init,
            )
            hi = X.relu(hi)
        hi = X.conv(
            hi,
            filter=256,
            kernel=3,
            stride=2,
            name='iou_head_conv_3',
            no_bias=False,
            init=msra_init
        )
        hi = X.relu(hi)
        hi = X.flatten(data=hi)
        fc1 = X.relu(X.fc(hi, filter=1024, name='iou_head_FC1', init=kaiming_uniform))
        fc2 = X.relu(X.fc(fc1, filter=1024, name='iou_head_FC2', init=kaiming_uniform))
        iou_pred_logits = X.fc(fc2, filter=num_class, name='iou_head_pred', init=normal_init)
        return iou_pred_logits

    def get_loss(self, conv_feat, mask_pred_logits, mask_target, mask_inds, mask_ratio):
        iou_pred_logits = self._get_output(mask_pred_logits, conv_feat)

        iou_pred_logits = mx.sym.split(iou_pred_logits, num_outputs=self.pBbox.batch_image, axis=0)
        mask_inds_split = mx.sym.split(mask_inds, num_outputs=self.pBbox.batch_image, axis=0, squeeze_axis=True)
        iou_pred_list = []
        for each_iou_pred, each_mask_inds in zip(iou_pred_logits, mask_inds_split):
            batch_ind = mx.sym.arange(self.pMask.num_fg_roi)
            each_mask_inds = mx.sym.stack(batch_ind, each_mask_inds)
            each_iou_pred = mx.sym.gather_nd(each_iou_pred, each_mask_inds, axis=1)
            iou_pred_list.append(each_iou_pred)
        iou_pred_logits = mx.sym.concat(*iou_pred_list, dim=0)
        iou_pred_logits = mx.sym.reshape(iou_pred_logits, (-1, 1))

        maskiou_target, weight_list = mx.sym.Custom(
            mask_pred_logits=mask_pred_logits,
            mask_target=mask_target,
            mask_ratio=mask_ratio,
            mask_inds=mask_inds,
            op_type='maskiou_compute'
        )
        maskiou_target = mx.sym.BlockGrad(maskiou_target, name='gtIoU_blockGrad')
        weight_list = mx.sym.BlockGrad(weight_list, name='weight_blockGrad')
        maskiou_head_loss = mx.sym.MakeLoss(0.5 * mx.sym.sum(((maskiou_target - iou_pred_logits) ** 2) * weight_list) /
                                                mx.sym.maximum(mx.sym.sum(weight_list), 1.), name='iou_head_loss')
        return (maskiou_head_loss, )

class BboxPostProcessor(object):
    def __init__(self, pTest):
        self.p = patch_config_as_nothrow(pTest)

    def get_post_processing(self, cls_score, bbox_xyxy):
        p = self.p
        max_det_per_image = p.max_det_per_image
        min_det_score = p.min_det_score
        nms_type = p.nms.type
        nms_thr = p.nms.thr

        post_cls_score, post_bbox_xyxy, post_cls = mx.sym.Custom(
            cls_score=cls_score,
            bbox_xyxy=bbox_xyxy,
            max_det_per_image = max_det_per_image,
            min_det_score = min_det_score,
            nms_type = nms_type,
            nms_thr = nms_thr,
            op_type='BboxPostProcessing')
        return post_cls_score, post_bbox_xyxy, post_cls


class MaskRpnHead(RpnHead):
    def __init__(self, pRpn, pMask):
        super().__init__(pRpn)
        self.pMask = patch_config_as_nothrow(pMask)

    def get_sampled_proposal(self, conv_fpn_feat, gt_bbox, gt_poly, im_info):
        p = self.p

        batch_image = p.batch_image

        proposal_wo_gt = p.subsample_proposal.proposal_wo_gt
        image_roi = p.subsample_proposal.image_roi
        fg_fraction = p.subsample_proposal.fg_fraction
        fg_thr = p.subsample_proposal.fg_thr
        bg_thr_hi = p.subsample_proposal.bg_thr_hi
        bg_thr_lo = p.subsample_proposal.bg_thr_lo

        num_reg_class = p.bbox_target.num_reg_class
        class_agnostic = p.bbox_target.class_agnostic
        bbox_target_weight = p.bbox_target.weight
        bbox_target_mean = p.bbox_target.mean
        bbox_target_std = p.bbox_target.std

        mask_size = self.pMask.resolution

        (proposal, proposal_score) = self.get_all_proposal(conv_fpn_feat, im_info)

        (bbox, label, bbox_target, bbox_weight, match_gt_iou, mask_target, mask_ratio) = mx.sym.ProposalMaskTarget(
            proposal,
            gt_bbox,
            gt_poly,
            mask_size=mask_size,
            num_classes=num_reg_class,
            class_agnostic=class_agnostic,
            batch_images=batch_image,
            proposal_without_gt=proposal_wo_gt,
            image_rois=image_roi,
            fg_fraction=fg_fraction,
            fg_thresh=fg_thr,
            bg_thresh_hi=bg_thr_hi,
            bg_thresh_lo=bg_thr_lo,
            bbox_weight=bbox_target_weight,
            bbox_mean=bbox_target_mean,
            bbox_std=bbox_target_std,
            output_iou=True,
            output_ratio=True,
            name="subsample_proposal"
        )

        num_fg_rois_per_img = int(image_roi * fg_fraction)
        mask_label = mx.sym.slice_axis(
            label,
            axis=1,
            begin=0,
            end=num_fg_rois_per_img
        )
        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))
        mask_target = X.reshape(mask_target, (-3, -2))
        mask_proposal = mx.sym.slice_axis(
            bbox,
            axis=1,
            begin=0,
            end=num_fg_rois_per_img)

        return bbox, label, bbox_target, bbox_weight, mask_proposal, mask_target, mask_label, mask_ratio

class MaskFPNRpnHead(FPNRpnHead):
    def __init__(self, pRpn, pMask):
        super().__init__(pRpn)
        self.pMask = patch_config_as_nothrow(pMask)

    def get_sampled_proposal(self, conv_fpn_feat, gt_bbox, gt_poly, im_info):
        p = self.p

        batch_image = p.batch_image

        proposal_wo_gt = p.subsample_proposal.proposal_wo_gt
        image_roi = p.subsample_proposal.image_roi
        fg_fraction = p.subsample_proposal.fg_fraction
        fg_thr = p.subsample_proposal.fg_thr
        bg_thr_hi = p.subsample_proposal.bg_thr_hi
        bg_thr_lo = p.subsample_proposal.bg_thr_lo
        post_nms_top_n = p.proposal.post_nms_top_n

        num_reg_class = p.bbox_target.num_reg_class
        class_agnostic = p.bbox_target.class_agnostic
        bbox_target_weight = p.bbox_target.weight
        bbox_target_mean = p.bbox_target.mean
        bbox_target_std = p.bbox_target.std

        mask_size = self.pMask.resolution

        (proposal, proposal_score) = self.get_all_proposal(conv_fpn_feat, im_info)

        (bbox, label, bbox_target, bbox_weight, match_gt_iou, mask_target, mask_ratio) = mx.sym.ProposalMaskTarget(
            rois=proposal,
            gt_boxes=gt_bbox,
            gt_polys=gt_poly,
            mask_size=mask_size,
            num_classes=num_reg_class,
            class_agnostic=class_agnostic,
            batch_images=batch_image,
            proposal_without_gt=proposal_wo_gt,
            image_rois=image_roi,
            fg_fraction=fg_fraction,
            fg_thresh=fg_thr,
            bg_thresh_hi=bg_thr_hi,
            bg_thresh_lo=bg_thr_lo,
            bbox_weight=bbox_target_weight,
            bbox_mean=bbox_target_mean,
            bbox_std=bbox_target_std,
            output_iou=True,
            output_ratio=True,
            name="subsample_proposal"
        )

        num_fg_rois_per_img = int(image_roi * fg_fraction)
        mask_label = mx.sym.slice_axis(
            label,
            axis=1,
            begin=0,
            end=num_fg_rois_per_img
        )
        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))
        mask_target = X.reshape(mask_target, (-3, -2))
        mask_proposal = mx.sym.slice_axis(
            bbox,
            axis=1,
            begin=0,
            end=num_fg_rois_per_img)

        return bbox, label, bbox_target, bbox_weight, mask_proposal, mask_target, mask_label, mask_ratio


class MaskFasterRcnnHead(object):
    def __init__(self, pBbox, pMask, pMaskRoi):
        self.pBbox = patch_config_as_nothrow(pBbox)
        self.pMask = patch_config_as_nothrow(pMask)
        self.pMaskRoi = patch_config_as_nothrow(pMaskRoi)

        self._head_feat = None

    def add_norm(self, sym):
        p = self.pMask
        if p.normalizer.__name__ == "fix_bn":
            pass
        elif p.normalizer.__name__ in ["sync_bn", "local_bn", "gn", "dummy"]:
            sym = p.normalizer(sym)
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))
        return sym

    def _get_mask_head_logit(self, conv_feat):
        raise NotImplementedError

    def get_output(self, conv_feat):
        pBbox = self.pBbox
        num_class = pBbox.num_class

        head_feat = self._get_mask_head_logit(conv_feat)

        msra_init = mx.init.Xavier(rnd_type="gaussian", factor_type="out", magnitude=2)

        if self.pMask:
            head_feat = X.to_fp32(head_feat, name="mask_head_to_fp32")

        mask_fcn_logit = X.conv(
            head_feat,
            filter=num_class,
            name="mask_fcn_logit",
            no_bias=False,
            init=msra_init
        )

        return mask_fcn_logit

    def get_prediction(self, conv_feat):
        """
        input: conv_feat, (1 * num_box, channel, pool_size, pool_size)
        """
        mask_fcn_logit = self.get_output(conv_feat)
        mask_prob = mx.symbol.Activation(
            data=mask_fcn_logit,
            act_type='sigmoid',
            name="mask_prob")
        return mask_prob

    def get_loss(self, conv_feat, mask_target, mask_ind):
        pBbox = self.pBbox
        pMask = self.pMask
        batch_image = pBbox.batch_image

        mask_fcn_logit = self.get_output(conv_feat)

        scale_loss_shift = 128.0 if pMask.fp16 else 1.0

        mask_fcn_logits = mx.sym.split(mask_fcn_logit, num_outputs=batch_image, axis=0)
        mask_inds = mx.sym.split(mask_ind, num_outputs=batch_image, axis=0, squeeze_axis=True)
        mask_fcn_logit_list = []
        for mask_fcn_logit, mask_ind in zip(mask_fcn_logits, mask_inds):
            batch_ind = mx.sym.arange(pMask.num_fg_roi)
            mask_ind = mx.sym.stack(batch_ind, mask_ind)
            mask_fcn_logit = mx.sym.gather_nd(mask_fcn_logit, mask_ind, axis=1)
            mask_fcn_logit_list.append(mask_fcn_logit)
        mask_fcn_logit = mx.sym.concat(*mask_fcn_logit_list, dim=0)

        # get mask prediction logits
        mask_pred_logits = mx.symbol.Activation(
            data=mask_fcn_logit,
            act_type='sigmoid',
            name='mask_pred_prob')

        mask_fcn_logit = X.reshape(
            mask_fcn_logit,
            shape=(1, -1),
            name="mask_fcn_logit_reshape"
        )
        mask_target = X.reshape(
            mask_target,
            shape=(1, -1),
            name="mask_target_reshape"
        )
        mask_loss = mx.sym.contrib.SigmoidCrossEntropy(
            mask_fcn_logit,
            mask_target,
            grad_scale=1.0 * scale_loss_shift,
            name="mask_loss"
        )
        return (mask_loss,), mask_pred_logits


class MaskFasterRcnn4ConvHead(MaskFasterRcnnHead):
    def __init__(self, pBbox, pMask, pMaskRoi):
        super().__init__(pBbox, pMask, pMaskRoi)

    def _get_mask_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        up_stride = int(self.pMask.resolution // self.pMaskRoi.out_size)
        dim_reduced = self.pMask.dim_reduced

        msra_init = mx.init.Xavier(rnd_type="gaussian", factor_type="out", magnitude=2)

        current = conv_feat
        for i in range(4):
            current = X.conv(
                current,
                name="mask_fcn_conv{}".format(i + 1),
                filter=dim_reduced,
                kernel=3,
                no_bias=False,
                init=msra_init
            )
            current = self.add_norm(current)
            current = X.relu(current)

        mask_up = current
        for i in range(up_stride // 2):
            weight = X.var(
                name="mask_up{}_weight".format(i),
                init=msra_init,
                lr_mult=1,
                wd_mult=1)
            mask_up = mx.sym.Deconvolution(
                mask_up,
                kernel=(2, 2),
                stride=(2, 2),
                num_filter=dim_reduced,
                no_bias=False,
                weight=weight,
                name="mask_up{}".format(i)
                )
            mask_up = X.relu(
                mask_up,
                name="mask_up{}_relu".format(i))

        mask_up = X.to_fp32(mask_up, name='mask_up_to_fp32')
        self._head_feat = mask_up

        return self._head_feat

