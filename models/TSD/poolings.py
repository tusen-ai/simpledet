from __future__ import print_function

import mxnet as mx
import numpy as np
import mxnext as X
from utils.patch_config import patch_config_as_nothrow
from utils.deprecated import deprecated

from symbol.builder import RoiAlign, RoiExtractor


def fpn_roi_assign(
    F=mx.ndarray, 
    rois=None, 
    rcnn_stride=None,
    roi_canonical_scale=None, 
    roi_canonical_level=None
):
    ############ constant #############
    scale0 = roi_canonical_scale
    lvl0 = roi_canonical_level
    k_min = np.log2(min(rcnn_stride))
    k_max = np.log2(max(rcnn_stride))

    with mx.name.Prefix("fpn_roi_assgin: "):
        x1, y1, x2, y2 = F.split(rois, num_outputs=4, axis=-1)
        rois_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        rois_scale = F.sqrt(rois_area)
        target_lvls = F.floor(lvl0 + F.log2(rois_scale / scale0 + 1e-6))
        target_lvls = F.clip(target_lvls, k_min, k_max)
        target_stride = F.pow(2, target_lvls).astype('uint8')

        outputs = []
        for i, s in enumerate(rcnn_stride):
            lvl_rois = F.zeros_like(rois)
            lvl_inds = (target_stride == s).astype('float32')
            lvl_inds = F.broadcast_like(lvl_inds, lvl_rois)
            lvl_rois = F.where(lvl_inds, rois, lvl_rois)
            outputs.append(lvl_rois)

    return outputs

def fpn_roi_assign_offset(
    F=mx.ndarray, 
    rois=None, 
    offset=None,
    rcnn_stride=None,
    roi_canonical_scale=None, 
    roi_canonical_level=None
):
    ############ constant #############
    scale0 = roi_canonical_scale
    lvl0 = roi_canonical_level
    k_min = np.log2(min(rcnn_stride))
    k_max = np.log2(max(rcnn_stride))

    with mx.name.Prefix("fpn_roi_assign_offset: "):
        x1, y1, x2, y2 = F.split(rois, num_outputs=4, axis=-1)
        rois_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        rois_scale = F.sqrt(rois_area)
        target_lvls = F.floor(lvl0 + F.log2(rois_scale / scale0 + 1e-6))
        target_lvls = F.clip(target_lvls, k_min, k_max)
        target_stride = F.pow(2, target_lvls).astype('uint8')

        out_rois = []
        out_offsets = []
        for i, s in enumerate(rcnn_stride):
            lvl_rois = F.ones_like(rois) * -1
            lvl_offset = F.zeros_like(offset) 
            lvl_inds = (target_stride == s).astype('float32') # [2, 512]
            lvl_inds_roi = F.broadcast_like(lvl_inds, lvl_rois) # [2, 512, 4]
            lvl_inds_offset = F.broadcast_like(lvl_inds, lvl_offset)
            lvl_rois = F.where(lvl_inds_roi, rois, lvl_rois)
            lvl_offset = F.where(lvl_inds_offset, offset, lvl_offset)
            out_rois.append(lvl_rois)
            out_offsets.append(lvl_offset)
            
    return out_rois, out_offsets


class FPNRoiAlign(RoiAlign):
    def __init__(self, pRoi):
        super().__init__(pRoi)

    def get_roi_feature(self, conv_fpn_feat, proposal):
        p = self.p
        rcnn_stride = p.stride
        roi_canonical_scale = p.roi_canonical_scale
        roi_canonical_level = p.roi_canonical_level

        group = fpn_roi_assign(mx.symbol, proposal, rcnn_stride, roi_canonical_scale, roi_canonical_level)

        proposal_fpn = dict()
        for i, stride in enumerate(rcnn_stride):
            proposal_fpn["stride%s" % stride] = group[i]

        if p.fp16:
            for stride in rcnn_stride:
                conv_fpn_feat["stride%s" % stride] = X.to_fp32(
                    conv_fpn_feat["stride%s" % stride],
                    name="fpn_stride%s_to_fp32"
                )

        fpn_roi_feats = list()
        for stride in rcnn_stride:
            feat_lvl = conv_fpn_feat["stride%s" % stride]
            proposal_lvl = proposal_fpn["stride%s" % stride]
            roi_feat = X.roi_align(
                feat_lvl,
                rois=proposal_lvl,
                out_size=p.out_size,
                stride=stride,
                name="roi_align"
            )
            roi_feat = X.reshape(
                data=roi_feat,
                shape=(-3, -2),
                name='roi_feat_reshape'
            )
            fpn_roi_feats.append(roi_feat)
        roi_feat = X.add_n(*fpn_roi_feats)

        if p.fp16:
            roi_feat = X.to_fp16(roi_feat, name="roi_feat_to_fp16")

        return roi_feat


class FPNRoIAlign_DeltaC(RoiExtractor):
    def __init__(self, pRoi):
        super().__init__(pRoi)

    def get_roi_feature(self, conv_fpn_feat, rois, trans, image_rois, batch_size):
        '''
        conv_fpn_feat: list of feature maps
        rois: [2, 512, 4]
        trans: [1024, 2*7*7, 1, 1]
        '''
        p = self.p
        rcnn_stride = p.stride
        roi_canonical_scale = p.roi_canonical_scale
        roi_canonical_level = p.roi_canonical_level


        _trans = mx.sym.reshape(trans, [-1, image_rois, 2*p.out_size**2])
        roi_group, offset_group = fpn_roi_assign_offset(mx.symbol, rois, _trans, rcnn_stride, roi_canonical_scale, roi_canonical_level)

        rois_fpn = dict()
        offset_fpn = dict()
        for i, stride in enumerate(rcnn_stride):
            rois_fpn["stride%s" % stride] = mx.sym.reshape(roi_group[i], [-1, 4])
            offset_fpn["stride%s" % stride] = mx.sym.reshape(offset_group[i], [-1, 2, p.out_size, p.out_size])

        if p.fp16:
            for stride in rcnn_stride:
                conv_fpn_feat["stride%s" % stride] = X.to_fp32(
                    conv_fpn_feat["stride%s" % stride],
                    name="fpn_stride%s_to_fp32"
                )

        batch_pad = mx.sym.reshape(mx.sym.repeat(mx.sym.arange(batch_size), repeats=image_rois, axis=0), [-1,1])

        fpn_roi_feats = list()
        for stride in rcnn_stride:
            feat_lvl = conv_fpn_feat["stride%s" % stride]
            rois_lvl = rois_fpn["stride%s" % stride]  # [1024, 4]
            offset_lvl = offset_fpn["stride%s" % stride] # [1024, 2, 7, 7]
            rois_lvl = mx.sym.concat(batch_pad, rois_lvl, dim=1)

            roi_feat = mx.sym.contrib.DeformablePSROIPooling(
                data=feat_lvl,
                rois=rois_lvl,
                trans=offset_lvl,
                spatial_scale=1./stride,
                output_dim=256,
                group_size=1,
                pooled_size=p.out_size,
                part_size=0,
                sample_per_part=4,
                trans_std=0.1,
                no_trans=False,
                name='delta_c_pooled_feat'
            ) # [1024, 256, 7, 7]

            # roi_feat = X.roi_align(
            #     feat_lvl,
            #     rois=mx.sym.slice_axis(mx.sym.reshape(rois_lvl, (batch_size, image_rois, 5)), axis=2, begin=1, end=5),
            #     out_size=p.out_size,
            #     stride=stride,
            #     name="delta_c_pooled_feat"
            # )
            # roi_feat = mx.sym.reshape(roi_feat, (batch_size*image_rois, -1, p.out_size, p.out_size))

            fpn_roi_feats.append(roi_feat)
        roi_feat = X.add_n(*fpn_roi_feats)

        if p.fp16:
            roi_feat = X.to_fp16(roi_feat, name="delta_c_roi_feat_to_fp16")

        return roi_feat
    
    def get_roi_feature_test(self, conv_fpn_feat, rois, trans, image_rois, batch_size=1):
        assert batch_size == 1
        return self.get_roi_feature(conv_fpn_feat, rois, trans, image_rois, batch_size)


class FPNRoIAlign_DeltaR(RoiExtractor):
    def __init__(self, pRoi):
        super().__init__(pRoi)

    def get_roi_feature(self, conv_fpn_feat, rois, trans, image_rois, batch_size):
        '''
        conv_fpn_feat: list of feature maps
        rois: [2, 512, 4]
        trans: [1024, 2*7*7, 1, 1]
        '''
        p = self.p
        rcnn_stride = p.stride
        roi_canonical_scale = p.roi_canonical_scale
        roi_canonical_level = p.roi_canonical_level

        _trans = mx.sym.reshape(trans, [-1, image_rois, 2])
        roi_group, offset_group = fpn_roi_assign_offset(mx.symbol, rois, _trans, rcnn_stride, roi_canonical_scale, roi_canonical_level)

        rois_fpn = dict()
        offset_fpn = dict()
        for i, stride in enumerate(rcnn_stride):
            rois_fpn["stride%s" % stride] = mx.sym.reshape(roi_group[i], [-1, 4])
            offset_fpn["stride%s" % stride] = mx.sym.tile(
                mx.sym.reshape(offset_group[i], [-1, 2, 1, 1]),
                (1, 1, p.out_size, p.out_size)
            )

        if p.fp16:
            for stride in rcnn_stride:
                conv_fpn_feat["stride%s" % stride] = X.to_fp32(
                    conv_fpn_feat["stride%s" % stride],
                    name="fpn_stride%s_to_fp32"
                )

        batch_pad = mx.sym.reshape(mx.sym.repeat(mx.sym.arange(batch_size), repeats=image_rois, axis=0), [-1,1])

        fpn_roi_feats = list()
        for stride in rcnn_stride:
            feat_lvl = conv_fpn_feat["stride%s" % stride]
            rois_lvl = rois_fpn["stride%s" % stride]  # [1024, 4]
            offset_lvl = offset_fpn["stride%s" % stride] # [1024, 2, 7, 7]
            rois_lvl = mx.sym.concat(batch_pad, rois_lvl, dim=1)

            roi_feat = mx.sym.contrib.DeformablePSROIPooling(
                data=feat_lvl,
                rois=rois_lvl,
                trans=offset_lvl,
                spatial_scale=1./stride,
                output_dim=256,
                group_size=1,
                pooled_size=p.out_size,
                part_size=0,
                sample_per_part=4,
                trans_std=0.1,
                no_trans=False,
                name='delta_r_pooled_feat_stride%s'%stride 
            ) # [1024, 256, 7, 7]

            # roi_feat = X.roi_align(
            #     feat_lvl,
            #     rois=mx.sym.slice_axis(mx.sym.reshape(rois_lvl, (batch_size, image_rois, 5)), axis=2, begin=1, end=5),
            #     out_size=p.out_size,
            #     stride=stride,
            #     name="delta_c_pooled_feat"
            # )
            # roi_feat = mx.sym.reshape(roi_feat, (batch_size*image_rois, -1, p.out_size, p.out_size))

            fpn_roi_feats.append(roi_feat)
        roi_feat = X.add_n(*fpn_roi_feats)

        if p.fp16:
            roi_feat = X.to_fp16(roi_feat, name="delta_r_roi_feat_to_fp16")

        return roi_feat
    
    def get_roi_feature_test(self, conv_fpn_feat, rois, trans, image_rois, batch_size=1):
        assert batch_size == 1
        return self.get_roi_feature(conv_fpn_feat, rois, trans, image_rois, batch_size)