/*!
 * Copyright (c) 2017 by TuSimple
 * \file proposal_mask_target.cc
 * \brief C++ version proposal target
 * \author Yuntao Chen, Zehao Huang, Ziyang Zhou
 */
#include <algorithm>
#include <cstdio>
#include "./proposal_mask_target-inl.h"
#include "../coco_api/common/maskApi.h"
using std::min;
using std::max;
using std::vector;
using std::begin;
using std::end;
using std::random_shuffle;
using std::log;

template <typename DType>
inline double convertPoly2MaskWithRatio(const DType *roi,
                             const DType *poly,
                             const int mask_size,
                             DType *mask){
      DType w = roi[2] - roi[0];
      DType h = roi[3] - roi[1];

      w = max((DType)1., w);
      h = max((DType)1., h);
      int category = static_cast<int>(poly[0]);
      int n_seg = static_cast<int>(poly[1]);

      RLE* rles;
      RLE* rles_origin;
      RLE* rles_crop;

      rlesInit(&rles, n_seg);
      rlesInit(&rles_origin, n_seg);
      rlesInit(&rles_crop, n_seg);

      int roi_x_1 = roi[0], roi_x_2 = roi[2], roi_y_1 = roi[1], roi_y_2 = roi[3];

      int crop_w = roi_x_2 - roi_x_1 + 1;
      int crop_h = roi_y_2 - roi_y_1 + 1;

      int offset = 2 + n_seg;
      double origin_x_1=roi[0], origin_x_2=roi[2], origin_y_1=roi[1], origin_y_2=roi[3];
      for(int i = 0; i < n_seg; i++){
        int cur_len = poly[i+2];
        double* xys = new double[cur_len];
        double* xys_crop = new double[cur_len];
        for(int j = 0; j < cur_len; j++){
          if (j % 2 == 0){
            double poly_index = poly[offset+j+1];
            origin_y_1 = min(origin_y_1, poly_index);
            origin_y_2 = max(origin_y_2, poly_index);
            xys[j] = (poly_index - roi[1]) * mask_size / h;
            xys_crop[j+1] = poly_index - roi[1];
          }
          else{
            double poly_index = poly[offset+j-1];
            origin_x_1 = min(origin_x_1, poly_index);
            origin_x_2 = max(origin_x_2, poly_index);
            xys[j] = (poly_index - roi[0]) * mask_size / w;
            xys_crop[j-1] = poly_index - roi[0];
          }
        }
        rleFrPoly(rles + i, xys, cur_len/2, mask_size, mask_size);
        rleFrPoly(rles_crop + i, xys_crop, cur_len/2, crop_h, crop_w);
        delete [] xys;
        delete [] xys_crop;
        offset += cur_len;
      }

      int int_x_1 = int(origin_x_1), int_x_2 = int(origin_x_2), int_y_1 = int(origin_y_1), int_y_2 = int(origin_y_2);

      double origin_mask_sum = 0;
      double crop_mask_sum = 0;

      int full_w = int_x_2 - int_x_1 + 1;
      int full_h = int_y_2 - int_y_1 + 1;

      offset = 2 + n_seg;
      for(int i=0; i<n_seg; i++){
        int cur_len = poly[i+2];
        double* xys_origin = new double[cur_len];
        for(int j=0; j<cur_len; j++){
            if (j % 2 == 0){
                double poly_index = poly[offset+j+1];
                xys_origin[j+1] = poly_index - origin_y_1;
            }
            else{
                double poly_index = poly[offset+j-1];
                xys_origin[j-1] = poly_index - origin_x_1;
            }
        }
        rleFrPoly(rles_origin+i, xys_origin, cur_len/2, full_h, full_w);
        delete xys_origin;
        offset += cur_len;
      }

      // Decode RLE to mask
      byte* byte_mask = new byte[mask_size*mask_size*n_seg];
      byte* byte_mask_origin = new byte[(int_y_2-int_y_1+1) * (int_x_2-int_x_1+1) *n_seg];
      byte* byte_mask_crop = new byte[(roi_x_2-roi_x_1+1) * (roi_y_2-roi_y_1+1) * n_seg];

      rleDecode(rles, byte_mask, n_seg);
      rleDecode(rles_origin, byte_mask_origin, n_seg);
      rleDecode(rles_crop, byte_mask_crop, n_seg);

      DType* mask_cat = mask;
      // Flatten mask
      for(int j = 0; j < mask_size * mask_size; j++){
        float cur_byte = 0;
        for(int i = 0; i< n_seg; i++){
          int offset = i * mask_size * mask_size + j;
          if(byte_mask[offset]==1){
            cur_byte = 1;
            break;
          }
        }
        mask_cat[j] = cur_byte;
      }

      for(int i=0; i<full_w * full_h; i++){
        for(int p = 0; p<n_seg; p++){
            int offset = p * full_w * full_h + i;
            if(byte_mask_origin[offset] == 1){
                origin_mask_sum += 1;
                break;
            }
        }
      }
      for(int i=0; i<crop_w*crop_h; i++){
        for(int p=0; p<n_seg; p++){
            int offset = p * crop_w * crop_h + i;
            if(byte_mask_crop[offset] == 1){
                crop_mask_sum += 1;
                break;
            }
        }
      }
      double mask_ratio = crop_mask_sum / (origin_mask_sum + 0.0001);
      mask_ratio = max(mask_ratio, 1e-10);
      // Check to make sure we don't have memory leak
      rlesFree(&rles, n_seg);
      rlesFree(&rles_origin, n_seg);
      rlesFree(&rles_crop, n_seg);
      delete [] byte_mask_origin;
      delete [] byte_mask;
      delete [] byte_mask_crop;
      return mask_ratio;
}//convertPoly2MaskWithRatio

template <typename DType>
inline void convertPoly2Mask(const DType *roi,
                             const DType *poly,
                             const int mask_size,
                             DType *mask){
     /* !
     Converts a polygon to a pre-defined mask wrt to an roi
     *****Inputs****
     roi: The RoI bounding box
     poly: The polygon points the pre-defined format(see below)
     mask_size: The mask size
     *****Outputs****
     overlap: overlap of each box in boxes1 to each box in boxes2
     */
      DType w = roi[2] - roi[0];
      DType h = roi[3] - roi[1];
      w = max((DType)1., w);
      h = max((DType)1., h);
      int category = static_cast<int>(poly[0]);
      int n_seg = static_cast<int>(poly[1]);

      int offset = 2 + n_seg;
      RLE* rles;

      rlesInit(&rles, n_seg);
      for(int i = 0; i < n_seg; i++){
        int cur_len = poly[i+2];
        double* xys = new double[cur_len];
        for(int j = 0; j < cur_len; j++){
          if (j % 2 == 0)
            xys[j] = (poly[offset+j+1] - roi[1]) * mask_size / h;
          else
            xys[j] = (poly[offset+j-1] - roi[0]) * mask_size / w;
        }
        rleFrPoly(rles + i, xys, cur_len/2, mask_size, mask_size);
        delete [] xys;
        offset += cur_len;
      }
      // Decode RLE to mask
      byte* byte_mask = new byte[mask_size*mask_size*n_seg];
      rleDecode(rles, byte_mask, n_seg);

      DType* mask_cat = mask;
      // Flatten mask
      for(int j = 0; j < mask_size*mask_size; j++){
        float cur_byte = 0;
        for(int i = 0; i< n_seg; i++){
          int offset = i * mask_size * mask_size + j;
          if(byte_mask[offset]==1){
            cur_byte = 1;
            break;
          }
        }
        mask_cat[j] = cur_byte;
      }

      // Check to make sure we don't have memory leak
      rlesFree(&rles, n_seg);
      delete [] byte_mask;
} // convertPoly2Mask


namespace mshadow {
namespace proposal_mask_target {
template <typename DType>
inline void SampleROIMask(const Tensor<cpu, 2, DType> &all_rois,
                      const Tensor<cpu, 2, DType> &gt_boxes,
                      const Tensor<cpu, 2, DType> &gt_polys,
                      const Tensor<cpu, 1, DType> &bbox_mean,
                      const Tensor<cpu, 1, DType> &bbox_std,
                      const Tensor<cpu, 1, DType> &bbox_weight,
                      const index_t fg_rois_per_image,
                      const index_t rois_per_image,
                      const index_t mask_size,
                      const index_t num_classes,
                      const float fg_thresh,
                      const float bg_thresh_hi,
                      const float bg_thresh_lo,
                      const int image_rois,
                      const bool class_agnostic,
                      const bool output_ratio,
                      Tensor<cpu, 2, DType> &&rois,
                      Tensor<cpu, 1, DType> &&labels,
                      Tensor<cpu, 2, DType> &&bbox_targets,
                      Tensor<cpu, 2, DType> &&bbox_weights,
                      Tensor<cpu, 1, DType> &&match_gt_ious,
                      Tensor<cpu, 3, DType> &&mask_targets,
                      Tensor<cpu, 1, DType> &&mask_ratios) {
  /*
  overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
  gt_assignment = overlaps.argmax(axis=1)
  overlaps = overlaps.max(axis=1)
  labels = gt_boxes[gt_assignment, 4]
  */
  TensorContainer<cpu, 2, DType> IOUs(Shape2(all_rois.size(0), gt_boxes.size(0)), 0.f);
  BBoxOverlap(all_rois, gt_boxes, IOUs);

  vector<DType> max_overlaps(all_rois.size(0), 0.f);
  vector<DType> all_labels(all_rois.size(0), 0.f);
  vector<DType> gt_assignment(all_rois.size(0), 0.f);
  for (index_t i = 0; i < IOUs.size(0); ++i) {
      DType max_value = IOUs[i][0];
      index_t max_index = 0;
      for (index_t j = 1; j < IOUs.size(1); ++j) {
        if (max_value < IOUs[i][j]) {
            max_value = IOUs[i][j];
            max_index = j;
        }
      }
      gt_assignment[i] = max_index;
      max_overlaps[i] = max_value;
      all_labels[i] = gt_boxes[max_index][4];
  }
  /*
  fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
  fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
  if len(fg_indexes) > fg_rois_per_this_image:
    fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)
  */
  vector<index_t> fg_indexes;
  vector<index_t> neg_indexes;
  for (index_t i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= fg_thresh) {
      fg_indexes.push_back(i);
    } else {
      neg_indexes.push_back(i);
    }
  }
  // when image_rois != -1, subsampling rois
  index_t fg_rois_this_image;
  if (image_rois != -1)
    fg_rois_this_image = min<index_t>(fg_rois_per_image, fg_indexes.size());
  else
    fg_rois_this_image = fg_indexes.size();
  if (fg_indexes.size() > fg_rois_this_image) {
    random_shuffle(begin(fg_indexes), end(fg_indexes));
    fg_indexes.resize(fg_rois_this_image);
  }

  /*
  bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
  bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
  bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
  if len(bg_indexes) > bg_rois_per_this_image:
    bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)
  */
  vector<index_t> bg_indexes;
  for (index_t i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= bg_thresh_lo && max_overlaps[i] < bg_thresh_hi) {
        bg_indexes.push_back(i);
    }
  }
  index_t bg_rois_this_image = min<index_t>(rois_per_image - fg_rois_this_image, bg_indexes.size());
  if (bg_indexes.size() > bg_rois_this_image) {
      random_shuffle(begin(bg_indexes), end(bg_indexes));
      bg_indexes.resize(bg_rois_this_image);
  }
  //printf("fg %d bg %d\n", fg_rois_this_image,bg_rois_this_image);
  // keep_indexes = np.append(fg_indexes, bg_indexes)
  vector<index_t> kept_indexes;
  for (index_t i = 0; i < fg_rois_this_image; ++i) {
      kept_indexes.push_back(fg_indexes[i]);
  }
  for (index_t i = 0; i < bg_rois_this_image; ++i) {
      kept_indexes.push_back(bg_indexes[i]);
  }

  // pad with negative rois, original code is GARBAGE and omitted
  while (kept_indexes.size() < rois_per_image && neg_indexes.size() > 0) {
      index_t gap = rois_per_image - kept_indexes.size();
      random_shuffle(begin(neg_indexes), end(neg_indexes));
      for (index_t idx = 0;idx < gap && idx < neg_indexes.size();++idx) {
          kept_indexes.push_back(neg_indexes[idx]);
      }
  }
  /*
  labels = labels[keep_indexes]
  labels[fg_rois_per_this_image:] = 0
  rois = rois[keep_indexes]
  */
  for (index_t i = 0; i < kept_indexes.size(); ++i) {
    if (i < fg_rois_this_image){
      labels[i] = all_labels[kept_indexes[i]];
    }
    Copy(rois[i], all_rois[kept_indexes[i]]);
    match_gt_ious[i] = max_overlaps[kept_indexes[i]];
  }


  TensorContainer<cpu, 2, DType> rois_tmp(Shape2(rois.size(0), 4));
  for (index_t i = 0; i < rois_tmp.size(0); ++i) {
      Copy(rois_tmp[i], rois[i]);
  }

  TensorContainer<cpu, 2, DType> gt_bboxes_tmp(Shape2(rois.size(0), 4));
  for (index_t i = 0; i < rois_tmp.size(0); ++i) {
      Copy(gt_bboxes_tmp[i], gt_boxes[gt_assignment[kept_indexes[i]]].Slice(0, 4));
  }
  TensorContainer<cpu, 2, DType> targets(Shape2(rois.size(0), 4));
  NonLinearTransformAndNormalization(rois_tmp, gt_bboxes_tmp, targets, bbox_mean, bbox_std);

  TensorContainer<cpu, 2, DType> bbox_target_data(Shape2(targets.size(0), 5));
  for (index_t i = 0; i < bbox_target_data.size(0); ++i) {
      if (class_agnostic){
        //class-agnostic regression class index = {0, 1}
//        bbox_target_data[i][0] = min(1.f, labels[i]);
        bbox_target_data[i][0] = labels[i]< static_cast<DType>(1) ? labels[i]:static_cast<DType>(1);
      }else{
        bbox_target_data[i][0] = labels[i];
      }
      Copy(bbox_target_data[i].Slice(1, 5), targets[i]);
  }

  ExpandBboxRegressionTargets(bbox_target_data, bbox_targets, bbox_weights, bbox_weight);
  if (output_ratio){
    for (index_t i=0; i < fg_rois_this_image; ++i) {
        mask_ratios[i] = convertPoly2MaskWithRatio(rois[i].dptr_, gt_polys[gt_assignment[kept_indexes[i]]].dptr_, mask_size, mask_targets[i].dptr_);
    }
  }
  else{
     for (index_t i=0; i < fg_rois_this_image; ++i) {
        convertPoly2Mask(rois[i].dptr_, gt_polys[gt_assignment[kept_indexes[i]]].dptr_, mask_size, mask_targets[i].dptr_);
     }
  }

}

template <typename DType>
void BBoxOverlap(const Tensor<cpu, 2, DType> &boxes,
                 const Tensor<cpu, 2, DType> &query_boxes,
                 Tensor<cpu, 2, DType> &overlaps) {
    const index_t n = boxes.size(0);
    const index_t k = query_boxes.size(0);
    for (index_t j = 0; j < k; ++j) {
        DType query_box_area = (query_boxes[j][2] - query_boxes[j][0] + 1.f) * (query_boxes[j][3] - query_boxes[j][1] + 1.f);
        for (index_t i = 0; i < n; ++i) {
            DType iw = min(boxes[i][2], query_boxes[j][2]) - max(boxes[i][0], query_boxes[j][0]) + 1.f;
            if (iw > 0) {
                DType ih = min(boxes[i][3], query_boxes[j][3]) - max(boxes[i][1], query_boxes[j][1]) + 1.f;
                if (ih > 0) {
                    DType box_area = (boxes[i][2] - boxes[i][0] + 1.f) * (boxes[i][3] - boxes[i][1] + 1.f);
                    DType union_area = box_area + query_box_area - iw * ih;
                    overlaps[i][j] = iw * ih / union_area;
                }
            }
        }
    }
}

template <typename DType>
void ExpandBboxRegressionTargets(const Tensor<cpu, 2, DType> &bbox_target_data,
                                 Tensor<cpu, 2, DType> &bbox_targets,
                                 Tensor<cpu, 2, DType> &bbox_weights,
                                 const Tensor<cpu, 1, DType> &bbox_weight) {
  index_t num_bbox = bbox_target_data.size(0);
  for (index_t i = 0; i < num_bbox; ++i) {
    if (bbox_target_data[i][0] > 0) {
      index_t cls = bbox_target_data[i][0];
      index_t start = 4 * cls;
      index_t end = start + 4;
      Copy(bbox_targets[i].Slice(start, end), bbox_target_data[i].Slice(1, 5));
      Copy(bbox_weights[i].Slice(start, end), bbox_weight);
    }
  }
}

template <typename DType>
void NonLinearTransformAndNormalization(const Tensor<cpu, 2, DType> &ex_rois,
                                        const Tensor<cpu, 2, DType> &gt_rois,
                                        Tensor<cpu, 2, DType> &targets,
                                        const Tensor<cpu, 1, DType> &bbox_mean,
                                        const Tensor<cpu, 1, DType> &bbox_std) {
  index_t num_roi = ex_rois.size(0);
  for (index_t i = 0; i < num_roi; ++i) {
      DType ex_width  = ex_rois[i][2] - ex_rois[i][0] + 1.f;
      DType ex_height = ex_rois[i][3] - ex_rois[i][1] + 1.f;
      DType ex_ctr_x  = ex_rois[i][0] + 0.5 * (ex_width - 1.f);
      DType ex_ctr_y  = ex_rois[i][1] + 0.5 * (ex_height - 1.f);
      DType gt_width  = gt_rois[i][2] - gt_rois[i][0] + 1.f;
      DType gt_height = gt_rois[i][3] - gt_rois[i][1] + 1.f;
      DType gt_ctr_x  = gt_rois[i][0] + 0.5 * (gt_width - 1.f);
      DType gt_ctr_y  = gt_rois[i][1] + 0.5 * (gt_height - 1.f);
      targets[i][0]   = (gt_ctr_x - ex_ctr_x) / (ex_width + 1e-14f);
      targets[i][1]   = (gt_ctr_y - ex_ctr_y) / (ex_height + 1e-14f);
      targets[i][2]   = log(gt_width / ex_width);
      targets[i][3]   = log(gt_height / ex_height);
      targets[i] -= bbox_mean;
      targets[i] /= bbox_std;
  }
}

}  // namespace pt

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ProposalMaskTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalMaskTargetOp<cpu, DType>(param);
  })
  return op;
}

template<>
Operator *CreateOp<gpu>(ProposalMaskTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalMaskTargetOp<gpu, DType>(param);
  })
  return op;
}

Operator *ProposalMaskTargetProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ProposalMaskTargetParam);

MXNET_REGISTER_OP_PROPERTY(ProposalMaskTarget, ProposalMaskTargetProp)
.describe("C++ version proposal target")
.add_argument("data", "Symbol[]", "rois, gtboxes, gt_polys[, valid_ranges]")
.add_arguments(ProposalMaskTargetParam::__FIELDS__())
.set_key_var_num_args("num_args");

}  // namespace op
}  // namespace mxnet