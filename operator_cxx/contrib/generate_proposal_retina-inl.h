/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file generate_proposal_retina-inl.h
 * \brief GenProposalRetina Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Pengfei Chen, Yuntao Chen, Yanghao Li, Chenxia Han
*/
#ifndef MXNET_OPERATOR_CONTRIB_GENERATE_PROPOSAL_RETINA_INL_H_
#define MXNET_OPERATOR_CONTRIB_GENERATE_PROPOSAL_RETINA_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace gen_proposal_retina {
enum GenProposalRetinaOpInputs {kClsProb, kBBoxPred, kImInfo, kAnchor};
enum GenProposalRetinaOpOutputs {kOut, kScore};
enum GenProposalRetinaForwardResource {kTempSpace};
}  // gen_proposal_retina

struct GenProposalRetinaParam : public dmlc::Parameter<GenProposalRetinaParam> {
  int rpn_pre_nms_top_n;
  int rpn_min_size;
  int feature_stride;
  int num_anchors;
  float thresh;
  nnvm::Tuple<float> anchor_mean;
  nnvm::Tuple<float> anchor_std;
  bool iou_loss;
  bool output_one_hot;
  bool batch_wise_anchor;
  uint64_t workspace;

  DMLC_DECLARE_PARAMETER(GenProposalRetinaParam) {
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(16)
    .describe("Minimum height or width in proposal");
    DMLC_DECLARE_FIELD(feature_stride).set_default(16)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
    DMLC_DECLARE_FIELD(num_anchors)
    .describe("The total number of anchors");
    DMLC_DECLARE_FIELD(thresh).set_default(0.0f)
    .describe("Minimum scores");
    DMLC_DECLARE_FIELD(anchor_mean)
    .set_default(nnvm::Tuple<float>({0.f, 0.f, 0.f, 0.f}))
    .describe("Anchor target mean");
    DMLC_DECLARE_FIELD(anchor_std)
    .set_default(nnvm::Tuple<float>({1.f, 1.f, 1.f, 1.f}))
    .describe("Anchor target std");
    DMLC_DECLARE_FIELD(iou_loss).set_default(false)
    .describe("Usage of IoU Loss");
    DMLC_DECLARE_FIELD(output_one_hot).set_default(true)
    .describe("Whether output score as one hot");
    DMLC_DECLARE_FIELD(batch_wise_anchor).set_default(false)
    .describe("Whether input anchor is batch-wise");
    DMLC_DECLARE_FIELD(workspace).set_default(256)
    .describe("Workspace for proposal in MB, default to 256");
  }
};

template<typename xpu>
Operator *CreateOp(GenProposalRetinaParam param);

#if DMLC_USE_CXX11
class GenProposalRetinaProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 4) << "Input:[cls_prob, bbox_pred, im_info, anchors]";
    const TShape &dshape = in_shape->at(gen_proposal_retina::kClsProb);
    if (dshape.ndim() == 0) return false;
    Shape<4> bbox_pred_shape;
    bbox_pred_shape = Shape4(dshape[0], param_.num_anchors * 4, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, gen_proposal_retina::kBBoxPred,
                       bbox_pred_shape);
    Shape<2> im_info_shape;
    im_info_shape = Shape2(dshape[0], 3);
    SHAPE_ASSIGN_CHECK(*in_shape, gen_proposal_retina::kImInfo, im_info_shape);
    bool batch_wise_anchor = param_.batch_wise_anchor;
    if (batch_wise_anchor) {
      Shape<3> anchors_shape;
      anchors_shape = Shape3(dshape[0], dshape[2] * dshape[3] * param_.num_anchors,  4);
      SHAPE_ASSIGN_CHECK(*in_shape, gen_proposal_retina::kAnchor, anchors_shape);
    } else {
      Shape<2> anchors_shape;
      anchors_shape = Shape2(dshape[2] * dshape[3] * param_.num_anchors,  4);
      SHAPE_ASSIGN_CHECK(*in_shape, gen_proposal_retina::kAnchor, anchors_shape);
    }
    out_shape->clear();
    // output
    const int num_class = dshape[1] / param_.num_anchors + 1;
    const int out_channel = param_.output_one_hot ? num_class : 1;
    out_shape->push_back(Shape3(dshape[0], param_.rpn_pre_nms_top_n, 4));
    out_shape->push_back(Shape3(dshape[0], param_.rpn_pre_nms_top_n, out_channel));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GenProposalRetinaProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_GenProposalRetina";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "bbox_pred", "im_info", "anchors"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "scores"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  GenProposalRetinaParam param_;
};  // class GenProposalRetinaProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_GENERATE_PROPOSAL_RETINA_INL_H_
