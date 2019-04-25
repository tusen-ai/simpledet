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
 * \file generate_anchor-inl.h
 * \brief GenerateAnchor Operator
 * \author Yanghao Li, Chenxia Han
*/
#ifndef MXNET_OPERATOR_CONTRIB_GENERATE_ANCHOR_INL_H_
#define MXNET_OPERATOR_CONTRIB_GENERATE_ANCHOR_INL_H_

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

namespace gen_anchor {
enum GenAnchorOpInputs {kClsProb, kAnchor};
enum GenAnchorOpOutputs {kOut};
}  // gen_anchor

struct GenAnchorParam : public dmlc::Parameter<GenAnchorParam> {
  // use double to keep consistency with python implementation
  nnvm::Tuple<double> scales;
  nnvm::Tuple<double> ratios;
  int feature_stride;

  DMLC_DECLARE_PARAMETER(GenAnchorParam) {
    DMLC_DECLARE_FIELD(scales).set_default(nnvm::Tuple<double>({4.0f, 8.0f, 16.0f, 32.0f}))
    .describe("Used to generate anchor windows by enumerating scales");
    DMLC_DECLARE_FIELD(ratios).set_default(nnvm::Tuple<double>({0.5f, 1.0f, 2.0f}))
    .describe("Used to generate anchor windows by enumerating ratios");
    DMLC_DECLARE_FIELD(feature_stride).set_default(16)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
  }
};

template<typename xpu>
Operator *CreateOp(GenAnchorParam param);

#if DMLC_USE_CXX11
class GenAnchorProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[cls_prob]";
    const TShape &dshape = in_shape->at(gen_anchor::kClsProb);
    if (dshape.ndim() == 0) return false;
    int num_anchors = param_.scales.ndim() * param_.ratios.ndim();
    out_shape->clear();
    // output
    out_shape->push_back(Shape2(dshape[2] * dshape[3] * num_anchors,  4));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GenAnchorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_GenAnchor";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  GenAnchorParam param_;
};  // class GenAnchorProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

//========================
// Anchor Generation Utils
//========================
namespace mxnet {
namespace op {
namespace gen_anchor_utils {

template <typename DType>
inline void _MakeAnchor(DType w,
                        DType h,
                        DType x_ctr,
                        DType y_ctr,
                        std::vector<DType>& out_anchors) {
  out_anchors.push_back(x_ctr - 0.5f * (w - 1.0f));
  out_anchors.push_back(y_ctr - 0.5f * (h - 1.0f));
  out_anchors.push_back(x_ctr + 0.5f * (w - 1.0f));
  out_anchors.push_back(y_ctr + 0.5f * (h - 1.0f));
}

template <typename DType>
inline void _Transform(DType scale,
                       DType ratio,
                       const std::vector<DType>& base_anchor,
                       std::vector<DType>& out_anchors) {
  // use double in intermedia computation for consistency with numpy
  DType w = base_anchor[2] - base_anchor[0] + 1.0f;
  DType h = base_anchor[3] - base_anchor[1] + 1.0f;
  DType x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  DType y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  DType size = w * h;
  DType size_ratios = size / ratio;
  DType new_w = std::rint(std::sqrt(size_ratios)) * scale;
  DType new_h = std::rint((new_w / scale * ratio)) * scale;

  _MakeAnchor(new_w, new_h, x_ctr, y_ctr, out_anchors);
}

// out_anchors must have shape (n, 4), where n is ratios.size() * scales.size()
template <typename DType>
inline void GenerateAnchors(const std::vector<DType>& base_anchor,
                            const std::vector<DType>& ratios,
                            const std::vector<DType>& scales,
                            std::vector<DType>& out_anchors) {
  for (size_t j = 0; j < ratios.size(); ++j) {
    for (size_t k = 0; k < scales.size(); ++k) {
      _Transform(scales[k], ratios[j], base_anchor, out_anchors);
    }
  }
}

}  // namespace anchor_utils
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_GENERATE_ANCHOR_INL_H_
