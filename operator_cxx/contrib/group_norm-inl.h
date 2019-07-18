/*
Copyright (c) 2016-present, Facebook Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
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
 * \file group_norm-inl.h
 * \author Yuntao Chen
*/

#ifndef MXNET_OPERATOR_CONTRIB_GROUP_NORM_INL_H_
#define MXNET_OPERATOR_CONTRIB_GROUP_NORM_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace group_norm {
enum GroupNormInputs { kData, kGamma, kBeta };
enum GroupNormOutputs { kOut, kMu, kRsig };
enum GroupNormBackResource { kTempSpace };
}  // namespace group_norm

struct GroupNormParam : public dmlc::Parameter<GroupNormParam> {
  int num_group;
  float eps;
  DMLC_DECLARE_PARAMETER(GroupNormParam) {
    DMLC_DECLARE_FIELD(num_group).set_default(32).describe(
        "number of groups used by GN.");
    DMLC_DECLARE_FIELD(eps).set_default(1e-5f).describe(
        "small constant added to var to prevent division by 0.");
  }
};  // struct GroupNormParam

template <typename xpu>
class GroupNormOp : public Operator {
 public:
  explicit GroupNormOp(GroupNormParam param) { param_ = param; }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);

    CHECK_EQ(in_data[group_norm::kData].ndim(), 4) << "GroupNorm only supports NCHW tensor.";

    float eps = param_.eps;
    int N = in_data[group_norm::kData].size(0);
    int C = in_data[group_norm::kData].size(1);
    int HxW = in_data[group_norm::kData].Size() / (N * C);
    int G = param_.num_group;
    CHECK_EQ(C % G, 0U);
    int D = C / G;

    Stream<xpu> *s = ctx.get_stream<xpu>();
    // Get Inputs
    Tensor<xpu, 1> X = in_data[group_norm::kData].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> gamma = in_data[group_norm::kGamma].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> beta = in_data[group_norm::kBeta].FlatTo1D<xpu, float>(s);
    CHECK_EQ(C, gamma.shape_.Size());
    CHECK_EQ(C, beta.shape_.Size());

    // Get Outputs
    Tensor<xpu, 1> Y = out_data[group_norm::kOut].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> mu = out_data[group_norm::kMu].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> rsig = out_data[group_norm::kRsig].FlatTo1D<xpu, float>(s);

    GroupNormForward(eps, N, G, D, HxW, X, gamma, beta, Y, mu, rsig);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);

    CHECK_EQ(in_data[group_norm::kData].ndim(), 4) << "GroupNorm only supports NCHW tensor.";

    Stream<xpu> *s = ctx.get_stream<xpu>();
    int N = in_data[group_norm::kData].size(0);
    int C = in_data[group_norm::kData].size(1);
    int HxW = in_data[group_norm::kData].Size() / (N * C);
    int G = param_.num_group;
    CHECK_EQ(C % G, 0U);
    int D = C / G;

    // Get Inputs
    Tensor<xpu, 1> X = in_data[group_norm::kData].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> dX = in_grad[group_norm::kData].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> gamma = in_data[group_norm::kGamma].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> dgamma = in_grad[group_norm::kGamma].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> dbeta = in_grad[group_norm::kBeta].FlatTo1D<xpu, float>(s);
    CHECK_EQ(C, dgamma.shape_.Size());
    CHECK_EQ(C, dbeta.shape_.Size());
    // Get Outputs
    Tensor<xpu, 1> dY = out_grad[group_norm::kOut].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> rsig = out_data[group_norm::kRsig].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1> mu = out_data[group_norm::kMu].FlatTo1D<xpu, float>(s);
    // Get temp space
    Tensor<xpu, 2> workspace =
      ctx.requested[group_norm::kTempSpace].get_space<xpu>(Shape2(2, mu.shape_[0]), s);
    Tensor<xpu, 1> ds = workspace[0];
    Tensor<xpu, 1> db = workspace[1];

    GroupNormBackward(N, G, D, HxW, dY, X, mu, rsig, gamma, ds, db, dX, dgamma, dbeta);
  }

 private:
  GroupNormParam param_;
};  // class GroupNormOp

template <typename xpu>
Operator *CreateOp(GroupNormParam param, int dtype);

#if DMLC_USE_CXX11
class GroupNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input: [data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);

    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape2(dshape[0], dshape[1]));
    out_shape->push_back(Shape2(dshape[0], dshape[1]));
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new GroupNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "_contrib_GroupNorm"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[group_norm::kOut],
            out_data[group_norm::kMu],
            out_data[group_norm::kRsig],
            in_data[group_norm::kData],
            in_data[group_norm::kGamma]};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 3; }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  GroupNormParam param_;
};      // GroupNormProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_GROUP_NORM_INL_H_
