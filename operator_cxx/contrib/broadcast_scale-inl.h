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
 * \file broadcast_scale-inl.h
 * \brief BroadcastScale Operator
 * \author Yuntao Chen
*/
#ifndef MXNET_OPERATOR_CONTRIB_BROADCAST_SCALE_INL_H_
#define MXNET_OPERATOR_CONTRIB_BROADCAST_SCALE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cmath>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../tensor/control_flow_op.h"
#include "../tensor/indexing_op.h"

namespace mxnet {
namespace op {

namespace broadcast_scale_enum {
enum BroadcastScaleOpInputs {kData, kScaler};
enum BroadcastScaleOpOutputs {kOut};
enum BroadcastScaleOpResource {kTempSpace};
}  // namespace broadcast_scale_enum

struct BroadcastScaleParam : public dmlc::Parameter<BroadcastScaleParam> {
  DMLC_DECLARE_PARAMETER(BroadcastScaleParam) {}
};

template<typename xpu, typename DType>
class BroadcastScaleOp : public Operator{
 public:
  explicit BroadcastScaleOp(BroadcastScaleParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[broadcast_scale_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> scaler = in_data[broadcast_scale_enum::kScaler].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[broadcast_scale_enum::kOut].get<xpu, 4, DType>(s);

    Assign(out, req[broadcast_scale_enum::kOut], data * broadcast<1>(scaler, data.shape_));
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
    using namespace mxnet_op;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(in_grad.size(), 2U);
    CHECK_EQ(req[broadcast_scale_enum::kData], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> ograd = out_grad[broadcast_scale_enum::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[broadcast_scale_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> scaler = in_data[broadcast_scale_enum::kScaler].FlatTo1D<xpu, DType>(s);

    Assign(gdata, req[broadcast_scale_enum::kData], ograd * broadcast<1>(scaler, ograd.shape_));
  }

 private:
  BroadcastScaleParam param_;
};  // class BroadcastScaleOp

template<typename xpu>
Operator *CreateOp(BroadcastScaleParam param, int dtype);

#if DMLC_USE_CXX11
class BroadcastScaleProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, scaler]";
    const TShape &dshape = in_shape->at(broadcast_scale_enum::kData);
    const TShape &sshape = in_shape->at(broadcast_scale_enum::kScaler);
    CHECK_EQ(dshape.ndim(), 4) << "Only 4D input is supported now";
    CHECK_EQ(sshape.ndim(), 4) << "Only 4D input is supported now";

    for (int i = 0; i < sshape.ndim(); ++i) {
      if (sshape[i] != 1) {
        CHECK_EQ(dshape[i], sshape[i]);
      }
    }

    out_shape->clear();
    // output
    out_shape->push_back(in_shape->at(broadcast_scale_enum::kData));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (size_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BroadcastScaleProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_BroadcastScale";
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "scaler"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[broadcast_scale_enum::kScaler], out_grad[broadcast_scale_enum::kOut]};
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[broadcast_scale_enum::kData], out_data[broadcast_scale_enum::kOut]}};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const {
    return {{out_grad[broadcast_scale_enum::kOut], in_grad[broadcast_scale_enum::kData]}};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BroadcastScaleParam param_;
};  // class BroadcastScaleProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_BROADCAST_SCALE_INL_H_
