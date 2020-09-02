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
 * \file get_anchor_offset-inl.h
 * \brief GetAnchorOffset Operator
 * \author Chenxia Han
*/
#ifndef MXNET_OPERATOR_CONTRIB_GET_ANCHOR_OFFSET_INL_H_
#define MXNET_OPERATOR_CONTRIB_GET_ANCHOR_OFFSET_INL_H_

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

namespace mxnet {
namespace op {

namespace get_anchor_offset {
enum GetAnchorOffsetOpInputs {kData, kAnchor};
enum GetAnchorOffsetOpOutputs {kOut};
}  // get_anchor_offset

struct GetAnchorOffsetParam : public dmlc::Parameter<GetAnchorOffsetParam> {
  mxnet::TShape kernel;
  int stride;
  DMLC_DECLARE_PARAMETER(GetAnchorOffsetParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Sample size for each anchor: (h, w)");
    DMLC_DECLARE_FIELD(stride).describe("Stride at current layer");
  }
};

struct anchor_to_offset {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i, int k1, int k2, int stride,
                                  int num_anchors, int height, int width,
                                  const DType *anchor, DType *out) {
    int w = i % width;
    int h = i / width % height;
    int a = i / width / height % num_anchors;
    int n = i / width / height / num_anchors;

    int num_offset = num_anchors * k1 * k2 * 2;

    for (int kh = 0; kh < k1; ++kh) {
      for (int kw = 0; kw < k2; ++kw) {
        const DType *box = anchor + ((((n * height) + h) * width + w) * num_anchors + a) * 4;
        DType x1 = box[0] / stride;
        DType y1 = box[1] / stride;
        DType x2 = box[2] / stride;
        DType y2 = box[3] / stride;

        DType bin_size_x = (x2 - x1 + 1) / k2;
        DType bin_size_y = (y2 - y1 + 1) / k1;

        int offset_idx = ((a * k1 + kh) * k2 + kw) * 2;
        int offset_idx_x = ((n * num_offset + offset_idx + 1) * height + h) * width + w;
        int offset_idx_y = ((n * num_offset + offset_idx + 0) * height + h) * width + w;

        out[offset_idx_x] = x1 + (bin_size_x-1) / 2 + kw * bin_size_x;
        out[offset_idx_y] = y1 + (bin_size_y-1) / 2 + kh * bin_size_y;

        out[offset_idx_x] -= w + kw - (k2-1) / 2;
        out[offset_idx_y] -= h + kh - (k1-1) / 2;
      }
    }
  }
};

template<typename xpu, typename DType>
class GetAnchorOffsetOp : public Operator {
 public:
  explicit GetAnchorOffsetOp(GetAnchorOffsetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mxnet_op;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[get_anchor_offset::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    /*
     * data:    (n, c, h, w)
     * anchor:  (n, h * w * a, 4)
     * out:     (n, a * k_1 * k_2 * 2, h, w)
     */
    Tensor<xpu, 4, DType> data = in_data[get_anchor_offset::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> anchor = in_data[get_anchor_offset::kAnchor].get<xpu, 3, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[get_anchor_offset::kOut].get<xpu, 4, DType>(s);

    int height = data.size(2);
    int width = data.size(3);
    int num_anchors = anchor.size(1) / height / width;
    int count = anchor.shape_.ProdShape(0, 2);

    Kernel<anchor_to_offset, xpu>::Launch(s, count, param_.kernel[0], param_.kernel[1],
      param_.stride, num_anchors, height, width, anchor.dptr_, out.dptr_);
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
    CHECK_EQ(in_grad.size(), 2U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> gdata  = in_grad[get_anchor_offset::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> ganchor = in_grad[get_anchor_offset::kAnchor].get<xpu, 3, DType>(s);

    Assign(gdata, req[get_anchor_offset::kData], 0);
    Assign(ganchor, req[get_anchor_offset::kAnchor], 0);
  }

 private:
  GetAnchorOffsetParam param_;
};  // class GetAnchorOffsetOp

template<typename xpu>
Operator *CreateOp(GetAnchorOffsetParam param, int dtype);

#if DMLC_USE_CXX11
class GetAnchorOffsetProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, anchor]";
    const TShape &dshape = in_shape->at(get_anchor_offset::kData);
    const TShape &ashape = in_shape->at(get_anchor_offset::kAnchor);

    const int num_image = dshape[0];
    const int channel = dshape[1];
    const int height = dshape[2];
    const int width = dshape[3];
    const int num_anchors = ashape[1] / (height * width);
    const TShape &kernel = param_.kernel;

    auto data_shape = Shape4(num_image, channel, height, width);
    auto anchor_shape = Shape3(num_image, height * width * num_anchors, 4);
    auto offset_shape = Shape4(num_image, num_anchors * kernel[0] * kernel[1] * 2, height, width);

    SHAPE_ASSIGN_CHECK(*in_shape, get_anchor_offset::kData, data_shape);
    SHAPE_ASSIGN_CHECK(*in_shape, get_anchor_offset::kAnchor, anchor_shape);

    out_shape->clear();
    // output
    out_shape->push_back(offset_shape);
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
    auto ptr = new GetAnchorOffsetProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_GetAnchorOffset";
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "anchor"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  /*
  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[get_anchor_offset::kData], out_data[get_anchor_offset::kOut]}};
  }
  */

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  GetAnchorOffsetParam param_;
};  // class GetAnchorOffsetProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_GET_ANCHOR_OFFSET_INL_H_
