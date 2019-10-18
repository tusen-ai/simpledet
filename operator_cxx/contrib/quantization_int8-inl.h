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
 * Copyright (c) 2019 by Contributors
 * \file quantization_int8-inl.h
 * paper link: https://arxiv.org/abs/1712.05877
* \author Xiaotao Chen, Jingqiu Zhou, Ruize Hou
*/

#ifndef MXNET_OPERATOR_QUANTIZATION_INT8_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_INT8_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/control_flow_op.h"
#include "../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

namespace Quantization_int8_enum {
enum Quantization_int8OpInputs {kData};
enum Quantization_int8OpOutputs {kOut};
enum Quantization_int8OpAuxiliary {kMinmax};
enum Quantization_int8OpResource {kTempSpace};
}  // namespace quantization_int8_enum

template<typename DType>
struct find_maxabs {
  MSHADOW_XINLINE static void Map(int i, DType *imin_range, DType* imax_range) {
    if (i < 1){
      *imax_range = MaxAbs(*imin_range, *imax_range);
    }
  }
};

template<typename xpu, typename DType>
void find_max(const OpContext &ctx, const TBlob &data, mshadow::Stream<xpu> *s, 
              mshadow::Tensor<xpu, 1, char> &temp_reduce_space, TBlob &in_min_t, TBlob &in_max_t,
              const mxnet::TShape &src_shape, const mxnet::TShape &dst_shape){
    using namespace mshadow;
    using namespace mshadow::expr;
    broadcast::Reduce<red::minimum, 2, DType, mshadow::op::identity>(
        s, in_min_t.reshape(dst_shape), kWriteTo, temp_reduce_space, data.reshape(src_shape));
    broadcast::Reduce<red::maximum, 2, DType, mshadow::op::identity>(
        s, in_max_t.reshape(dst_shape), kWriteTo, temp_reduce_space, data.reshape(src_shape));

    // the maxabs value is save in in_max_t
    mxnet_op::Kernel<find_maxabs<DType>, xpu>::Launch(s, 1, in_min_t.dptr<DType>(), in_max_t.dptr<DType>());
}

struct Quantization_int8Para : public dmlc::Parameter<Quantization_int8Para> {
  std::string quant_mode; 
  bool is_weight;
  bool is_weight_perchannel;
  int delay_quant;
  float ema_decay;
  std::string grad_mode;
  bool fix_act_scale;
  DMLC_DECLARE_PARAMETER(Quantization_int8Para) {
    DMLC_DECLARE_FIELD(quant_mode).set_default("minmax")
    .describe("select quantization mode");
    DMLC_DECLARE_FIELD(is_weight).set_default(true)
    .describe("if true, this quantization layer is used for weight");
    DMLC_DECLARE_FIELD(is_weight_perchannel).set_default(false)
    .describe("if true, this quantization layer is used for weight with per channel quantize");
    DMLC_DECLARE_FIELD(delay_quant).set_default(0)
    .describe("number of steps before quatization is used");
    DMLC_DECLARE_FIELD(ema_decay).set_default(0.99)
    .describe("the rate at which quantization range decay in ema");
    DMLC_DECLARE_FIELD(grad_mode).set_default("ste")
    .describe("select gradient pass mode");
    DMLC_DECLARE_FIELD(fix_act_scale).set_default(false)
    .describe("fix the minmax value of activation or not. defalut is False");
  }
};

template<typename xpu, typename DType>
class Quantization_int8Op : public Operator {
 public:
  explicit Quantization_int8Op(Quantization_int8Para param) {
    this->param_ = param;
    quant_countdown = param.delay_quant;
    init=true;
    QUANT_LEVEL = 127;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[Quantization_int8_enum::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[Quantization_int8_enum::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[Quantization_int8_enum::kData].shape_[0],
                               in_data[Quantization_int8_enum::kData].shape_[1], 1, 1);
      data = in_data[Quantization_int8_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[Quantization_int8_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[Quantization_int8_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[Quantization_int8_enum::kOut].get<xpu, 4, DType>(s);
    }
    Tensor<xpu, 1, DType> aux = aux_states[Quantization_int8_enum::kMinmax].get<xpu, 1, DType>(s);

    /*
    for weight per channel, OIHW, the axis should be the O dimension
    for all per channel, OIHW the axis should be the I dimension, currently this is not considered
    transpose conv weight is also not considered
    */
    if (ctx.is_train > 0 && quant_countdown > 0) {
      mshadow::Copy(out, data, s);
      quant_countdown = quant_countdown - 1;
    }
    else {
      if (param_.quant_mode == std::string("minmax")) {
        mxnet::TShape src_shape, dst_shape;
        const size_t temp_reduce_size = ConfigReduce<xpu, DType>(
            s, data.shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);

        Tensor<xpu, 1, uint8_t> workspace = ctx.requested[Quantization_int8_enum::kTempSpace]
          .get_space_typed<xpu, 1, uint8_t>(Shape1(temp_reduce_size + 2 * sizeof(DType)), s);
        uint64_t allocated_bytes = 0ULL;

        Tensor<xpu, 1, char> temp_reduce_space(reinterpret_cast<char*>(workspace.dptr_ + allocated_bytes), 
                                       Shape1(temp_reduce_size), s);
        allocated_bytes += temp_reduce_size;
        const int dev_id = ctx.run_ctx.ctx.dev_id;
        TBlob in_min_t(reinterpret_cast<DType *>(workspace.dptr_ + allocated_bytes), Shape1(1), xpu::kDevMask,
                      dev_id);
        allocated_bytes += sizeof(DType);
        TBlob in_max_t(reinterpret_cast<DType *>(workspace.dptr_ + allocated_bytes), Shape1(1), xpu::kDevMask,
                      dev_id);
        allocated_bytes += sizeof(DType);

        Tensor<xpu, 1, DType> max_val = in_max_t.get<xpu, 1, DType>(s);
        
        DType threshold = DType(0.0f);
        Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
        Tensor<cpu, 1, DType> threshold_tensor(&threshold, Shape1(1), s_cpu);


        if (param_.is_weight) {
          if (ctx.is_train > 0 && param_.fix_act_scale == false) {
            find_max<xpu, DType>(ctx, in_data[0], s, temp_reduce_space, in_min_t, in_max_t, src_shape, dst_shape);
            mshadow::Copy(aux, max_val, s);
          }
          // DType quant_unit = aux[0] / QUANT_LEVEL;
          mshadow::Copy(threshold_tensor, aux, s);
          DType quant_unit = threshold / QUANT_LEVEL;
          // there must scalarExp in mshadow calculate when data is single value.
          const ScalarExp<DType> quant_unit_expr(quant_unit);
          const ScalarExp<DType> threshold_expr(threshold);
          // Assign(out, req[Quantization_int8_enum::kOut], 
          //        F<mshadow_op::round>(F<mshadow_op::clip>(data, threshold_expr) /quant_unit_expr) * quant_unit_expr);
          // there is no need to clip in max quantize mode.
          Assign(out, req[Quantization_int8_enum::kOut], 
                 F<mshadow_op::round>(data /quant_unit_expr) * quant_unit_expr);
        } 
        else {
          if (ctx.is_train > 0 && param_.fix_act_scale == false) {
            find_max<xpu, DType>(ctx, in_data[0], s, temp_reduce_space, in_min_t, in_max_t, src_shape, dst_shape);
            // upate act threshold with ema
            if (init) {
                // check the value of minmax in aux is  equal to 0, so this should be initialized. otherwise, the value of
                // minmax load from checkpoint, this don't need init.
                mshadow::Copy(threshold_tensor, aux, s);
                if (threshold < 1e-6) {
                  mshadow::Copy(aux, max_val, s);
                }
                init = false;
            }
            else {
              const ScalarExp<DType> ema_scalar(param_.ema_decay);
              const ScalarExp<DType> ema_minus_scalar(1 - param_.ema_decay);
              aux = ema_scalar * aux + ema_minus_scalar * max_val;
            }
          }
          // DType quant_unit = aux[0] / QUANT_LEVEL;
          
          mshadow::Copy(threshold_tensor, aux, s);
          DType quant_unit = threshold / QUANT_LEVEL;
          const ScalarExp<DType> quant_unit_expr(quant_unit);
          const ScalarExp<DType> threshold_expr(threshold);
          Assign(out, req[Quantization_int8_enum::kOut], 
                 F<mshadow_op::round>(F<mshadow_op::clip>(data, threshold_expr) /quant_unit_expr) * quant_unit_expr);
        }
      }
      else {
        LOG(FATAL) << "quantization int8 only support minmax mode currently";
      }
    }
  }
  

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data_grad;
    Tensor<xpu, 4, DType> out_data_grad;
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (out_grad[Quantization_int8_enum::kOut].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[Quantization_int8_enum::kOut].shape_[0],
                               out_grad[Quantization_int8_enum::kOut].shape_[1], 1, 1);
      data = in_data[Quantization_int8_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
      out = out_data[Quantization_int8_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);

      out_data_grad = out_grad[Quantization_int8_enum::kOut].get_with_shape<xpu, 4, DType>(dshape, s);
      data_grad = in_grad[Quantization_int8_enum::kData].get_with_shape<xpu, 4, DType>(dshape, s);
    } else {
      data = in_data[Quantization_int8_enum::kData].get<xpu, 4, DType>(s);
      out = out_data[Quantization_int8_enum::kOut].get<xpu, 4, DType>(s);

      out_data_grad = out_grad[Quantization_int8_enum::kOut].get<xpu, 4, DType>(s);
      data_grad = in_grad[Quantization_int8_enum::kData].get<xpu, 4, DType>(s);
    }
    // Assign(data_grad, req[Quantization_int8_enum::kOut], out_data_grad);
    if (param_.grad_mode == std::string("ste") || param_.is_weight) {
      mshadow::Copy(data_grad, out_data_grad, s);
    }
    else if (param_.grad_mode == std::string("clip")) {
      Tensor<xpu, 1, uint8_t> workspace = ctx.requested[Quantization_int8_enum::kTempSpace]
        .get_space_typed<xpu, 1, uint8_t>(Shape1(2 * data.shape_.Size() * sizeof(DType) + sizeof(DType)), s);
      uint64_t allocated_bytes = 0ULL;
      Tensor<xpu, 4, DType> clip_condition(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
      allocated_bytes += clip_condition.shape_.Size() * sizeof(DType);
      Tensor<xpu, 4, DType> zero_bc(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), data.shape_, s);
      allocated_bytes += zero_bc.shape_.Size() * sizeof(DType);
      
      Tensor<xpu, 1, DType> temp(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
      allocated_bytes += temp.shape_.Size() * sizeof(DType);

      temp =  ScalarExp<DType>(0.0f);
      zero_bc = broadcast_scalar(temp, zero_bc.shape_);


      DType threshold = DType(0.0f);
      Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
      Tensor<cpu, 1, DType> threshold_tensor(&threshold, Shape1(1), s_cpu);
      Tensor<xpu, 1, DType> aux = aux_states[Quantization_int8_enum::kMinmax].get<xpu, 1, DType>(s);
      mshadow::Copy(threshold_tensor, aux, s);
      const ScalarExp<DType> min_expr(-threshold);
      const ScalarExp<DType> max_expr(threshold);
      clip_condition = F<mshadow_op::ge>(data, min_expr) * F<mshadow_op::le>(data, max_expr);

      Kernel<mxnet::op::where<kWriteTo>, xpu>::Launch(s, data_grad.shape_.Size(),
      data_grad.dptr_, clip_condition.dptr_, out_data_grad.dptr_, zero_bc.dptr_);
    }
    else {
      LOG(FATAL) << "quantization int8 only support ste/clip gradient pass mode currently";
    }
  }

 private:
  Quantization_int8Para param_;
  int quant_countdown;
  int QUANT_LEVEL;
  bool init;

};  // class Quantization_int8Op

template<typename xpu>
Operator* CreateOp(Quantization_int8Para type, int dtype);

#if DMLC_USE_CXX11
class Quantization_int8Prop : public OperatorProperty {
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

    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    const TShape &dshape = in_shape->at(Quantization_int8_enum::kData);
    out_shape->clear();
    out_shape->push_back(dshape);
    
    Shape<1>  tmp_aux_shape = Shape1(1);
    if (param_.is_weight && param_.is_weight_perchannel) {
      tmp_aux_shape = Shape1(dshape[0]);
    }

    aux_shape->clear();
    aux_shape->push_back(TShape(tmp_aux_shape));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
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
    aux_type->clear();
    aux_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new Quantization_int8Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_Quantization_int8";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[Quantization_int8_enum::kOut], 
            in_data[Quantization_int8_enum::kData], 
            out_data[Quantization_int8_enum::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"minmax"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                           std::vector<int> *in_type) const override;

 private:
  Quantization_int8Para param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_Qunatization_Int8_INL_H_

