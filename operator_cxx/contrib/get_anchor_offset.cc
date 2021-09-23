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
 * \file get_anchor_offset.cc
 * \brief
 * \author Chenxia Han
*/

#include "./get_anchor_offset-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(GetAnchorOffsetParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GetAnchorOffsetOp<cpu, DType>(param);
  });
  return op;
}

Operator *GetAnchorOffsetProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(GetAnchorOffsetParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_GetAnchorOffset, GetAnchorOffsetProp)
.describe("Compute offset for Deformable Convolution")
.add_argument("data", "NDArray-or-Symbol", "Data to determine height and width")
.add_argument("anchor", "NDArray-or-Symbol", "Anchor to determine offset")
.add_arguments(GetAnchorOffsetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
