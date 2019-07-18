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
 * \file group_norm.cc
 * \author Yuntao Chen
*/

#include "./group_norm-inl.h"

namespace mxnet {
namespace op {
template <>
Operator* CreateOp<cpu>(GroupNormParam param, int dtype) {
  LOG(FATAL) << "not implemented.";
  return NULL;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* GroupNormProp::CreateOperatorEx(Context ctx,
                                          std::vector<TShape>* in_shape,
                                          std::vector<int>* in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(GroupNormParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_GroupNorm, GroupNormProp)
.add_argument("data", "NDArray-or-Symbol",
              "An n-dimensional input array (n > 2) of the form [batch, "
              "channel, spatial_dim1, spatial_dim2, ...].")
.add_argument("gamma", "NDArray-or-Symbol",
              "A vector of length \'channel\', which multiplies the "
              "normalized input.")
.add_argument("beta", "NDArray-or-Symbol",
              "A vector of length \'channel\', which is added to the "
              "product of the normalized input and the weight.")
.add_arguments(GroupNormParam::__FIELDS__())
.describe(R"code(Group Normalization (GN) operation: https://arxiv.org/abs/1803.08494)code" ADD_FILELINE);
}  // namespace op
}  // namespace mxnet
