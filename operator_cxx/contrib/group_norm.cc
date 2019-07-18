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
