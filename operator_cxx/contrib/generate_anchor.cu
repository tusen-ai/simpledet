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
 * \file generate_proposal.cu
 * \brief Proposal Operator
 * \author Yanghao Li, Chenxia Han
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "../tensor/sort_op.h"

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iterator>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./generate_anchor-inl.h"

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace {
// all_anchors are (h * w * anchor, 4)
// w defines "x" and h defines "y"
// count should be total anchors numbers, h * w * anchors
template<typename DType>
__global__ void AnchorGridKernel(const int count,
                                 const int num_anchors,
                                 const int height,
                                 const int width,
                                 const int feature_stride,
                                 double* all_anchors,
                                 DType* out) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % width;
    int h = index / num_anchors / width;

    out[index * 4 + 0] = static_cast<DType>(all_anchors[a * 4 + 0] + w * feature_stride);
    out[index * 4 + 1] = static_cast<DType>(all_anchors[a * 4 + 1] + h * feature_stride);
    out[index * 4 + 2] = static_cast<DType>(all_anchors[a * 4 + 2] + w * feature_stride);
    out[index * 4 + 3] = static_cast<DType>(all_anchors[a * 4 + 3] + h * feature_stride);
  }
}

}  // namespace
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu>
class GenAnchorGPUOp : public Operator{
 public:
  explicit GenAnchorGPUOp(GenAnchorParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda;

    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[gen_anchor::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    // batch_idx, anchor_idx, height_idx, width_idx
    Tensor<xpu, 4> scores = in_data[gen_anchor::kClsProb].get<xpu, 4, float>(s);
    // height * width * anchors, 4(x1, y1, x2, y2)
    Tensor<xpu, 2> out = out_data[gen_anchor::kOut].get<xpu, 2, float>(s);

    std::vector<double> scales(param_.scales.begin(), param_.scales.end());
    std::vector<double> ratios(param_.ratios.begin(), param_.ratios.end());

    int num_anchors = scales.size() * ratios.size();
    int height = scores.size(2);
    int width = scores.size(3);

    // Generate first anchors based on base anchor
    std::vector<double> base_anchor({
      0.0f, 0.0f, param_.feature_stride - 1.0f, param_.feature_stride - 1.0f
    });
    std::vector<double> anchors;
    gen_anchor_utils::GenerateAnchors(
      base_anchor, ratios, scales, anchors
    );

    // cast to fp32 during AnchorGrid to keep consistency with python implementation
    TensorContainer<gpu, 2, double> out_fp64(out.shape_);

    FRCNN_CUDA_CHECK(
      cudaMemcpy(
        out_fp64.dptr_,
        anchors.data(),
        sizeof(decltype(anchors)::value_type) * anchors.size(),
        cudaMemcpyHostToDevice
      )
    ); // less than 64K

    /* copy proposals to a mesh grid */
    dim3 dimGrid((out.size(0) + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "AnchorGrid");
    AnchorGridKernel<<<dimGrid, dimBlock>>>(
      out_fp64.size(0), num_anchors, height, width, param_.feature_stride,
      out_fp64.dptr_, out.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());
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
    CHECK_EQ(in_grad.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[gen_anchor::kClsProb].get<xpu, 4, float>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[gen_anchor::kClsProb], 0);
  }

 private:
  GenAnchorParam param_;
};  // class GenAnchorGPUOp

template<>
Operator* CreateOp<gpu>(GenAnchorParam param) {
  return new GenAnchorGPUOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
