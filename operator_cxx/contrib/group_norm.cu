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
 * \file group_norm.cu
 * \author Yuntao Chen
*/

#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "./group_norm-inl.h"
#include "./group_norm_helper.h"

namespace mshadow {
namespace cuda {

template <typename DType>
__device__ inline DType Cube(const DType x) {
  return x * x * x;
}


template <typename DType>
__global__ void GroupNormForwardCUDAKernel(
    const int size,
    const int G,
    const int D,
    const int HxW,
    const DType* X,
    const DType* mu,
    const DType* rsig,
    const DType* gamma,
    const DType* beta,
    DType* Y) {
  const int C = G * D;
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int i_mu = i / (D * HxW);
    const int i_gamma = (i / HxW) % C;
    Y[i] = __ldg(gamma + i_gamma) * (__ldg(X + i) - __ldg(mu + i_mu)) *
           __ldg(rsig + i_mu) +
           __ldg(beta + i_gamma);
  }
}

template <typename DType>
__global__ void ComputeInternalGradientsCUDAKernel(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const DType* dY,
    const DType* X,
    const DType* gamma,
    DType* ds,
    DType* db) {
  const int outer_size = N * G;
  const int inner_size = D * HxW;
  __shared__ typename BlockReduce<DType>::TempStorage ds_storage;
  __shared__ typename BlockReduce<DType>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    DType ds_val = 0;
    DType db_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int i_gamma = i % G * D + j / HxW;
      const int index = i * inner_size + j;
      ds_val += __ldg(gamma + i_gamma) * __ldg(dY + index) * __ldg(X + index);
      db_val += __ldg(gamma + i_gamma) * __ldg(dY + index);
    }
    ds_val = BlockReduce<DType>(ds_storage).Reduce(ds_val, cub::Sum());
    db_val = BlockReduce<DType>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      ds[i] = ds_val;
      db[i] = db_val;
    }
    __syncthreads();
  }
}

// Math:
// Y = gamma * (X - mu) * rsig + beta
// let s = gamma * rsig
// let b = beta - mu * rsig
// Y = s * X + b
// let n = D * HxW
// dL/dX = dL/dY * dY/dX = dL/dY * (d(s * X)/dX + db/dX)
// d(s * X)/dX = s + X * ds/dX = s + gamma * X * drsig/dX
// db/dX = -u * drsig/dX - rsig * dmu/dX
// drsig/dX = -rsig^3 * (X - mu) / n
// dmu/dX = 1 / n
template <typename DType>
__global__ void GroupNormBackwardCUDAKernel(
    const int size,
    const int G,
    const int D,
    const int HxW,
    const DType* dY,
    const DType* X,
    const DType* mu,
    const DType* rsig,
    const DType* gamma,
    const DType* ds,
    const DType* db,
    DType* dX) {
  const int C = G * D;
  const DType denom = DType(1) / static_cast<DType>(D * HxW);
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int i_mu = i / (D * HxW);
    const int i_gamma = (i / HxW) % C;
    const DType u = (__ldg(db + i_mu) * __ldg(mu + i_mu) - __ldg(ds + i_mu)) *
        (__ldg(X + i) - __ldg(mu + i_mu)) *
        Cube<DType>(__ldg(rsig + i_mu));
    const DType v = __ldg(db + i_mu) * __ldg(rsig + i_mu);
    dX[i] = __ldg(gamma + i_gamma) * __ldg(dY + i) * __ldg(rsig + i_mu) +
        (u - v) * denom;
  }
}

template <typename DType>
__global__ void GammaBetaBackwardCUDAKernel(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const DType* dY,
    const DType* X,
    const DType* mu,
    const DType* rsig,
    DType* dgamma,
    DType* dbeta) {
  const int outer_size = G * D;
  const int inner_size = N * HxW;
  __shared__ typename BlockReduce<DType>::TempStorage dg_storage;
  __shared__ typename BlockReduce<DType>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    DType dg_val = 0;
    DType db_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int n = j / HxW;
      const int index = (n * outer_size + i) * HxW + j % HxW;
      const int i_mu = n * G + i / D;
      dg_val += __ldg(dY + index) * (__ldg(X + index) - __ldg(mu + i_mu)) *
          __ldg(rsig + i_mu);
      db_val += __ldg(dY + index);
    }
    dg_val = BlockReduce<DType>(dg_storage).Reduce(dg_val, cub::Sum());
    db_val = BlockReduce<DType>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
    __syncthreads();
  }
}

template<typename T>
inline void GroupNormForward(cudaStream_t stream,
                             T eps,
                             const int N,
                             const int G,
                             const int D,
                             const int HxW,
                             const T* X_data,
                             const T* gamma_data,
                             const T* beta_data,
                             T* Y_data,
                             T* mu_data,
                             T* rsig_data){
  const int size = N * G * D * HxW;
  const std::array<int, 2> dims = {N * G, D * HxW};
  const int axis = 1;
  Moments<T>(2, dims.data(), 1, &axis, X_data, mu_data, rsig_data, stream);
  InvStd<T>(N * G, eps, rsig_data, rsig_data, stream);

  GroupNormForwardCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         stream>>>(
          size,
          G,
          D,
          HxW,
          X_data,
          mu_data,
          rsig_data,
          gamma_data,
          beta_data,
          Y_data);
}


// Math:
// let: s = gamma * rsig
// let: b = beta - mu * gamma * rsig
// then: Y = s * X + b
template<typename T>
inline void GroupNormBackward(cudaStream_t stream,
                              const int N,
                              const int G,
                              const int D,
                              const int HxW,
                              const T* dY_data,
                              const T* X_data,
                              const T* mu_data,
                              const T* rsig_data,
                              const T* gamma_data,
                              T* ds_data,
                              T* db_data,
                              T* dX_data,
                              T* dgamma_data,
                              T* dbeta_data) {
  const int size = N * G * D * HxW;
  const int C = G * D;

  ComputeInternalGradientsCUDAKernel<T>
      <<<std::min(N * G, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         stream>>>(
          N, G, D, HxW, dY_data, X_data, gamma_data, ds_data, db_data);

  // Computes dL/dX.
  GroupNormBackwardCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         stream>>>(
          size,
          G,
          D,
          HxW,
          dY_data,
          X_data,
          mu_data,
          rsig_data,
          gamma_data,
          ds_data,
          db_data,
          dX_data);

  // Computes dL/dgamma and dL/dbeta.
  GammaBetaBackwardCUDAKernel<T>
      <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         stream>>>(
          N,
          G,
          D,
          HxW,
          dY_data,
          X_data,
          mu_data,
          rsig_data,
          dgamma_data,
          dbeta_data);
}

} // namespace cuda

inline void GroupNormForward(float eps,
                             const int N,
                             const int G,
                             const int D,
                             const int HxW,
                             const Tensor<gpu, 1> &X,
                             const Tensor<gpu, 1> &gamma,
                             const Tensor<gpu, 1> &beta,
                             Tensor<gpu, 1> &Y,
                             Tensor<gpu, 1> &mu,
                             Tensor<gpu, 1> &rsig) {
  cudaStream_t stream = Stream<gpu>::GetStream(Y.stream_);
  const float *X_data = X.dptr_;
  const float *gamma_data = gamma.dptr_;
  const float *beta_data = beta.dptr_;
  float *Y_data = Y.dptr_;
  float *mu_data = mu.dptr_;
  float *rsig_data = rsig.dptr_;
  cuda::GroupNormForward<float>(stream, eps, N, G, D, HxW, X_data, gamma_data, beta_data, Y_data, mu_data, rsig_data);
}

inline void GroupNormBackward(const int N,
                              const int G,
                              const int D,
                              const int HxW,
                              const Tensor<gpu, 1> &dY,
                              const Tensor<gpu, 1> &X,
                              const Tensor<gpu, 1> &mu,
                              const Tensor<gpu, 1> &rsig,
                              const Tensor<gpu, 1> &gamma,
                              Tensor<gpu, 1> &ds,
                              Tensor<gpu, 1> &db,
                              Tensor<gpu, 1> &dX,
                              Tensor<gpu, 1> &dgamma,
                              Tensor<gpu, 1> &dbeta) {
  cudaStream_t stream = Stream<gpu>::GetStream(dX.stream_);
  const float *dY_data    = dY.dptr_;
  const float *X_data     = X.dptr_;
  const float *mu_data    = mu.dptr_;
  const float *rsig_data  = rsig.dptr_;
  const float *gamma_data = gamma.dptr_;
  float *ds_data          = ds.dptr_;
  float *db_data          = db.dptr_;
  float *dX_data          = dX.dptr_;
  float *dgamma_data      = dgamma.dptr_;
  float *dbeta_data       = dbeta.dptr_;
  cuda::GroupNormBackward<float>(stream, N, G, D, HxW, dY_data, X_data, mu_data, rsig_data, gamma_data, ds_data,
                                 db_data, dX_data, dgamma_data, dbeta_data);
}

} // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(GroupNormParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GroupNormOp<gpu>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
