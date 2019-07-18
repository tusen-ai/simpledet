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
#include <algorithm>
#include <vector>
#include <cub/block/block_reduce.cuh>
#include "../../common/cuda_utils.h"


#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

constexpr int CAFFE_CUDA_NUM_THREADS = 512;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;

inline int CAFFE_GET_BLOCKS(const int N) {
  return std::min((N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
                  CAFFE_MAXIMUM_NUM_BLOCKS);
}

template <typename T, int N>
struct SimpleArray {
  T data[N];
};

template <typename DType>
using BlockReduce = cub::BlockReduce<DType, CAFFE_CUDA_NUM_THREADS>;


constexpr int kCUDATensorMaxDims = 8;

#define DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(val, Func, T, ...) \
  do {                                                            \
    CHECK_LT(val, kCUDATensorMaxDims);                    \
    switch (val) {                                                \
      case 1: {                                                   \
        Func<T, 1>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 2: {                                                   \
        Func<T, 2>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 3: {                                                   \
        Func<T, 3>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 4: {                                                   \
        Func<T, 4>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 5: {                                                   \
        Func<T, 5>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 6: {                                                   \
        Func<T, 6>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 7: {                                                   \
        Func<T, 7>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 8: {                                                   \
        Func<T, 8>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      default: { break; }                                         \
    }                                                             \
  } while (false)


void ComputeTransposeAxesForReduceOp(
    const int num_dims,
    const int num_reduce_axes,
    const int* reduce_axes,
    int* transpose_axes);


void ComputeTransposedStrides(
    const int ndim,
    const int* dims,
    const int* axes,
    int* strides);


bool IsRowwiseReduce(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    int* rows,
    int* cols);


bool IsColwiseReduce(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    int* rows,
    int* cols);


template <typename T>
void Set(const size_t N, const T alpha, T* X, cudaStream_t context);


template <typename T>
void Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    cudaStream_t context);

template <typename T>
void InvStd(
    const int N,
    const T epsilon,
    const T* var,
    T* inv_std,
    cudaStream_t context);