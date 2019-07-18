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
#include "./group_norm_helper.h"
#include "./fixed_divisor.h"


template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(cols);
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      const int X_index = i * cols + j;
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const T mu = m_val * scale;
      mean[i] = mu;
      variance[i] = v_val * scale - mu * mu;
    }
    __syncthreads();
  }
}


template <typename T>
__global__ void ColwiseMomentsCUDAKernel(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(rows);
  for (int i = blockIdx.x; i < cols; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < rows; j += blockDim.x) {
      const int X_index = j * cols + i;
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const T mu = m_val * scale;
      mean[i] = mu;
      variance[i] = v_val * scale - mu * mu;
    }
    __syncthreads();
  }
}


template <typename T, int D>
__global__ void MomentsCUDAKernel(
    const int outer_size,
    const int inner_size,
    SimpleArray<int, D> X_strides,
    SimpleArray<FixedDivisor<int>, D> Y_dims,
    const T* X,
    T* mean,
    T* variance) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(inner_size);
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T m_val = 0;
    T v_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      int X_index = 0;
      int Y_index = i * inner_size + j;
#pragma unroll
      for (int d = D - 1; d >= 0; --d) {
        int r;
        Y_dims.data[d].DivMod(Y_index, &Y_index, &r);
        X_index += r * X_strides.data[d];
      }
#if __CUDA_ARCH__ >= 350
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
    }
    m_val = BlockReduce<T>(m_storage).Sum(m_val);
    v_val = BlockReduce<T>(v_storage).Sum(v_val);
    if (threadIdx.x == 0) {
      const T mu = m_val * scale;
      mean[i] = mu;
      variance[i] = v_val * scale - mu * mu;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void SetKernel(const int N, const T alpha, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = alpha;
  }
}

#define CAFFE2_SPECIALIZED_CUDA_SET(T)                              \
  template <>                                                       \
  void Set<T>(                         \
      const size_t N, const T alpha, T* Y, cudaStream_t context) {  \
    if (N == 0) {                                                   \
      return;                                                       \
    }                                                               \
    if (alpha == T(0)) {                                            \
      cudaMemsetAsync(Y, 0, sizeof(T) * N, context);                \
    } else {                                                        \
      SetKernel<T>                                                  \
          <<<CAFFE_GET_BLOCKS(N),                                   \
             CAFFE_CUDA_NUM_THREADS,                                \
             0,                                                     \
             context>>>(N, alpha, Y);                               \
    }                                                               \
  }
CAFFE2_SPECIALIZED_CUDA_SET(float);
#undef CAFFE2_SPECIALIZED_CUDA_SET


void ComputeTransposeAxesForReduceOp(
    const int num_dims,
    const int num_reduce_axes,
    const int* reduce_axes,
    int* transpose_axes) {
  const int d = num_dims - num_reduce_axes;
  std::copy_n(reduce_axes, num_reduce_axes, transpose_axes + d);
  std::sort(transpose_axes + d, transpose_axes + num_dims);
  int p = 0;
  int q = d;
  for (int i = 0; i < num_dims; ++i) {
    if (q < num_dims && i == transpose_axes[q]) {
      ++q;
    } else {
      transpose_axes[p++] = i;
    }
  }
}


void ComputeTransposedStrides(
    const int ndim,
    const int* dims,
    const int* axes,
    int* strides) {
  std::vector<int> buff(ndim);
  int cur_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buff[i] = cur_stride;
    cur_stride *= dims[i];
  }
  for (int i = 0; i < ndim; ++i) {
    strides[i] = buff[axes[i]];
  }
}


bool IsRowwiseReduce(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols) {
  *cols = 1;
  int pivot = ndim - 1;
  for (; pivot >= 0 && B_dims[pivot] == 1; --pivot) {
    *cols *= A_dims[pivot];
  }
  *rows = 1;
  for (int i = pivot; i >= 0; --i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *rows *= A_dims[i];
  }
  return true;
}


bool IsColwiseReduce(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols) {
  *rows = 1;
  int pivot = 0;
  for (; pivot < ndim && B_dims[pivot] == 1; ++pivot) {
    *rows *= A_dims[pivot];
  }
  *cols = 1;
  for (int i = pivot; i < ndim; ++i) {
    if (A_dims[i] != B_dims[i]) {
      return false;
    }
    *cols *= A_dims[i];
  }
  return true;
}


template <typename T>
__global__ void
InvStdCUDAKernel(const int N, const T epsilon, const T* var, T* inv_std);
#define DELEGATE_INV_STD_KERNEL_FUNCTION(T, Func)               \
  template <>                                                   \
  __global__ void InvStdCUDAKernel<T>(                          \
      const int N, const T epsilon, const T* var, T* inv_std) { \
    CUDA_1D_KERNEL_LOOP(i, N) {                                 \
      inv_std[i] = Func(var[i] + epsilon);                      \
    }                                                           \
  }
DELEGATE_INV_STD_KERNEL_FUNCTION(float, rsqrtf)
#undef DELEGATE_INV_STD_KERNEL_FUNCTION


#define CAFFE2_SPECIALIZED_CUDA_INV_STD(T)                      \
  template <>                                                   \
  void InvStd<T>(                                  \
      const int N,                                              \
      const T epsilon,                                          \
      const T* var,                                             \
      T* inv_std,                                               \
      cudaStream_t context) {                                   \
    InvStdCUDAKernel<T>                                         \
        <<<CAFFE_GET_BLOCKS(N),                                 \
           CAFFE_CUDA_NUM_THREADS,                              \
           0,                                                   \
           context>>>(N, epsilon, var, inv_std); \
  }
CAFFE2_SPECIALIZED_CUDA_INV_STD(float)
#undef CAFFE2_SPECIALIZED_CUDA_INV_STD


template <typename T, int D>
void MomentsCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    cudaStream_t context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FixedDivisor<int>, D> Y_dims;
  ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FixedDivisor<int>(dims[axes[i]]);
  }
  MomentsCUDAKernel<T, D>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context>>>(
          outer_size, inner_size, X_strides, Y_dims, X, mean, variance);
}


template <typename T>
void MomentsCUDA(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* X,
    T* mean,
    T* variance,
    cudaStream_t context) {
  CHECK_LE(num_axes, num_dims);
  std::vector<int> Y_dims_vector(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    Y_dims_vector[axes[i]] = 1;
  }
  const int* X_dims = dims;
  const int* Y_dims = Y_dims_vector.data();
  const int X_size = std::accumulate(X_dims, X_dims + num_dims, 1, std::multiplies<int>());
  const int Y_size = std::accumulate(Y_dims, Y_dims + num_dims, 1, std::multiplies<int>());
  if (X_size == 0) {
    Set<T>(Y_size, T(0), mean, context);
    Set<T>(Y_size, T(0), variance, context);
    return;
  }
  if (std::equal(X_dims, X_dims + num_dims, Y_dims)) {
    cudaMemcpyAsync(
        mean,
        X,
        sizeof(T) * X_size,
        cudaMemcpyDeviceToDevice,
        context);
    Set<T>(Y_size, T(0), variance, context);
    return;
  }
  int rows;
  int cols;
  if (IsRowwiseReduce(num_dims, X_dims, Y_dims, &rows, &cols)) {
    RowwiseMomentsCUDAKernel<T>
        <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context>>>(rows, cols, X, mean, variance);
    return;
  }
  if (IsColwiseReduce(num_dims, X_dims, Y_dims, &rows, &cols)) {
    ColwiseMomentsCUDAKernel<T>
        <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context>>>(rows, cols, X, mean, variance);
    return;
  }
  std::vector<int> transpose_axes(num_dims);
  ComputeTransposeAxesForReduceOp(
      num_dims, num_axes, axes, transpose_axes.data());
  const int pivot = num_dims - num_axes;
  int outer_size = 1;
  for (int i = 0; i < pivot; ++i) {
    outer_size *= dims[transpose_axes[i]];
  }
  int inner_size = 1;
  for (int i = pivot; i < num_dims; ++i) {
    inner_size *= dims[transpose_axes[i]];
  }
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      num_dims,
      MomentsCUDAImpl,
      T,
      outer_size,
      inner_size,
      dims,
      transpose_axes.data(),
      X,
      mean,
      variance,
      context);
}


#define CAFFE2_SPECIALIZED_CUDA_MOMENTS(T)                           \
  template <>                                                        \
  void Moments<T>(                                                   \
      const int num_dims,                                            \
      const int* dims,                                               \
      const int num_axes,                                            \
      const int* axes,                                               \
      const T* X,                                                    \
      T* mean,                                                       \
      T* variance,                                                   \
      cudaStream_t context) {                                        \
    MomentsCUDA<T>(                                                  \
        num_dims, dims, num_axes, axes, X, mean, variance, context); \
  }
CAFFE2_SPECIALIZED_CUDA_MOMENTS(float)
#undef CAFFE2_SPECIALIZED_CUDA_MOMENTS