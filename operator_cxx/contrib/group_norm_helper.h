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