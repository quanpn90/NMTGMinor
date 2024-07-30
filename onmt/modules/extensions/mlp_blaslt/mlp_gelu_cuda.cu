#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>
#include <cmath>
#include <math.h>

/* Includes, cuda */
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include <curand_kernel.h>

// includes cublaslt
#include <cublasLt.h>
#include "cublaslt_wrapper.cuh"

// constants for fused bias+relu kernel
#define BIAS_RELU_FW_NTHREADS 256 // forward number of thread per block
#define BIAS_RELU_BW_NTHREADS_X 32 // backward number of thread in feature dim
#define BIAS_RELU_BW_NTHREADS_Y 16 // backward number of thread in batch dim
#define BIAS_RELU_RED_PER_THREAD 16 // backward minimal reduction length per thread

// move to a header later on
#define ILP 4
#define BACKCOEFF M_2_SQRTPI * M_SQRT1_2 * 0.5f
template<typename T>
__host__ __device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}
template<typename T>
__device__ __forceinline__ void load_store(T* dst, volatile T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}
template<typename T>
__device__ __forceinline__ void load_store(volatile T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

// Keep ReLU in float only. When using half, cast to float before calling.
__device__ __inline__ float relu(float a) {
  float retf = max(a, 0.f);
  return (retf);
}


// Keep gelu in float only. When using half, cast to float before calling.
__device__ __inline__ float gelu(float a) {
  float retf = a * normcdff(a);
  return (retf);
}


// Keep gelu in float only. When using half, cast to float before calling.
__device__ __inline__ float gelu_back(float dy, float a) {

  // dy is the gradient w.r.t the gelu output
  float cdf = normcdff(a);
  float pdf = BACKCOEFF * expf(-0.5f * a * a);
  float retf = cdf + a * pdf;

  return (dy * retf);
}




// FP64 Wrapper around cublas GEMMEx
cublasStatus_t mlp_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    const float beta,
    double* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      &alpha,
      A,
      CUDA_R_64F,
      lda,
      B,
      CUDA_R_64F,
      ldb,
      &beta,
      C,
      CUDA_R_64F,
      ldc,
      CUDA_R_64F,
      CUBLAS_GEMM_DEFAULT);
}

// FP32 Wrapper around cublas GEMMEx
cublasStatus_t mlp_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float beta,
    float* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      &alpha,
      A,
      CUDA_R_32F,
      lda,
      B,
      CUDA_R_32F,
      ldb,
      &beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t mlp_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha,
    const at::Half* A,
    int lda,
    const at::Half* B,
    int ldb,
    float beta,
    at::Half* C,
    int ldc) {

  const half halpha = __float2half_rn(alpha);
  const half hbeta = __float2half_rn(beta);
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      &halpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      &hbeta,
      C,
      CUDA_R_16F,
      ldc,
      CUBLAS_COMPUTE_16F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

///////// CUBLAS LT FUNCTIONS /////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////

// Bias ADD + ReLU. Assume input X is [features x batch size], column major.
// Activation support fuesed ReLU. Safe to call in-place.
template <typename T>
__global__ void DropoutGeLU_fprop(T *X, T *Y, uint8_t *mask, uint batch_size, uint features, float p,
                                         std::pair<uint64_t, uint64_t> seeds) {
  T r_x[ILP];
  T r_y[ILP];
  uint8_t r_m[ILP];
//  uint8_t r_m[ILP];
  float pinv = 1.f/(1.f-p);

  if(is_aligned(X) && is_aligned(Y) && features % ILP ==0) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(
        seeds.first,
        tid,
        seeds.second,
        &state);

    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      load_store(r_x, X, 0 , tid);
      load_store(r_y, Y, 0 , tid);
      load_store(r_m, mask, 0, tid);  // mask has the same size with X

      float4 rand = curand_uniform4(&state);
      rand.x = rand.x >= p;
      rand.y = rand.y >= p;
      rand.z = rand.z >= p;
      rand.w = rand.w >= p;

#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]);
        r_y[ii] = bias_sum;  // store the mm + bias output
        r_x[ii] = gelu(bias_sum) * (float)(&rand.x)[ii]*pinv;  // gelu * dropout mask
        r_m[ii] = (uint8_t)(&rand.x)[ii];  // store the mask values in buffer
      }
      load_store(X, r_x, tid , 0);
      load_store(mask, r_m, tid , 0);
      load_store(Y, r_y, tid , 0);
    }
  } else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(
        seeds.first,
        tid,
        seeds.second,
        &state);

    for (; tid < features * batch_size; tid += ILP * blockDim.x * gridDim.x) {

      float4 rand = curand_uniform4(&state);
      rand.x = rand.x >= p;
      rand.y = rand.y >= p;
      rand.z = rand.z >= p;
      rand.w = rand.w >= p;
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          r_x[ii] = X[idx];
          r_m[ii] = mask[idx];
          r_y[ii] = Y[idx];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]);
        r_y[ii] = bias_sum;
        r_x[ii] = gelu(bias_sum)*(float)(&rand.x)[ii]*pinv;
        r_m[ii] = (uint8_t)(&rand.x)[ii];
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          X[idx] = r_x[ii];
          mask[idx] = r_m[ii];
          Y[idx] = r_y[ii];
        }
      }
    }
  }
}

template <typename T>
__global__ void GeLU_fprop(T *X, T *Y, uint batch_size, uint features) {
  T r_x[ILP];
  T r_y[ILP];

  if(is_aligned(X) && is_aligned(Y) && features % ILP ==0) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      load_store(r_x, X, 0 , tid);
      load_store(r_y, Y, 0 , tid);

#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]);
        //r_x[ii] = relu(bias_sum)*(&rand.x)[ii]*pinv;
        r_y[ii] = bias_sum;  // store the mm + bias output
        r_x[ii] = gelu(bias_sum);  // gelu * dropout mask
      }
      load_store(X, r_x, tid , 0);
      load_store(Y, r_y, tid , 0);
    }
  } else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid < features * batch_size; tid += ILP * blockDim.x * gridDim.x) {

#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          r_x[ii] = X[idx];
          r_y[ii] = Y[idx];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]);
        r_y[ii] = bias_sum;
        r_x[ii] = gelu(bias_sum);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          X[idx] = r_x[ii];
          Y[idx] = r_y[ii];
        }
      }
    }
  }
}


// Bias ADD + ReLU. Assume input X is [features x batch size], column major.
// Activation support fuesed ReLU. Safe to call in-place.
template <typename T>
__global__ void DropoutGeLU_presampled_fprop(T *X, T *Y, uint8_t *mask, uint batch_size, uint features, float p) {
  T r_x[ILP];
  T r_y[ILP];
  uint8_t r_m[ILP];
//  uint8_t r_m[ILP];
  float pinv = 1.f/(1.f-p);

  if(is_aligned(X) && is_aligned(Y) && features % ILP ==0) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      load_store(r_x, X, 0 , tid);
      load_store(r_y, Y, 0 , tid);
      load_store(r_m, mask, 0, tid);  // mask has the same size with X

#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]);
        r_y[ii] = bias_sum;  // store the mm + bias output
        r_x[ii] = gelu(bias_sum) * (float)(r_m[ii]) *pinv;  // gelu * dropout mask
      }
      load_store(X, r_x, tid , 0);  // X stores gelu output
      load_store(Y, r_y, tid , 0);  // Y stores gelu input
    }
  } else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid < features * batch_size; tid += ILP * blockDim.x * gridDim.x) {
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          r_x[ii] = X[idx];
          r_m[ii] = mask[idx];
          r_y[ii] = Y[idx];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]);
        r_y[ii] = bias_sum;
        r_x[ii] = gelu(bias_sum)*(float)(r_m[ii])*pinv;
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          X[idx] = r_x[ii];
          Y[idx] = r_y[ii];
        }
      }
    }
  }
}


//////////////////////// BACKPROP///////////////////////////////////////


// ReLU. Assume input X is [features x batch size], column major.
// Safe to call in-place.
template <typename T>
__global__ void Gelu_bprop(T *dY, T* H, T *Y, uint features, uint batch_size, T *dX) {
  T r_dy[ILP];
//  T r_y[ILP];
  T r_h[ILP];
  if(is_aligned(dY) &&
     is_aligned(Y) &&
     is_aligned(H) &&
     is_aligned(dX) &&
     features % ILP ==0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      load_store(r_dy, dY, 0 , tid);
//      load_store(r_y, Y, 0 , tid);
      load_store(r_h, H, 0 , tid);
#pragma unroll
      for(int ii=0;ii<ILP;ii++){
          r_dy[ii] = gelu_back((float)r_dy[ii], (float)r_h[ii]);
      }
      load_store(dX, r_dy, tid, 0);
    }
  } else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid < features * batch_size; tid += ILP * blockDim.x * gridDim.x) {
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          r_dy[ii] = dY[idx];
//          r_y[ii] = Y[idx];
          r_h[ii] = H[idx];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        r_dy[ii] = gelu_back((float)r_dy[ii], (float)r_h[ii]);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          dX[idx] = r_dy[ii];
        }
      }
    }
  }
}


// ReLU. Assume input X is [features x batch size], column major.
// Safe to call in-place.
template <typename T>
__global__ void GeluDropout_bprop(T *dY, T* H, T *Y, uint8_t* mask, uint features, uint batch_size, T *dX, float p) {
  T r_dy[ILP];
//  T r_y[ILP];
  T r_h[ILP];
  uint8_t r_m[ILP];
  float pinv = 1.0f / (1.0f - p);
  if(is_aligned(dY) &&
     is_aligned(Y) &&
     is_aligned(H) &&
     is_aligned(dX) &&
     features % ILP ==0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      load_store(r_dy, dY, 0 , tid);
//      load_store(r_y, Y, 0 , tid);
      load_store(r_h, H, 0 , tid);
      load_store(r_m, mask, 0 , tid);
#pragma unroll
      for(int ii=0;ii<ILP;ii++){
            r_dy[ii] = gelu_back((float)r_dy[ii] * (float)r_m[ii] * pinv, (float)r_h[ii]);
      }
      load_store(dX, r_dy, tid, 0);
    }
  } else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid < features * batch_size; tid += ILP * blockDim.x * gridDim.x) {
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          r_dy[ii] = dY[idx];
//          r_y[ii] = Y[idx];
          r_h[ii] = H[idx];
          r_m[ii] = mask[idx];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        r_dy[ii] = gelu_back( (float)r_dy[ii] * (float)r_m[ii] * pinv, (float)r_h[ii] );
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          dX[idx] = r_dy[ii];
        }
      }
    }
  }
}




// Lists where the num_layers-1 intermediate Y buffers start in reserved space on fprop, starting
// offset 0. The last Y value is, of course, stored in the user provided output buffer.
void get_y_offsets(
    int batch_size,
    int num_layers,
    const int* output_features,
    int* y_start_offsets) {
  y_start_offsets[0] = 0;
  for (int i = 1; i < num_layers; i++) {
    y_start_offsets[i] = y_start_offsets[i - 1] + batch_size * output_features[i - 1];
  }
}

// Returns the size of all fprop activations combined
size_t get_all_activations_size(int64_t batch_size, int num_layers, const int* output_features) {
  size_t acts_size = 0;
  for (int l = 0; l < num_layers; l++) {
    acts_size += output_features[l] * batch_size;
  }
  return acts_size;
}

// Returns the reserved space (in elements) needed for the MLP
size_t get_mlp_reserved_space(int64_t batch_size, int num_layers, const int* output_features) {
  size_t res_space = 0;
  // Need to store output of every intermediate MLP - size equal to output_features[i] * batch_size
  // for all 'i' in [0, num_layers-1)
  for (int l = 0; l < num_layers - 1; l++) {
    res_space += output_features[l] * batch_size;
  }
  return res_space;
}

// Returns the size of all fprop activations combined
// no dropout and activation at the last layer so no need that one
size_t get_mlp_activation_space(int64_t batch_size, int num_layers, const int* output_features) {
  size_t acts_size = 0;
  for (int l = 0; l < num_layers - 1; l++) {
    acts_size += output_features[l] * batch_size;
  }
  return acts_size;
}

#if 0
// Returns the work space (in elements) needed for the MLP bprop.
size_t get_mlp_bp_workspace (int batch_size, int num_layers, const int* output_features) {
    /*
       Workspace is partitioned as
       DY_GEMMs : DX_GEMMs
    */
    size_t work_space = 0;

    // Store each intermediate dY explicitly. Need 2 dYs per MLP layer (one for o/p
    // of biasReLU_bp and one for o/p of dgrad GEMM).
    work_space += 2*get_all_activations_size(batch_size, num_layers, output_features);

    return work_space;
}
#endif

// Returns the work space (in elements) needed for the MLP bprop.
template <typename T>
size_t get_mlp_bp_workspace_in_bytes(int batch_size, int num_layers, const int* output_features) {
  size_t work_space = 0;

  // Store each intermediate dY explicitly. Need 2 dYs per MLP layer (one for o/p
  // of biasReLU_bp and one for o/p of dgrad GEMM).
  work_space += 2 * get_all_activations_size(batch_size, num_layers, output_features) * sizeof(T);

  return work_space;
}

// Returns pointers to each segment of the workspace
template <typename T>
void partition_mlp_bp_workspace(
    int batch_size,
    int num_layers,
    const int* output_features,
    void* work_space,
    T** dy_gemms,
    T** dx_gemms) {
  /*
     Workspace is partitioned as
     DY_GEMMs : DX_GEMMs : DB_SCRATCH : SEMAPHORES
  */
  // Start address where dy_gemm tensors are stored
  *dy_gemms = reinterpret_cast<T*>(work_space);
  // Start address where dx_gemm tensors are stored
  *dx_gemms = *dy_gemms + get_all_activations_size(batch_size, num_layers, output_features);
  // Start address where db intermediate tensors are stored

  return;
}

// Does a simple MLP fprop (GEMM+bias+ReLU).
// Can handle num_layers number of layers, each with its own shape. Output of layer i is assumed
// to be input of layer i+1. output_features, WPtr and BPtr are arrays of length num_layers, and
// must be in the same order i.e. WPtr[i] and BPtr[i] are respectively the weight and bias of layer
// 'i'.
template <typename T>
int mlp_fp(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    T** BPtr,
    int num_layers,
    int* output_features,
    T* Y,
    T* reserved_space,
    T* reserved_activations,
    uint8_t* reserved_mask,
    void* lt_workspace,
    float p) {
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  T *weight, *input, *output, *hidden, *bias;
  uint8_t *mask,  *reserved_space_m;
  T *reserved_space_x, *reserved_space_y, *reserved_space_a;
  reserved_space_x = NULL;
  reserved_space_a = reserved_activations;
  reserved_space_y = reserved_space;
  reserved_space_m = reserved_mask;

  // Get cublas handle from Pytorch
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // Get the strea* from cublas handle to reuse for biasReLU kernel.
  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  for (int layer = 0; layer < num_layers; layer++) {
    weight = WPtr[layer];
    input = (layer == 0) ? X : reserved_space_x;
    output = (layer == num_layers - 1) ? Y : reserved_space_y;  // after activation/dropout
    mask = (layer == num_layers - 1 || p == 0) ? NULL : reserved_space_m;
    hidden = (layer == num_layers - 1) ? NULL : reserved_space_a; // before activation/dropout
    bias = BPtr[layer];
    int ifeat = (layer == 0) ? input_features : output_features[layer - 1];
    int ofeat = output_features[layer];

    float one = 1.f;
    float zero = 0.f;

    int cublas_status;
    // Call GEMM: fprop is Y = W'X

    // gelu is only applied in the middle layers and without dropout
    bool use_gelu = (layer < (num_layers - 1) && p == 0);

    if (use_gelu)
        cublas_status = gemm_bias_gelu_lt(
            (cublasLtHandle_t)handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            ofeat,
            batch_size,
            ifeat,
            &one, /* host pointer */
            weight,
            ifeat,
            input,
            ifeat,
            &zero, /* host pointer */
            output,
            ofeat,
            lt_workspace,
            1 << 22,
            stream,
            true,
            static_cast<const void*>(hidden),
            static_cast<const void*>(bias));

    else
        cublas_status = gemm_bias_lt(
            (cublasLtHandle_t)handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            ofeat,
            batch_size,
            ifeat,
            &one, /* host pointer */
            weight,
            ifeat,
            input,
            ifeat,
            &zero, /* host pointer */
            output,
            ofeat,
            lt_workspace,
            1 << 22,
            stream,
            true,
            static_cast<const void*>(bias));

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("GEMM fprop failed with %d\n", cublas_status);
        return 1;
    }

    const uint &input_size = ofeat;
    int num_blocks = 0;
    int num_SMs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    // Call biasReLU
    if (layer == (num_layers -1)) { // no activation
      // do nothing here
    } else {  // GELU
      if (p == 0) {
           // gelu has already been applied at mlp_gemm
//         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, GeLU_fprop<T>, BIAS_RELU_FW_NTHREADS, 0);
//         GeLU_fprop<<<num_SMs*num_blocks, BIAS_RELU_FW_NTHREADS, 0, stream>>>(output, hidden,
//                                                                                     batch_size, input_size);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, DropoutGeLU_fprop<T>, BIAS_RELU_FW_NTHREADS, 0);
        //number of times random will be generated per thread, to offset philox counter in thc random state
        int64_t counter_offset = ((input_size*batch_size-1)/(BIAS_RELU_FW_NTHREADS*num_SMs*num_blocks*ILP)+1)*ILP;
        std::pair<uint64_t, uint64_t> rng_engine_inputs;
        {
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(counter_offset);

        }
        // actually its gelu -> dropout
        DropoutGeLU_fprop<<<num_SMs*num_blocks, BIAS_RELU_FW_NTHREADS, 0,
                             stream>>>(output, hidden, mask, batch_size, input_size, p, rng_engine_inputs);
      }

    }

    // Set current output (after activation) as next layer input
    reserved_space_x = reserved_space_y;
    // Set next layer output

    if (layer < (num_layers -1)) {
        reserved_space_y += ofeat * batch_size;
        reserved_space_a += ofeat * batch_size;
        if (p > 0.0)
            reserved_space_m += ofeat * batch_size;
    }
  }

  return 0;
}

// Does a simple MLP bprop (GEMM+bias+ReLU).
// Needs reserved space to come back exactly as it was populated in fprop.
// Does dgrad and wgrad sequentially.
template <typename T>
int mlp_bp(
    T* X,
    T* Y,
    int input_features,
    int batch_size,
    T** WPtr,
    T** BPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    T* reserved_activations,
    uint8_t* reserved_mask,
    T* work_space,
    T* dX,
    T** dwPtr,
    T** dbPtr,
    bool requires_grad,
    void* lt_workspace,
    float p,
    bool recompute) {
  T *weight, *input, *output, *hidden, *bias;
  T *dweight, *dx, *dy, *dbias;
  T *x, *y, *h;
  uint8_t *mask;
  float one = 1.f;
  float zero = 0.f;
  int cublas_status;
  int xfeat, yfeat, ofeat, ifeat;

  // if recompute: reserved space, reserved activation have to be recompute from X, Weights and biases

  // Get cublas handle from Pytorch
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // Get the stream from cublas handle to reuse for biasReLU kernel.
  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  if (recompute)    {
    uint8_t *mask,  *reserved_space_m;
    T *reserved_space_x, *reserved_space_y, *reserved_space_a;
    reserved_space_x = NULL;
    reserved_space_a = reserved_activations;
    reserved_space_y = reserved_space;
    reserved_space_m = reserved_mask;

    for (int layer = 0; layer < num_layers; layer++) {
        weight = WPtr[layer];
        input = (layer == 0) ? X : reserved_space_x;
        output = (layer == num_layers - 1) ? Y : reserved_space_y;  // after activation/dropout
        mask = (layer == num_layers - 1 || p == 0) ? NULL : reserved_space_m;
        hidden = (layer == num_layers - 1) ? NULL : reserved_space_a; // before activation/dropout
        bias = BPtr[layer];
        ifeat = (layer == 0) ? input_features : output_features[layer - 1];
        ofeat = output_features[layer];

        bool use_gelu = (layer < (num_layers - 1) && p == 0);

        if (use_gelu)
            cublas_status = gemm_bias_gelu_lt(
                (cublasLtHandle_t)handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                ofeat,
                batch_size,
                ifeat,
                &one, /* host pointer */
                weight,
                ifeat,
                input,
                ifeat,
                &zero, /* host pointer */
                output,
                ofeat,
                lt_workspace,
                1 << 22,
                stream,
                true,
                static_cast<const void*>(hidden),
                static_cast<const void*>(bias));

        else
            cublas_status = gemm_bias_lt(
                (cublasLtHandle_t)handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                ofeat,
                batch_size,
                ifeat,
                &one, /* host pointer */
                weight,
                ifeat,
                input,
                ifeat,
                &zero, /* host pointer */
                output,
                ofeat,
                lt_workspace,
                1 << 22,
                stream,
                true,
                static_cast<const void*>(bias));

        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("GEMM fprop failed with %d\n", cublas_status);
            return 1;
        }

        const uint &input_size = ofeat;
        int num_blocks = 0;
        int num_SMs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

        if (layer == (num_layers -1)) { // no activation
          // do nothing here
        } else {  // GELU
          if (p == 0) {
               // gelu has already been applied at mlp_gemm
          } else {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, DropoutGeLU_presampled_fprop<T>,
                                                          BIAS_RELU_FW_NTHREADS, 0);
            //number of times random will be generated per thread, to offset philox counter in thc random state
            int64_t counter_offset = ((input_size*batch_size-1)/(BIAS_RELU_FW_NTHREADS*num_SMs*num_blocks*ILP)+1)*ILP;
            // actually its gelu -> dropout
            DropoutGeLU_presampled_fprop<<<num_SMs*num_blocks, BIAS_RELU_FW_NTHREADS, 0,
                                 stream>>>(output, hidden, mask, batch_size, input_size, p);
          }
        }

        // Set current output (after activation) as next layer input
        reserved_space_x = reserved_space_y;
        // Set next layer output

        if (layer < (num_layers -1)) {
            reserved_space_y += ofeat * batch_size;
            reserved_space_a += ofeat * batch_size;
            if (p > 0.0)
                reserved_space_m += ofeat * batch_size;
        }

    }

  }

  // Backward starts

  // Where the dx of the biasReLU (== dy of gemm) is stored. Can be thrown away
  // after bp call.
  T* dy_gemm_base;
  // Where the dx after GEMM is stored.
  T* dx_gemm_base;

  partition_mlp_bp_workspace<T>(
      batch_size,
      num_layers,
      output_features,
      work_space,
      &dy_gemm_base,
      &dx_gemm_base);

  int* y_offsets = (int*)malloc(num_layers * sizeof(int));
  get_y_offsets(batch_size, num_layers, output_features, y_offsets);

  for (int layer = num_layers - 1; layer >= 0; layer--) {
    weight = WPtr[layer];
    dweight = dwPtr[layer];

    // x is read from reserved space
    x = (layer == 0) ? X : reserved_space + y_offsets[layer - 1];  // gemm + bias output

    // dx is written in workspace for all but layer==0
    dx = (layer == 0) ? dX : dx_gemm_base + y_offsets[layer - 1];

    // y is read from reserved space
    y = (layer == num_layers - 1) ? Y : reserved_space + y_offsets[layer];

    // note: last layer doesn't have h and mask
    h = (layer == num_layers - 1) ? NULL : reserved_activations + y_offsets[layer];  // activation + dropout output
    mask = ((layer == num_layers - 1) || (p == 0.0)) ? NULL : reserved_mask + y_offsets[layer];  // mask

    // dx from layer+1
    dy = (layer == num_layers - 1) ? dY : dx_gemm_base + y_offsets[layer];
    // dy_gemm is written to and read immediately
    T* dy_gemm = dy_gemm_base + y_offsets[layer];

    dbias = dbPtr[layer];
    xfeat = (layer == 0) ? input_features : output_features[layer - 1];
    yfeat = output_features[layer];

    if (layer == (num_layers - 1)) { // no activation
        // Don't need to do anything
        // bypass dgrad through reset pointer
        dy_gemm = dy;
    } else  { // gelu
        int num_blocks = 0;
        int num_SMs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

        if (p == 0) {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, Gelu_bprop<T>, BIAS_RELU_FW_NTHREADS, 0);
            Gelu_bprop<<<num_SMs*num_blocks, BIAS_RELU_FW_NTHREADS, 0, stream>>>(dy, h, y, yfeat, batch_size, dy_gemm);
        } else {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, GeluDropout_bprop<T>, BIAS_RELU_FW_NTHREADS, 0);
            GeluDropout_bprop<<<num_SMs*num_blocks, BIAS_RELU_FW_NTHREADS, 0, stream>>>(dy, h, y, mask, yfeat,
                                                                                       batch_size, dy_gemm, p);
        }
    }
    cublasStatus_t cublas_status;
    // Call GEMM dgrad
    if (layer > 0 || requires_grad == 1) {
      cublas_status = mlp_gemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        xfeat,
        batch_size,
        yfeat,
        one,
        weight,
        xfeat,
        dy_gemm,
        yfeat,
        zero,
        dx,
        xfeat);

      if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("GEMM dgrad failed with %d\n", cublas_status);
        return 1;
      }
    }

    // Call GEMM wgrad and bgrad
    int cublaslt_status_ = 1;
    cublaslt_status_ = gemm_bgradb_lt(
        (cublasLtHandle_t)handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        xfeat,
        yfeat,
        batch_size,
        &one, /* host pointer */
        x,
        xfeat,
        dy_gemm,
        yfeat,
        &zero, /* host pointer */
        dweight,
        xfeat,
        lt_workspace,
        1 << 22,
        stream,
        true,
        static_cast<void*>(dbias));

    if (cublaslt_status_ != CUBLAS_STATUS_SUCCESS) {
      printf("GEMM wgrad failed with %d\n", cublas_status);
      return 1;
    }
  }

  return 0;
}


template <typename T>
int mlp_bp_input_only(
    T* X,
    T* Y,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    T* reserved_activations,
    uint8_t* reserved_mask,
    T* work_space,
    T* dX,
    bool requires_grad,
    void* lt_workspace,
    float p) {
  T* weight;
  T *dx, *dy;
  T *x, *y, *h;
  uint8_t *mask;
//  int activation = 1;

  // Where the dx of the biasReLU (== dy of gemm) is stored. Can be thrown away
  // after bp call.
  T* dy_gemm_base;
  // Where the dx after GEMM is stored.
  T* dx_gemm_base;

  partition_mlp_bp_workspace<T>(
      batch_size,
      num_layers,
      output_features,
      work_space,
      &dy_gemm_base,
      &dx_gemm_base);

  // Get cublas handle from Pytorch
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // Get the stream from cublas handle to reuse for biasReLU kernel.
  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  int* y_offsets = (int*)malloc(num_layers * sizeof(int));
  get_y_offsets(batch_size, num_layers, output_features, y_offsets);

  for (int layer = num_layers - 1; layer >= 0; layer--) {
    weight = WPtr[layer];
//     dweight = dwPtr[layer];

    // x is read from reserved space
    x = (layer == 0) ? X : reserved_space + y_offsets[layer - 1];  // gemm + bias output

    // dx is written in workspace for all but layer==0
    dx = (layer == 0) ? dX : dx_gemm_base + y_offsets[layer - 1];

    // y is read from reserved space
    y = (layer == num_layers - 1) ? Y : reserved_space + y_offsets[layer];

    // note: last layer doesn't have h and mask
    h = (layer == num_layers - 1) ? NULL : reserved_activations + y_offsets[layer];  // activation + dropout output
    mask = ((layer == num_layers - 1) || (p == 0.0)) ? NULL : reserved_mask + y_offsets[layer];  // mask

    // dx from layer+1
    dy = (layer == num_layers - 1) ? dY : dx_gemm_base + y_offsets[layer];
    // dy_gemm is written to and read immediately
    T* dy_gemm = dy_gemm_base + y_offsets[layer];

//     dbias = dbPtr[layer];
    int xfeat = (layer == 0) ? input_features : output_features[layer - 1];
    int yfeat = output_features[layer];

    float one = 1.f;
    float zero = 0.f;

    if (layer == (num_layers -1)) { // no activation

        dy_gemm = dy;
    } else  { // gelu
        int num_blocks = 0;
        int num_SMs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

        if (p == 0) {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, Gelu_bprop<T>, BIAS_RELU_FW_NTHREADS, 0);
            Gelu_bprop<<<num_SMs*num_blocks, BIAS_RELU_FW_NTHREADS, 0, stream>>>(dy, h, y, yfeat, batch_size, dy_gemm);
        } else {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, GeluDropout_bprop<T>, BIAS_RELU_FW_NTHREADS, 0);
            GeluDropout_bprop<<<num_SMs*num_blocks, BIAS_RELU_FW_NTHREADS, 0, stream>>>(dy, h, y, mask, yfeat,
                                                                                       batch_size, dy_gemm, p);
        }
    }
    cublasStatus_t cublas_status;
    // Call GEMM dgrad
    if (layer > 0 || requires_grad == 1) {
      cublas_status = mlp_gemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        xfeat,
        batch_size,
        yfeat,
        one,
        weight,
        xfeat,
        dy_gemm,
        yfeat,
        zero,
        dx,
        xfeat);

      if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("GEMM dgrad failed with %d\n", cublas_status);
        return 1;
      }
    }

    // No need to Call GEMM wgrad and bgrad

  }

  return 0;
}






// Instantiate for floating point types
template int mlp_fp<float>(
    float* X,
    int input_features,
    int batch_size,
    float** WPtr,
    float** BPtr,
    int num_layers,
    int* output_features,
    float* Y,
    float* reserved_space,
    float* reserved_activations,
    uint8_t* reserved_mask,
    void* lt_workspace,
    float p);

template int mlp_bp<float>(
    float* X,
    float* Y,
    int input_features,
    int batch_size,
    float** WPtr,
    float** BPtr,
    int num_layers,
    int* output_features,
    float* dY,
    float* reserved_space,
    float* reserved_activations,
    uint8_t* reserved_mask,
    float* work_space,
    float* dX,
    float** dwPtr,
    float** dbPtr,
    bool requires_grad,
    void* lt_workspace,
    float p,
    bool recompute);

template int mlp_bp_input_only<float>(
    float* X,
    float* Y,
    int input_features,
    int batch_size,
    float** WPtr,
    int num_layers,
    int* output_features,
    float* dY,
    float* reserved_space,
    float* reserved_activations,
    uint8_t* reserved_mask,
    float* work_space,
    float* dX,
    bool requires_grad,
    void* lt_workspace,
    float p);

template int mlp_fp<at::Half>(
    at::Half* X,
    int input_features,
    int batch_size,
    at::Half** WPtr,
    at::Half** BPtr,
    int num_layers,
    int* output_features,
    at::Half* Y,
    at::Half* reserved_space,
    at::Half* reserved_activations,
    uint8_t* reserved_mask,
    void* lt_workspace,
    float p);

template int mlp_bp<at::Half>(
    at::Half* X,
    at::Half* Y,
    int input_features,
    int batch_size,
    at::Half** WPtr,
    at::Half** BPtr,
    int num_layers,
    int* output_features,
    at::Half* dY,
    at::Half* reserved_space,
    at::Half* reserved_activations,
    uint8_t* reserved_mask,
    at::Half* work_space,
    at::Half* dX,
    at::Half** dwPtr,
    at::Half** dbPtr,
    bool requires_grad,
    void* lt_workspace,
    float p,
    bool recompute);

template int mlp_bp_input_only<at::Half>(
    at::Half* X,
    at::Half* Y,
    int input_features,
    int batch_size,
    at::Half** WPtr,
    int num_layers,
    int* output_features,
    at::Half* dY,
    at::Half* reserved_space,
    at::Half* reserved_activations,
    uint8_t* reserved_mask,
    at::Half* work_space,
    at::Half* dX,
    bool requires_grad,
    void* lt_workspace,
    float p);

template int mlp_fp<double>(
    double* X,
    int input_features,
    int batch_size,
    double** WPtr,
    double** BPtr,
    int num_layers,
    int* output_features,
    double* Y,
    double* reserved_space,
    double* reserved_activations,
    uint8_t* reserved_mask,
    void* lt_workspace,
    float p);

template int mlp_bp<double>(
    double* X,
    double* Y,
    int input_features,
    int batch_size,
    double** WPtr,
    double** Btr,
    int num_layers,
    int* output_features,
    double* dY,
    double* reserved_space,
    double* reserved_activations,
    uint8_t* reserved_mask,
    double* work_space,
    double* dX,
    double** dwPtr,
    double** dbPtr,
    bool requires_grad,
    void* lt_workspace,
    float p,
    bool recompute);

template int mlp_bp_input_only<double>(
    double* X,
    double* Y,
    int input_features,
    int batch_size,
    double** WPtr,
    int num_layers,
    int* output_features,
    double* dY,
    double* reserved_space,
    double* reserved_activations,
    uint8_t* reserved_mask,
    double* work_space,
    double* dX,
    bool requires_grad,
    void* lt_workspace,
    float p);




template size_t get_mlp_bp_workspace_in_bytes<float>(
    int batch_size,
    int num_layers,
    const int* output_features);
template size_t get_mlp_bp_workspace_in_bytes<at::Half>(
    int batch_size,
    int num_layers,
    const int* output_features);
template size_t get_mlp_bp_workspace_in_bytes<double>(
    int batch_size,
    int num_layers,
    const int* output_features);
