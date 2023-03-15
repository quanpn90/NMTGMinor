#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>
#include <cmath>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cublasLt.h>

// Keep Sigmoid in float only. When using half, cast to float before calling.
__device__ __inline__ float sigmoid(float a) {
  float retf = 1.f / (1.f + expf(-a));
  return (retf);
}

// Sigmoid. Assume input X is [features x batch size], column major.
// Safe to call in-place.
template <typename T>
__global__ void Sigmoid_fprop(T *X, uint batch_size, uint features) {
  T r_x[ILP];
  if(is_aligned(X) && features % ILP ==0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      load_store(r_x, X, 0 , tid);
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        r_x[ii] = sigmoid(static_cast<float>(r_x[ii]));
      }
      load_store(X, r_x, tid , 0);
    }
  } else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid < features * batch_size; tid += ILP * blockDim.x * gridDim.x) {
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          r_x[ii] = X[idx];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        r_x[ii] = sigmoid(static_cast<float>(r_x[ii]));
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          X[idx] = r_x[ii];
        }
      }
    }
  }
}

// Sigmoid. Assume input X is [features x batch size], column major.
// Safe to call in-place.
template <typename T>
__global__ void Sigmoid_bprop(T *dY, T *Y, uint batch_size, uint features, T *dX) {
  T r_dy[ILP];
  T r_y[ILP];
  if(is_aligned(dY) &&
     is_aligned(Y) &&
     is_aligned(dX) &&
     features % ILP ==0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      load_store(r_dy, dY, 0 , tid);
      load_store(r_y, Y, 0 , tid);
#pragma unroll
      for(int ii=0;ii<ILP;ii++){
        float grad_out = r_dy[ii];
        float out = r_y[ii];
        float grad_i = out * ( 1.f - out) * grad_out;
        r_dy[ii] = grad_i;
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
          r_y[ii] = Y[idx];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float grad_out = r_dy[ii];
        float out = r_y[ii];
        float grad_i = out * ( 1.f - out) * grad_out;
        r_dy[ii] = grad_i;
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

// FP64 Wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    double* A,
    int lda,
    double* B,
    int ldb,
    const float* beta,
    double* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_64F,
      lda,
      B,
      CUDA_R_64F,
      ldb,
      beta,
      C,
      CUDA_R_64F,
      ldc,
      CUDA_R_64F,
      CUBLAS_GEMM_DEFAULT);
}

// FP32 Wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    float* A,
    int lda,
    float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_32F,
      lda,
      B,
      CUDA_R_32F,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float* beta,
    at::Half* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      beta,
      C,
      CUDA_R_16F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600


int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float *beta, /* host pointer */
    at::Half* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS)
    goto CLEANUP;


  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}







int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    double* A,
    int lda,
    double* B,
    int ldb,
    const float *beta, /* host pointer */
    double* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  return 1;
}

int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta, /* host pointer */
    float *C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_32F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }

  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          &heuristicResult.algo,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}


int gemm_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float *beta, /* host pointer */
    at::Half* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    void* bgrad) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad));
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Fail 1");
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("Fail 2");
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("Can't find workspace.\n");
    goto CLEANUP;
  }

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("can't find algorithm\n");
    goto CLEANUP;
  }

  if (returnedResults == 0) {
    printf("CUBLAS_STATUS_NOT_SUPPORTED");
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}







int gemm_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    double* A,
    int lda,
    double* B,
    int ldb,
    const float *beta, /* host pointer */
    double* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    void* bgrad) {
  return 1;
}

int gemm_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta, /* host pointer */
    float *C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    void* bgrad) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_32F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }

  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          &heuristicResult.algo,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}


#endif

template <typename T>
int linear_bias_forward_cuda(at::Tensor input, T *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    const float beta_one       = 1.0;
    int status = 1;
    status = gemm_bias_lt(
        (cublasLtHandle_t)handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch_size,
        in_features,
        &alpha, /* host pointer */
        weight,
        in_features,
        input.data_ptr<T>(),
        in_features,
        &beta_zero, /* host pointer */
        output.data_ptr<T>(),
        out_features,
        lt_workspace,
        1 << 22,
        stream,
        true,
        static_cast<const void*>(bias.data_ptr<T>()));

    if (status != 0){
//         printf("GEMM BIAS LT not available. Backoff to GEMM and manual bias.");
        return 1;
    }
    return status;
}


template <typename T>
int linear_bias_forward_sigmoid_cuda(at::Tensor input, T *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    const float beta_one       = 1.0;
    int status = 1;
    status = gemm_bias_lt(
        (cublasLtHandle_t)handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch_size,
        in_features,
        &alpha, /* host pointer */
        weight,
        in_features,
        input.data_ptr<T>(),
        in_features,
        &beta_zero, /* host pointer */
        output.data_ptr<T>(),
        out_features,
        lt_workspace,
        1 << 22,
        stream,
        true,
        static_cast<const void*>(bias.data_ptr<T>()));

    if (status != 0){
//         printf("GEMM BIAS LT not available. Backoff to GEMM and manual bias.");
        return 1;
    }
    return status;
}


template <typename T>
int linear_bias_backward_cuda(bool compute_grad_input, T *input, T *weight, T *d_output, int in_features, int batch_size, int out_features, T *d_weight, T *d_bias, T *d_input,  void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    int status = 1;

    status = gemm_bgradb_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    in_features,
    out_features,
    batch_size,
    &alpha, /* host pointer */
    input,
    in_features,
    d_output,
    out_features,
    &beta_zero, /* host pointer */
    d_weight,
    in_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    static_cast<void*>(d_bias));

    if (status != 0){
        return 1;
    }

    if (compute_grad_input)
        status = gemm_bias(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          in_features,
          batch_size,
          out_features,
          &alpha,
          weight,
          in_features,
          d_output,
          out_features,
          &beta_zero,
          d_input,
          in_features);

    return status;

}



template <typename T>
int linear_bias_backward_input_only_cuda(T *input, T *weight, T *d_output, int in_features, int batch_size, int out_features,
    T *d_input,  void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    int status = 1;

    status = gemm_bias(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      in_features,
      batch_size,
      out_features,
      &alpha,
      weight,
      in_features,
      d_output,
      out_features,
      &beta_zero,
      d_input,
      in_features);

    return status;

}

// FORWARD
template int linear_bias_forward_cuda<at::Half>(at::Tensor input, at::Half *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template int linear_bias_forward_cuda<float>(at::Tensor input, float *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template int linear_bias_forward_cuda<double>(at::Tensor input, double *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

// FORWARD with sigmoid
template int linear_bias_sigmoid_forward_cuda<at::Half>(at::Tensor input, at::Half *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template int linear_bias_sigmoid_forward_cuda<float>(at::Tensor input, float *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template int linear_bias_sigmoid_forward_cuda<double>(at::Tensor input, double *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

// BACKWARD
template int linear_bias_backward_cuda<at::Half>(bool compute_grad_input, at::Half *input, at::Half *weight, at::Half *d_output, int in_features, int batch_size, int out_features, at::Half *d_weight, at::Half *d_bias, at::Half *d_input,  void *lt_workspace) ;

template int linear_bias_backward_cuda<float>(bool compute_grad_input, float *input, float *weight, float *d_output, int in_features, int batch_size, int out_features, float *d_weight, float *d_bias, float *d_input,  void *lt_workspace) ;

template int linear_bias_backward_cuda<double>(bool compute_grad_input, double *input, double *weight, double *d_output, int in_features, int batch_size, int out_features, double *d_weight, double *d_bias, double *d_input,  void *lt_workspace) ;

// BACKWARD with sigmoid
template int linear_bias_sigmoid_backward_cuda<at::Half>(bool compute_grad_input, at::Half *input, at::Half *weight, at::Half *d_output, int in_features, int batch_size, int out_features, at::Half *d_weight, at::Half *d_bias, at::Half *d_input,  void *lt_workspace) ;

template int linear_bias_sigmoid_backward_cuda<float>(bool compute_grad_input, float *input, float *weight, float *d_output, int in_features, int batch_size, int out_features, float *d_weight, float *d_bias, float *d_input,  void *lt_workspace) ;

template int linear_bias_sigmoid_backward_cuda<double>(bool compute_grad_input, double *input, double *weight, double *d_output, int in_features, int batch_size, int out_features, double *d_weight, double *d_bias, double *d_input,  void *lt_workspace) ;

// BACKWARD input only (no weight computation)
template int linear_bias_backward_input_only_cuda<at::Half>(at::Half *input, at::Half *weight, at::Half *d_output, int in_features, int batch_size, int out_features, at::Half *d_input,  void *lt_workspace) ;

template int linear_bias_backward_input_only_cuda<float>(float *input, float *weight, float *d_output, int in_features, int batch_size, int out_features, float *d_input,  void *lt_workspace) ;

template int linear_bias_backward_input_only_cuda<double>(double *input, double *weight, double *d_output, int in_features, int batch_size, int out_features, double *d_input,  void *lt_workspace) ;
