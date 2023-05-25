#include <stdio.h>
/* Includes, cuda */
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#define N 256
#define THREADS_PER_BLOCK 256

__global__ void reverse_array_func(int* d_array)
{
    __shared__ int s_array[N];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    // Load data from global memory to shared memory
    s_array[tid] = d_array[idx];
    __syncthreads();

    // Reverse the array in shared memory
    int temp;
    if (tid < N / 2)
    {
        temp = s_array[tid];
        s_array[tid] = s_array[N - 1 - tid];
        s_array[N - 1 - tid] = temp;
    }
    __syncthreads();

    // Write the reversed array back to global memory
    d_array[idx] = s_array[tid];
}





template size_t reverse_array<float>(T* d_array);
template size_t reverse_array<double>(T* d_array);
template size_t reverse_array<at::Half>(T* d_array);

