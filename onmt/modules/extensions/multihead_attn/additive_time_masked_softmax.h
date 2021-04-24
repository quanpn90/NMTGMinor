#pragma once
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <curand_kernel.h>
#include "philox.h"

#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cmath>
#include "softmax.h"

//namespace {
//    template <typename Datatype, int ELEMENTS_PER_LDG>
//    __device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);
//
//    template <>
//    __device__ __inline__ void copy_vector<__half, 1>(__half *dst, const __half *src) { *dst = *src; }
//
//    template <>
//    __device__ __inline__ void copy_vector<float, 1>(float *dst, const float *src) { *dst = *src; }
//
//    template <>
//    __device__ __inline__ void copy_vector<__half, 4>(__half *dst, const __half *src) { *((float2*) dst) = *((float2*) src); }
//    template <>
//    __device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst, const uint8_t *src) { *dst = *src; }
//
//    template <>
//    __device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst, const uint8_t *src) {*((half2*) dst) = *((half2*) src); }
//
//    template <typename Datatype, int ELEMENTS_PER_LDG>
//    __device__ __inline__ void apply_mask(Datatype *dst, Datatype value, const uint8_t *src);
//
//    template <>
//    __device__ __inline__ void apply_mask<__half, 1>(__half *dst, __half value, const uint8_t *src) {
//      if (*src == 1) { *dst = value; }
//    }
//    template <typename Datatype, int ELEMENTS_PER_LDG>
//    __device__ __inline__ void apply_additive_mask(Datatype *dst, const Datatype *additive_mask);
//    template <>
//    __device__ __inline__ void apply_additive_mask<__half, 1>(__half *dst, const __half *additive_mask) {
//      *dst += *additive_mask;
//    }
//    template <>
//    __device__ __inline__ void apply_additive_mask<__half, 4>(__half *dst, const __half *additive_mask) {
//      *dst += *additive_mask;
//      *(dst+1) += *(additive_mask+1);
//      *(dst+2) += *(additive_mask+2);
//      *(dst+3) += *(additive_mask+3);}
//} // namespace anonymous

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t,
typename output_t,
typename acc_t,
int WARP_BATCH,
int WARP_ITERATIONS,
int WARP_SIZE,
int ELEMENTS_PER_LDG_STG>
__global__ void additive_time_masked_softmax_dropout_warp_forward_vec4(output_t *dst,
uint8_t *dropout_mask,
const input_t *src,
const input_t *pad_mask,
int batch_size, int stride,
int element_count,
int mod_seq_len,
at::PhiloxCudaState philox_args,
float p)
{

    assert(ELEMENTS_PER_LDG_STG==4);
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    acc_t pinv = acc_t(1)/p;
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    //vectorize if element_count is multiple of 4, else don't vectorize
    input_t elements_input[WARP_BATCH][WARP_ITERATIONS];

    int thread_offset =  first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    src += thread_offset;
    dst += thread_offset;
    dropout_mask += thread_offset;

    // load data from global memory
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
//        int pad_thread_offset = ( (first_batch + i) / pad_batch_stride) * stride + ELEMENTS_PER_LDG_STG * local_idx;
        int pad_thread_offset = ( (first_batch + i) % mod_seq_len) * stride + ELEMENTS_PER_LDG_STG * local_idx;
        const half* curr_mask    = pad_mask + pad_thread_offset;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            #pragma unroll
            for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
    	        //masking_value is a large negative value
                elements_input[i][it + element] = -65,504;
            }

            if (element_index < batch_element_count) {
                int itr_jmp = it * WARP_SIZE;
                int itr_idx = i * element_count + itr_jmp;
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], src + itr_idx);
                apply_additive_mask<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], curr_mask + itr_jmp);
                //(__half)-std::numeric_limits<float>::infinity()
            }

        }
    }
    // convert input_t to acc_t
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = elements_input[i][it];
        }
    }

    constexpr uint32_t  FULL_MASK = 0xffffffff;

    // compute local max_value

    // take the max_value of the first element to avoid one max call
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        max_value[i] = elements[i][0];
    }

    #pragma unroll
    for (int it = 1;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }

    // reduction max_value
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float val[WARP_BATCH];
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
        }
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }

    // compute local sum
    acc_t sum[WARP_BATCH] { 0.0f };

    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }

    // reduction sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }
    auto seeds = at::cuda::philox::unpack(philox_args);
    Philox ph(std::get<0>(seeds), tid, std::get<1>(seeds));
    uint8_t rands[WARP_BATCH][WARP_ITERATIONS];
    float4 rand_num;
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
	#pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it+=ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
		rand_num = uniform4(ph());
                rands[i][it] = (rand_num.x <= p) > 0.5;
                rands[i][it+1] = (rand_num.y <= p) > 0.5;
                rands[i][it+2] = (rand_num.z <= p) > 0.5;
                rands[i][it+3] = (rand_num.w <= p) > 0.5;
                copy_vector<uint8_t, ELEMENTS_PER_LDG_STG>(dropout_mask + i * element_count + it * WARP_SIZE, &rands[i][it]);
	    }
        }
    }

    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                output_t out[ELEMENTS_PER_LDG_STG];
                #pragma unroll
                for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                    out[element] = rands[i][it+element] * (pinv * (elements[i][it + element] / sum[i]));
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + i * element_count + it * WARP_SIZE, out);

            }
            else {
                break;
            }
        }
    }
}


template <typename input_t,
typename output_t,
typename acc_t,
int WARP_BATCH,
int WARP_ITERATIONS,
int WARP_SIZE,
int ELEMENTS_PER_LDG_STG>
__global__ void additive_time_masked_softmax_dropout_warp_forward(output_t *dst,
                                                        uint8_t *dropout_mask,
                                                        const input_t *src,
                                                        const input_t *pad_mask,
                                                        int batch_size,
                                                        int stride,
                                                        int element_count,
                                                        int mod_seq_len,
                                                        at::PhiloxCudaState philox_args, float p)
{
    assert(ELEMENTS_PER_LDG_STG==1);

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // require for dropout
    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    acc_t pinv = acc_t(1)/p;
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    //vectorize if element_count is multiple of 4, else don't vectorize
    input_t elements_input[WARP_BATCH][WARP_ITERATIONS];

    int thread_offset =  first_batch * stride + local_idx;
    src += thread_offset;
    dst += thread_offset;
    dropout_mask += thread_offset;

    // load data from global memory
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
//        int pad_thread_offset = ( (first_batch + i) / pad_batch_stride) * stride + local_idx;
        int pad_thread_offset = ( (first_batch + i) % mod_seq_len) * stride + ELEMENTS_PER_LDG_STG * local_idx;
        const half* curr_mask    = pad_mask + pad_thread_offset;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += 1) {
            int element_index = local_idx + it * WARP_SIZE;
            #pragma unroll
            for (int element = 0;element < 1;++element) {
    	        //masking_value is a large negative value
                elements_input[i][it + element] = -65,504;
            }

            if (element_index < batch_element_count) {
                int itr_jmp = it * WARP_SIZE;
                int itr_idx = i * element_count + itr_jmp;
                copy_vector<input_t, 1>(&elements_input[i][it], src + itr_idx);
                apply_additive_mask<input_t, 1>(&elements_input[i][it], curr_mask + itr_jmp);
            }

        }
    }
    // convert input_t to acc_t
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = elements_input[i][it];
        }
    }

    constexpr uint32_t  FULL_MASK = 0xffffffff;

    // compute local max_value

    // take the max_value of the first element to avoid one max call
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        max_value[i] = elements[i][0];
    }

    #pragma unroll
    for (int it = 1;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }

    // reduction max_value
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float val[WARP_BATCH];
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
        }
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }

    // compute local sum
    acc_t sum[WARP_BATCH] { 0.0f };

    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }

    // reduction sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }
    curandStatePhilox4_32_10_t state;
    auto seeds = at::cuda::philox::unpack(philox_args);
    curand_init(
      std::get<0>(seeds),
      tid,
      std::get<1>(seeds),
      &state);

    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += 1) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                output_t out[1];
                acc_t softmax_out[1];
                uint8_t dropout_mask_temp[1];
                //generate a vector of random numbers here
                float rand = curand_uniform(&state);
                float *rand_ptr = (float*)(&rand);
                #pragma unroll
                for (int element = 0;element < 1;++element) {
    	        softmax_out[element] = (elements[i][it + element] / sum[i]);
                    rand_ptr[element] = rand_ptr[element] <= p;
                    out[element] = rand_ptr[element] * pinv * softmax_out[element];
    	            dropout_mask_temp[element] = rand_ptr[element] > 0.5; // just to distinguish 0.0f and 1.0f
                }
                copy_vector<output_t, 1>(dst + i * element_count + it * WARP_SIZE, out);
                copy_vector<uint8_t, 1>(dropout_mask + i * element_count + it * WARP_SIZE, dropout_mask_temp);

            }
            else {
                break;
            }
        }
    }
}


// WARP_BATCH number of batches.
// WARP_ITERATIONS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// ELEMENTS_PER_LDG_STG has to be 1.
template <typename input_t, typename output_t, typename acc_t>
using additive_time_masked_softmax_dropout_forward_func = void(*)(output_t *dst,
                                                             uint8_t *dropout_mask, const input_t *src,
                                                             const input_t *pad_mask, int batch_size, int stride,
                                                             int element_count, int mod_seq_len,
                                                             at::PhiloxCudaState philox_args, float p);


template <typename input_t, typename output_t, typename acc_t>
bool warp_time_additive_masked_softmax_dropout_kernel(int element_count,
                                                 int log2_elements,
                                                 int &warp_size, int &batches_per_warp,
                                                 additive_time_masked_softmax_dropout_forward_func<input_t, output_t,
                                                 acc_t> &kernel)
{
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    bool flag_vec4 = (element_count % 4 == 0);
    switch (log2_elements) {
    case 0: // 1
        kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,1,1,1>;
        break;
    case 1: // 2
        kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,1,2,1>;
        break;
    case 2: // 4
        kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,1,4,1>;
        break;
    case 3: // 8
        kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,1,8,1>;
        break;
    case 4: // 16
        kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,1,16,1>;
        break;
    case 5: // 32
        kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,1,32,1>;
        break;
    case 6: // 64
        kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,2,32,1>;
        break;
    case 7: // 128
	if (flag_vec4) kernel = &additive_time_masked_softmax_dropout_warp_forward_vec4<input_t, output_t, acc_t, 2,4,32,4>;
	else kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 2,4,32,1>;
        break;
    case 8: // 256
	if (flag_vec4) kernel = &additive_time_masked_softmax_dropout_warp_forward_vec4<input_t, output_t, acc_t, 1,8,32,4>;
	else kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 1,8,32,1>;
        break;
    case 9: // 512
        if (flag_vec4) kernel = &additive_time_masked_softmax_dropout_warp_forward_vec4<input_t, output_t, acc_t, 1,16,32,4>;
	else kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 1,16,32,1>;
        break;
    case 10: // 1024
        if (flag_vec4) kernel = &additive_time_masked_softmax_dropout_warp_forward_vec4<input_t, output_t, acc_t, 1,32,32,4>;
	else kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 1,32,32,1>;
        break;
    case 11: // 2048
        if (flag_vec4) kernel = &additive_time_masked_softmax_dropout_warp_forward_vec4<input_t, output_t, acc_t, 1,64,32,4>;
	else kernel = &additive_time_masked_softmax_dropout_warp_forward<input_t, output_t, acc_t, 1,64,32,1>;
        break;
    default:
        return false;
    }
    return true;
}



template<typename input_t, typename output_t, typename acc_t>
bool dispatch_additive_time_masked_softmax_dropout(
                                                output_t *dst,
                                                uint8_t *dropout_mask,
                                                const input_t *src,
                                                const input_t *pad_mask,
                                                int totalElements,
                                                int softmax_elements,
                                                int softmax_elements_stride,
                                                int batch_count,
                                                int mod_seq_len,
                                                float p,
                                                cudaStream_t streamid)// p is the probability to keep, not drop
{

    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 2048) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;

        // define the kernel here
        additive_time_masked_softmax_dropout_forward_func<input_t, output_t, acc_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_time_additive_masked_softmax_dropout_kernel<input_t, output_t, acc_t>(softmax_elements, log2_elements,
                                                                               warp_size, batches_per_warp, kernel))
        {
            return false;
        }

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
	    c10::optional<at::Generator> gen_;
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_,
                                                                       at::cuda::detail::getDefaultCUDAGenerator());
        int64_t counter_offset = (totalElements/(blocks*threads_per_block)+1);
        at::PhiloxCudaState rng_engine_inputs;
	    {
	        std::lock_guard<std::mutex> lock(gen->mutex_);
	        rng_engine_inputs = gen->philox_cuda_state(counter_offset);
        }

        // compute launch size
        dim3 threads(warp_size, warps_per_block, 1);

        // launch
        kernel<<<blocks, threads, 0, streamid>>>(dst,
                                                 dropout_mask,
                                                 src,
                                                 pad_mask,
                                                 batch_count,
                                                 softmax_elements_stride, softmax_elements,
                                                 mod_seq_len,
                                                 rng_engine_inputs, p);
        return true;
    }
    return false;
}