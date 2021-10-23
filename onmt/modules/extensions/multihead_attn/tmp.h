template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE=32, int ELEMENTS_PER_LDG_STG, bool is_log_softmax>
__global__ void masked_scale_softmax_warp_backward_recompute(output_t *gradInput, const input_t *grad, const input_t *softmax_input, const input_t *pad_mask, const uint8_t *mask, acc_t scale, int batch_size, int stride, int pad_batch_stride, int element_count)
{
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x % WARP_SIZE;
    //vectorize if a row length is multiple of 4
    int flag_vec4 = element_count & 3 == 0;
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS]  ;
    input_t elements_input[WARP_BATCH][WARP_ITERATIONS] ;

    // the first element to process by the current thread
    int thread_offset =  first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;

    grad += thread_offset;
    softmax_input += thread_offset;
    gradInput += thread_offset;
    mask += thread_offset;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        int pad_thread_offset = ( (first_batch + i) / pad_batch_stride) * stride + ELEMENTS_PER_LDG_STG * local_idx;
        const input_t* curr_mask    = pad_mask + pad_thread_offset;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

            #pragma unroll
            for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
    	//masking_value is a large negative value
                elements_input[i][it + element] = -10000;
    	        grad_reg[i][it+element] = acc_t(0);
            }

            if (element_index < batch_element_count) {
                int itr_jmp = it * WARP_SIZE;
                int itr_idx = i * element_count + itr_jmp;
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], softmax_input + itr_idx);
                apply_additive_mask<input_t, ELEMENTS_PER_LDG_STG>(&elements_input[i][it], curr_mask + itr_jmp); //(__half)-std::numeric_limits<float>::infinity()
                uint8_t mask_temp[ELEMENTS_PER_LDG_STG];
                input_t grad_temp[ELEMENTS_PER_LDG_STG];
                copy_vector<uint8_t, ELEMENTS_PER_LDG_STG>(&mask_temp[0], mask + itr_idx);
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(&grad_temp[0], grad + itr_idx);
                #pragma unroll
                for (int element = 0;element < ELEMENTS_PER_LDG_STG;++element) {
                    grad_reg[i][it+element] = ((acc_t)mask_temp[element] * (acc_t)grad_temp[element] * (acc_t)scale );
                }
            }

        }
    }
    // load data from global memory

    // convert input_t to acc_t
    // TODO : remove this, input is already acc_t type in register
    acc_t elements[WARP_BATCH][WARP_ITERATIONS] ;
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
            //elements[i][it] = expf(elements[i][it] - max_value[i]);
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

    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it ++) {
	   elements[i][it] = elements[i][it] / sum[i];
           grad_reg[i][it] = grad_reg[i][it] * elements[i][it];
	}
    }

    acc_t grad_sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        grad_sum[i] = grad_reg[i][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            grad_sum[i] += grad_reg[i][it];
        }
    }
    warp_reduce_sum<acc_t, WARP_BATCH, WARP_SIZE>(grad_sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it += ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
	            output_t grad_input_reg[ELEMENTS_PER_LDG_STG];
                #pragma unroll
	            for (int element=0; element<ELEMENTS_PER_LDG_STG; element++) {
                    if (is_log_softmax) {
                        grad_input_reg[element] = (grad_reg[i][it+element] - std::exp(elements[i][it+element]) * grad_sum[i]);
                    } else {
                        grad_input_reg[element] = (grad_reg[i][it+element] - elements[i][it+element] * grad_sum[i]);
                    }

	            }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(gradInput + i * element_count + it * WARP_SIZE, grad_input_reg);
            }
        }
    }
}



template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
using masked_scale_softmax_warp_backward_recompute_func = void(*)(output_t *gradInput, const input_t *grad, const input_t *softmax_input, const input_t *pad_mask, const uint8_t *mask, acc_t scale, int batch_size, int stride, int pad_batch_stride, int element_count);

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
bool masked_scale_softmax_warp_backward_recompute_kernel(int element_count, int log2_elements, int &warp_size, int &batches_per_warp, masked_scale_softmax_warp_backward_recompute_func<input_t, output_t, acc_t, is_log_softmax> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    bool flag_vec4 = (element_count % 4 == 0);
    switch (log2_elements) {
    case 0: // 1
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,1,1,1, is_log_softmax>;
        break;
    case 1: // 2
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,1,2,1, is_log_softmax>;
        break;
    case 2: // 4
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,1,4,1, is_log_softmax>;
        break;
    case 3: // 8
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,1,8,1, is_log_softmax>;
        break;
    case 4: // 16
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,1,16,1, is_log_softmax>;
        break;
    case 5: // 32
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,1,32,1, is_log_softmax>;
        break;
    case 6: // 64
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,2,32,1, is_log_softmax>;
        break;
    case 7: // 128
        kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 2,4,32,1, is_log_softmax>;
        break;
    case 8: // 256
	if (flag_vec4) kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,8,32,4, is_log_softmax>;
	else kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,8,32,1, is_log_softmax>;
        break;
    case 9: // 512
        if (flag_vec4) kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,16,32,4, is_log_softmax>;
	else kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,16,32,1, is_log_softmax>;
        break;
    case 10: // 1024
        if (flag_vec4) kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,32,32,4, is_log_softmax>;
	else kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,32,32,1, is_log_softmax>;
        break;
    case 11: // 2048
        if (flag_vec4) kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,64,32,4, is_log_softmax>;
	else kernel = &masked_scale_softmax_warp_backward_recompute<input_t, output_t, acc_t, 1,64,32,1, is_log_softmax>;
        break;
    default:
        return false;
    }
    return true;
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
bool dispatch_masked_scale_softmax_backward_recompute(output_t *grad_input, const input_t *grad, const input_t *softmax_input, const input_t *pad_mask, const uint8_t *mask, acc_t scale, int softmax_elements, int softmax_elements_stride, int pad_batch_stride, int batch_count, cudaStream_t streamid)
{

    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 2048) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;

        masked_scale_softmax_warp_backward_recompute_func<input_t, output_t, acc_t, is_log_softmax> kernel;
        int warp_size, batches_per_warp;
        if (!masked_scale_softmax_warp_backward_recompute_kernel<input_t, output_t, acc_t, is_log_softmax>(softmax_elements, log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;

        // compute launch size
        dim3 threads(warp_size, warps_per_block, 1);

        // launch
        kernel<<<blocks, threads, 0, streamid>>>(grad_input, grad, softmax_input, pad_mask, mask, scale, batch_count, softmax_elements_stride, pad_batch_stride, softmax_elements);
        return true;
    }
    return false;
}
