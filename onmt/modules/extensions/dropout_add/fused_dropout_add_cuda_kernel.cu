#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>

#include "dropout.h"

namespace multihead_attn {
namespace fused_dropout_add {

std::vector<torch::Tensor> fwd_cuda(
                                   bool                 is_training,
                                   torch::Tensor const& inputs,
                                   torch::Tensor const& residuals,
                                   float                dropout_prob
                                  )
{
//    auto input_sizes = inputs[0].sizes();

//    int q_seq_len = 1;
//
//    const int   embed_dim         = inputs.size(2);
//    const int   sequences         = inputs.size(1);
//    const int   q_seq_len         = inputs.size(0);

    const int   total_tokens_q    = at::numel(inputs);

    // There is no reason to use more than one stream as every kernel is
    // sequentially dependent
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    auto act_options                   = inputs.options().requires_grad(false);
    auto mask_options                  = act_options.dtype(torch::kUInt8);
    torch::Tensor dropout_add_mask     = torch::empty_like(inputs, mask_options);
//    torch::Tensor outputs              = torch::empty_like(inputs, act_options);
    torch::Tensor outputs = inputs;   // inplace everything?

    if (is_training && dropout_prob > 0.0) {
        apex_dropout_add_cuda<half,float,uint32_t>(
                             static_cast<const half*>(inputs.data_ptr()),
                             static_cast<const half*>(residuals.data_ptr()),
                             static_cast<half*>(outputs.data_ptr()),
                             static_cast<uint8_t*>(dropout_add_mask.data_ptr()),
                             total_tokens_q,
                             (1.0f - dropout_prob));
    } else {
        apex_add_cuda<half,float,uint32_t>(
                                 static_cast<const half*>(inputs.data_ptr()),
                                 static_cast<const half*>(residuals.data_ptr()),
                                 static_cast<half*>(outputs.data_ptr()),
                                 total_tokens_q);

    }
    return {dropout_add_mask, outputs};

}

torch::Tensor bwd_cuda(
                                   torch::Tensor const& output_grads,
                                   torch::Tensor const& dropout_add_mask,
                                   float                dropout_prob
                                  )
{
    const int   total_tokens_q    = at::numel(output_grads);
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    at::Tensor dropout_add_grads = torch::empty_like(output_grads);

    if (dropout_prob > 0.0) {
        apex_masked_scale_cuda<half,float,uint32_t>(
                                         static_cast<const half*>(output_grads.data_ptr()),
                                         static_cast<half*>(dropout_add_grads.data_ptr()),
                                         static_cast<const uint8_t*>(dropout_add_mask.data_ptr()),
                                         total_tokens_q,
                                         (1.0 / (1.0 - dropout_prob)));
    } else {
        dropout_add_grads.copy_(output_grads);
    }

    // Dropout Add Backward

    return dropout_add_grads;
}

}
}