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
    const int   embed_dim         = inputs.size(2);
    const int   sequences         = inputs.size(1);
    const int   q_seq_len         = inputs.size(0);

    const int   total_tokens_q    = sequences * q_seq_len * embed_dim;

    // There is no reason to use more than one stream as every kernel is
    // sequentially dependent
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    auto act_options                   = inputs.options().requires_grad(false);
    auto lyr_nrm_options               = act_options.dtype(torch::kFloat32);
    auto mask_options                  = act_options.dtype(torch::kUInt8);
    torch::Tensor dropout_add_mask     = torch::empty_like(inputs, mask_options);
    torch::Tensor outputs              = torch::empty_like(inputs, act_options);

    if (is_training) {
        apex_dropout_add_cuda<at::Half,float,uint32_t>(
                             static_cast<at::Half const*>(inputs.data_ptr()),
                             static_cast<at::Half const*>(residuals.data_ptr()),
                             static_cast<at::Half*>(outputs.data_ptr()),
                             static_cast<uint8_t*>(dropout_add_mask.data_ptr()),
                             total_tokens_q,
                             (1.0f - dropout_prob));
    } else {
        apex_add_cuda<at::Half,float,uint32_t>(
                                 static_cast<at::Half const*>(inputs.data_ptr()),
                                 static_cast<at::Half const*>(residuals.data_ptr()),
                                 static_cast<at::Half*>(outputs.data_ptr()),
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
    const int   embed_dim         = output_grads.size(2);
    const int   sequences         = output_grads.size(1);
    const int   q_seq_len         = output_grads.size(0);

    const int   total_tokens_q    = sequences * q_seq_len * embed_dim;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    at::Tensor dropout_add_grads         = torch::empty_like(output_grads);

    // Dropout Add Backward
    apex_masked_scale_cuda<at::Half,float,uint32_t>(
                                 static_cast<at::Half const*>(output_grads.data_ptr()),
                                 static_cast<at::Half*>(dropout_add_grads.data_ptr()),
                                 static_cast<uint8_t const*>(dropout_add_mask.data_ptr()),
                                 total_tokens_q,
                                 (1.0 / (1.0 - dropout_prob)));

    return dropout_add_grads;
}

}
}