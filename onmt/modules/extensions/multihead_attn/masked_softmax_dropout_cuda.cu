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

#include "softmax.h"
#include "dropout.h"
#include "additive_time_masked_softmax.h"

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

namespace multihead_attn {
namespace fused_softmax {
namespace mask_softmax_dropout {

std::vector<torch::Tensor> fwd_cuda(
			                   bool                 is_training,
			                   bool                 time_mask,
                               int                  heads,
                               torch::Tensor const& input, 
                               const half*        pad_mask,
                               float                dropout_prob
                                   )
{
  const int   attn_batches   = input.size(0);
  const int   sequences      = attn_batches / heads;
  const int   q_seq_len      = input.size(1);
  const int   k_seq_len      = input.size(2);
  const int   dropout_elems  = attn_batches * q_seq_len * k_seq_len;

  // There is no reason to use more than one stream as every kernel is 
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  auto act_options  = input.options().requires_grad(false);
  auto mask_options = act_options.dtype(torch::kUInt8);

  torch::Tensor softmax_results   = torch::empty({attn_batches, q_seq_len, k_seq_len},   act_options);
//  torch::Tensor dropout_results   = torch::empty({attn_batches, q_seq_len, k_seq_len},   act_options);
  torch::Tensor dropout_mask      = torch::empty({attn_batches, q_seq_len, k_seq_len},   mask_options);

  // Softmax Intermediate Result Ptr (used by Matmul1 -> Softmax)
  void* input_ptr = static_cast<void*>(input.data_ptr());

  float dropout_keep_prob = 1.0 - dropout_prob;
  if (is_training) {
    dropout_keep_prob = 1.0;
  }

  // Padded Softmax
  bool softmax_success = false;

  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());

  if (time_mask) {

  softmax_success = dispatch_additive_time_masked_softmax_dropout<half, half, float>(
                             reinterpret_cast<half*>(softmax_results_ptr),
                             reinterpret_cast<uint8_t*>(dropout_mask.data_ptr()),
                             reinterpret_cast<const half*>(input_ptr),
                             pad_mask,
                             dropout_elems,
                             k_seq_len,
                             k_seq_len,
                             attn_batches*q_seq_len,
                             q_seq_len,
                             dropout_keep_prob,
                             stream);

  }
  else {

    softmax_success = dispatch_additive_masked_softmax_dropout<half, half, float>(
                             reinterpret_cast<half*>(softmax_results_ptr),
                             reinterpret_cast<uint8_t*>(dropout_mask.data_ptr()),
                             reinterpret_cast<const half*>(input_ptr),
                             pad_mask,
                             dropout_elems,
                             k_seq_len,
                             k_seq_len,
                             attn_batches*q_seq_len,
                             attn_batches*q_seq_len/sequences,
                             dropout_keep_prob,
                             stream);
  }

  assert(softmax_success);

  return {
           dropout_mask, 
           softmax_results
         };
}

torch::Tensor bwd_cuda(
		               int heads,
                       torch::Tensor const& output_grads,
                       torch::Tensor const& softmax_results,
                       torch::Tensor const& dropout_mask,
                       float                dropout_prob
                       )
{
  const int   attn_batches   = output_grads.size(0);
  const int   q_seq_len      = output_grads.size(1);
  const int   k_seq_len      = output_grads.size(2);
  const int   dropout_elems  = attn_batches * q_seq_len * k_seq_len;
  // TODO: Streams can be used in Backprop but I haven't added more than one
  // in my first attempt to create the code
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // Output Tensor Allocations
//  torch::Tensor input_grads         = torch::empty_like(output_grads);

  // Apply Dropout Mask and Scale by Dropout Probability 
  // Softmax Grad
//  if (padding_mask == nullptr) {
//      dispatch_masked_scale_softmax_backward_stream<half, half, float,false>(
//                             static_cast<half*>(output_grads.data_ptr()),
//                             static_cast<half*>(output_grads.data_ptr()),
//                             reinterpret_cast<half const*>(softmax_results.data_ptr()),
//			     static_cast<uint8_t const*>(dropout_mask.data_ptr()),
//			     1.0/(1.0-dropout_prob),
//                             k_seq_len,
//                             k_seq_len,
//                             attn_batches*q_seq_len, stream);
//  } else{
//      dispatch_masked_scale_softmax_backward_masked_out_stream<half, half, float,false>(
//                             static_cast<half*>(output_grads.data_ptr()),
//                             static_cast<half*>(output_grads.data_ptr()),
//                             reinterpret_cast<half const*>(softmax_results.data_ptr()),
//			     static_cast<uint8_t const*>(dropout_mask.data_ptr()),
//			     static_cast<uint8_t const*>(padding_mask),
//			     1.0/(1.0-dropout_prob),
//                             k_seq_len,
//                             k_seq_len,
//                             attn_batches*q_seq_len,
//			     heads, stream);
  dispatch_masked_scale_softmax_backward_stream<half, half, float,false>(
                             static_cast<half*>(output_grads.data_ptr()),
                             static_cast<half*>(output_grads.data_ptr()),
                             reinterpret_cast<half const*>(softmax_results.data_ptr()),
                             static_cast<uint8_t const*>(dropout_mask.data_ptr()),
                             1.0/(1.0-dropout_prob),
                             k_seq_len,
                             k_seq_len,
                             attn_batches*q_seq_len, stream);

//  // alternatively
//  // Apply Dropout Mask and Scale by Dropout Probability
//  apex_masked_scale_cuda<at::Half,float,uint32_t>(
//                             static_cast<at::Half const*>(output_grads.data_ptr()),
//                             static_cast<at::Half*>(output_grads.data_ptr()),
//                             static_cast<uint8_t const*>(dropout_mask.data_ptr()),
//                             dropout_elems,
//                             (1.0 / (1.0 - dropout_prob)));
////
////  // Softmax Grad
////  bool softmax_success = false;
//  softmax_success = dispatch_softmax_backward<half, half, float>(
//                             static_cast<half*>(output_grads.data_ptr()),
//                             static_cast<half*>(output_grads.data_ptr()),
//                             reinterpret_cast<half const*>(softmax_results.data_ptr()),
//                             k_seq_len,
//                             k_seq_len,
//                             attn_batches*q_seq_len);


  //backward pass is completely in-place
  return output_grads;
}
}
}
}

