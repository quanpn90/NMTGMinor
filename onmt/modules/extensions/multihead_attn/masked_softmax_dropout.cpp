#include <torch/extension.h>
#include <vector>
#include <cuda_fp16.h>

namespace multihead_attn {
namespace fused_softmax_dropout {

std::vector<torch::Tensor> fwd_cuda(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& input,
                               float                dropout_prob
                                                  );


torch::Tensor bwd_cuda(
		               int heads,
                       torch::Tensor const& output_grads,
                       torch::Tensor const& softmax_results,
                       torch::Tensor const& dropout_mask,
                       float                dropout_prob
                      );

torch::Tensor bwd_recompute_cuda(
                                   int heads,
                                   torch::Tensor const& output_grads,
                                   torch::Tensor const& softmax_inputs,
                                   torch::Tensor const& dropout_mask,
                                   float                dropout_prob
                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(bool 				is_training,
                               int                  heads,
                               torch::Tensor const& input,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(input.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(input.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");

  return fwd_cuda(
                 is_training,
                 heads,
                 input,
                 dropout_prob
                );
}

torch::Tensor bwd(
		                       int heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& softmax_results,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                 )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(softmax_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(
		                 heads,
                         output_grads,
                         softmax_results,
                         dropout_mask,
                         dropout_prob
                        );
}

torch::Tensor bwd_recompute(
		                       int heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& softmax_inputs,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                 )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_inputs.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(softmax_inputs.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_recompute_cuda(
		                 heads,
                         output_grads,
                         softmax_inputs,
                         dropout_mask,
                         dropout_prob
                        );
}

} // end namespace mask_softmax_dropout
} // end namespace fused_softmax

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &multihead_attn::fused_softmax_dropout::fwd, "Fused softmax dropout 3D -- Forward.");
  m.def("backward", &multihead_attn::fused_softmax_dropout::bwd, "Self softmax dropout 3D -- Backward.");
  m.def("backward_recompute", &multihead_attn::fused_softmax_dropout::bwd_recompute, "Self softmax dropout 3D -- Backward.");
}

