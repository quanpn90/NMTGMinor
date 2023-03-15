#include <torch/extension.h>
#include <vector>
#include <cuda_fp16.h>

namespace multihead_attn {
namespace self_bias {
namespace cublaslt {

std::vector<torch::Tensor> fwd_cuda(
                               bool                 use_time_mask,
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace);


std::vector<torch::Tensor> fwd_bias_cuda(
                               bool                 use_time_mask,
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               torch::Tensor & bias,
                               float                dropout_prob,
                               torch::Tensor lt_workspace);


std::vector<torch::Tensor> bwd_cuda(
                               bool use_time_mask,
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& attn_scores,
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace);

std::vector<torch::Tensor> bwd_bias_cuda(
                               bool use_time_mask,
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& attn_scores,
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace);

std::vector<torch::Tensor> bwd_cuda_recompute(
                               bool use_time_mask,
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace);

//torch::Tensor bwd_cuda_input_only(
//                               bool use_time_mask,
//                               int                  heads,
//                               torch::Tensor const& output_grads,
//                               torch::Tensor const& matmul2_results,
//                               torch::Tensor const& dropout_results,
//                               torch::Tensor const& attn_scores,
//                               torch::Tensor const& input_lin_results,
//                               torch::Tensor const& inputs,
//                               torch::Tensor const& input_weights,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& dropout_mask,
//                               float                dropout_prob
//                                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
                               bool                 use_time_mask,
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs, torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases, torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(inputs.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()  == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim() == 2, "expected 2D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(pad_mask.dim()                     == 2,                    "expected 2D tensor");
  auto lt_workspace = torch::empty({1 << 22}, inputs.type());

  return fwd_cuda(
                                 use_time_mask,
                                 is_training,
                                 heads,
                                 inputs,
                                 input_weights,
                                 output_weights,
                                 input_biases,
                                 output_biases,
                                 pad_mask,
                                 dropout_prob,
                                 lt_workspace);
//                                 (void*) (.data_ptr<scalar_t>());

}


std::vector<torch::Tensor> fwd_bias(
                               bool                 use_time_mask,
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs, torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases, torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               torch::Tensor & bias,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(inputs.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()  == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(bias.dim() == 3,           "expected 3D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(bias.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(pad_mask.dim()                     == 2,                    "expected 2D tensor");
  auto lt_workspace = torch::empty({1 << 22}, inputs.type());

  return fwd_bias_cuda(
                                 use_time_mask,
                                 is_training,
                                 heads,
                                 inputs,
                                 input_weights,
                                 output_weights,
                                 input_biases,
                                 output_biases,
                                 pad_mask,
                                 bias,
                                 dropout_prob,
                                 lt_workspace);
//                                 (void*) (.data_ptr<scalar_t>());

}



std::vector<torch::Tensor> bwd(
                               bool use_time_mask,
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& attn_scores,
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(matmul2_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_results.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(inputs.dim()            == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()     == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()    == 2, "expected 2D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(matmul2_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_lin_results.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");
  auto lt_workspace = torch::empty({1 << 22}, inputs.type());


  return bwd_cuda(
                                 use_time_mask,
                                 heads,
                                 output_grads,
                                 matmul2_results,
                                 dropout_results,
                                 attn_scores,
                                 input_lin_results,
                                 inputs,
                                 input_weights,
                                 output_weights,
                                 dropout_mask,
                                 dropout_prob,
                                 lt_workspace);

}


std::vector<torch::Tensor> bwd_bias(
                               bool use_time_mask,
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& attn_scores,
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(matmul2_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_results.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(inputs.dim()            == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()     == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()    == 2, "expected 2D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(matmul2_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_lin_results.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");
  auto lt_workspace = torch::empty({1 << 22}, inputs.type());


  return bwd_bias_cuda(
                                 use_time_mask,
                                 heads,
                                 output_grads,
                                 matmul2_results,
                                 dropout_results,
                                 attn_scores,
                                 input_lin_results,
                                 inputs,
                                 input_weights,
                                 output_weights,
                                 dropout_mask,
                                 dropout_prob,
                                 lt_workspace);

}


std::vector<torch::Tensor> bwd_recompute(
                               bool use_time_mask,
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(inputs.dim()            == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()     == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()    == 2, "expected 2D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(inputs.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");
  auto lt_workspace = torch::empty({1 << 22}, inputs.type());


  return bwd_cuda_recompute(
                                 use_time_mask,
                                 heads,
                                 output_grads,
                                 inputs,
                                 input_weights,
                                 output_weights,
                                 input_biases,
                                 output_biases,
                                 pad_mask,
                                 dropout_mask,
                                 dropout_prob,
                                 lt_workspace);

}
//
//torch::Tensor bwd_input_only(
//                               bool use_time_mask,
//                               int                  heads,
//                               torch::Tensor const& output_grads,
//                               torch::Tensor const& matmul2_results,
//                               torch::Tensor const& dropout_results,
//                               torch::Tensor const& attn_scores,
//                               torch::Tensor const& input_lin_results,
//                               torch::Tensor const& inputs,
//                               torch::Tensor const& input_weights,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& dropout_mask,
//                               float                dropout_prob
//                                                  )
//{
//  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
//  AT_ASSERTM(matmul2_results.dim()   == 3, "expected 3D tensor");
//  AT_ASSERTM(dropout_results.dim()   == 3, "expected 3D tensor");
//  AT_ASSERTM(input_lin_results.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(inputs.dim()            == 3, "expected 3D tensor");
//  AT_ASSERTM(input_weights.dim()     == 2, "expected 2D tensor");
//  AT_ASSERTM(output_weights.dim()    == 2, "expected 2D tensor");
//  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");
//
//  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(matmul2_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(dropout_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_lin_results.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(inputs.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_weights.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(output_weights.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");
//
//  return bwd_cuda_input_only(
//                                 use_time_mask,
//                                 heads,
//                                 output_grads,
//                                 matmul2_results,
//                                 dropout_results,
//                                 attn_scores,
//                                 input_lin_results,
//                                 inputs,
//                                 input_weights,
//                                 output_weights,
//                                 dropout_mask,
//                                 dropout_prob
//                                );
//}

} // end namespace cublas_gemmex
} // end namespace self
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &multihead_attn::self_bias::cublaslt::fwd, "Self Multihead Attention -- Forward.");
  m.def("backward", &multihead_attn::self_bias::cublaslt::bwd, "Self Multihead Attention  -- Backward.");
  m.def("forward_bias", &multihead_attn::self_bias::cublaslt::fwd_bias, "Self Multihead Attention with attn Bias -- Forward.");
  m.def("backward_bias", &multihead_attn::self_bias::cublaslt::bwd_bias, "Self Multihead Attention with attn Bias -- Backward.");
  m.def("backward_recompute", &multihead_attn::self_bias::cublaslt::bwd_recompute, "Self Multihead Attention with Bias -- Backward.");
//  m.def("backward_input_only", &multihead_attn::self_bias_additive_mask::cublas_gemmex::bwd_input_only,
//  "Self Multihead Attention with Bias -- Backward input only (ignore weights).");
}
