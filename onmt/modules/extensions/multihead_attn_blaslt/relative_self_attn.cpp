#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace relative_partially_learnable_self {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(
                               bool                 is_training,
                               bool                 use_time_mask,
                               int                  heads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& pos,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& pos_weights,
                               torch::Tensor const& input_biases,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pos_biases,
                               torch::Tensor const& r_w_bias,
                               torch::Tensor const& r_r_bias,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace
                                                  );
std::vector<torch::Tensor> bwd_cuda(
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& attn_scores,
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& pos_lin_results,
//                               torch::Tensor const& rw_head_q,
//                               torch::Tensor const& rr_head_q,
                               torch::Tensor const& r_w_bias,
                               torch::Tensor const& r_r_bias,
                               torch::Tensor const& inputs,
                               torch::Tensor const& pos,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& pos_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace
                                                  );

//std::vector<torch::Tensor> bwd_recompute_cuda(
//                               bool use_time_mask,
//                               int                  heads,
//                               torch::Tensor const& output_grads,
//                               torch::Tensor const& inputs,
//                               torch::Tensor const& pos,
//                               torch::Tensor const& input_weights,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& input_biases,
//                               torch::Tensor const& output_biases,
//                               torch::Tensor const& r_w_bias,
//                               torch::Tensor const& r_r_bias,
//                               torch::Tensor const& dropout_mask,
//                               torch::Tensor const& pad_mask,
//                               float                dropout_prob,
//                               torch::Tensor lt_workspace
//                                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
                               bool                 is_training,
                               bool                 use_time_mask,
                               int                  heads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& pos,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& pos_weights,
                               torch::Tensor const& input_biases,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pos_biases,
                               torch::Tensor const& r_w_bias,
                               torch::Tensor const& r_r_bias,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(inputs.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(pos.dim()        == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()  == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(input_biases.dim()  == 1, "expected 2D tensor");
  AT_ASSERTM(output_biases.dim() == 1, "expected 2D tensor");
  AT_ASSERTM(r_w_bias.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(r_w_bias.dim() == 2, "expected 2D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(pos.type().scalarType()        == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_biases.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_biases.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(r_w_bias.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(r_w_bias.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");

//  AT_ASSERTM(pad_mask.dim()                     == 4,                    "expected 4D tensor");
  auto lt_workspace = torch::empty({1 << 22}, inputs.type());
  // AT_ASSERTM(pad_mask.type().scalarType()       == at::ScalarType::Bool, "Only BOOL is supported");

  return fwd_cuda(
                                 is_training,
                                 use_time_mask,
                                 heads,
                                 inputs,
                                 pos,
                                 input_weights,
                                 output_weights,
                                 pos_weights,
                                 input_biases,
                                 output_biases,
                                 pos_biases,
                                 r_w_bias,
                                 r_r_bias,
                                 pad_mask,
                                 dropout_prob,
                                 lt_workspace
                                );
}

std::vector<torch::Tensor> bwd(
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& attn_scores,
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& pos_lin_results,
                               torch::Tensor const& r_w_bias,
                               torch::Tensor const& r_r_bias,
                               torch::Tensor const& inputs,
                               torch::Tensor const& pos,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& pos_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(matmul2_results.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_results.dim()      == 3, "expected 3D tensor");
//  AT_ASSERTM(attn_scores.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_results.dim()  == 3, "expected 3D tensor");
//  AT_ASSERTM(rw_head_q.dim()  == 3, "expected 3D tensor");
//  AT_ASSERTM(rr_head_q.dim()  == 3, "expected 3D tensor");
  AT_ASSERTM(r_w_bias.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(r_w_bias.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(inputs.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(pos.dim()        == 3, "expected 3D tensor");  // len_q x len_q x head_dim
  AT_ASSERTM(input_weights.dim()  == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(dropout_mask.dim()         == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(matmul2_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(attn_scores.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(pos.type().scalarType()        == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  auto lt_workspace = torch::empty({1 << 22}, inputs.type());

  return bwd_cuda(
                               heads,
                               output_grads,
                               matmul2_results,
                               dropout_results,
                               attn_scores,
                               input_lin_results,
                               pos_lin_results,
                               r_w_bias,
                               r_r_bias,
                               inputs,
                               pos,
                               input_weights,
                               output_weights,
                               pos_weights,
                               dropout_mask,
                               dropout_prob,
                               lt_workspace
                                );
}

//
//std::vector<torch::Tensor> bwd_recompute(
//                               int                  heads,
//                               torch::Tensor const& output_grads,
//                               torch::Tensor const& inputs,
//                               torch::Tensor const& pos,
//                               torch::Tensor const& input_weights,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& pos_weights,
//                               torch::Tensor const& input_biases,
//                               torch::Tensor const& output_biases,
//                               torch::Tensor const& pos_biases,
//                               torch::Tensor const& r_w_bias,
//                               torch::Tensor const& r_r_bias,
//                               torch::Tensor const& dropout_mask,
//                               torch::Tensor const& pad_mask,
//                               float                dropout_prob
//                                                  )
//{
//  AT_ASSERTM(output_grads.dim()         == 3, "expected 3D tensor");
//  AT_ASSERTM(inputs.dim()         == 3, "expected 3D tensor");
//  AT_ASSERTM(pos.dim()        == 3, "expected 3D tensor");
//  AT_ASSERTM(input_weights.dim()  == 2, "expected 2D tensor");
//  AT_ASSERTM(output_weights.dim() == 2, "expected 2D tensor");
//  AT_ASSERTM(input_biases.dim()  == 1, "expected 2D tensor");
//  AT_ASSERTM(output_biases.dim() == 1, "expected 2D tensor");
//  AT_ASSERTM(r_w_bias.dim() == 2, "expected 2D tensor");
//  AT_ASSERTM(r_w_bias.dim() == 2, "expected 2D tensor");
//
//  AT_ASSERTM(output_grads.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(pos.type().scalarType()        == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_weights.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(output_weights.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_biases.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(output_biases.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(r_w_bias.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(r_w_bias.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//
//  AT_ASSERTM(dropout_mask.dim()         == 3, "expected 3D tensor");
//  auto lt_workspace = torch::empty({1 << 22}, inputs.type());
//
//  return bwd_recompute_cuda(
//                               heads,
//                               output_grads,
//                               inputs,
//                               pos,
//                               input_weights,
//                               output_weights,
//                               input_biases,
//                               output_biases,
//                               r_w_bias,
//                               r_r_bias,
//                               dropout_mask,
//                               pad_mask,
//                               dropout_prob,
//                               lt_workspace
//                                );
//}


} // end namespace cublas_gemmex
} // end namespace encdec
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &multihead_attn::relative_partially_learnable_self::cublas_gemmex::fwd, "Relative Self-Attention Forward.");
  m.def("backward", &multihead_attn::relative_partially_learnable_self::cublas_gemmex::bwd, "Relative Self-Attention Backward.");
//  m.def("backward_recompute", &multihead_attn::relative_partially_learnable_self::cublas_gemmex::bwd_recompute, "Relative Self-Attention Backward.");
}
