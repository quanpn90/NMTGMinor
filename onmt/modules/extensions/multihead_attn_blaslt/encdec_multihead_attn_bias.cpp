#include <torch/extension.h>
#include <vector>
#include <cuda_fp16.h>

namespace multihead_attn {
namespace encdec_bias {
namespace cublaslt {

std::vector<torch::Tensor> fwd_cuda(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases_q,
                               torch::Tensor const& input_biases_kv,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace);

std::vector<torch::Tensor> bwd_cuda(
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& attn_scores,
//                               const half* pad_mask,
                               torch::Tensor const& input_lin_q_results,
                               torch::Tensor const& input_lin_kv_results,
                               torch::Tensor const& inputs_q,
                               torch::Tensor const& inputs_kv,
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob,
                               torch::Tensor lt_workspace
                                                  );
//
//std::vector<torch::Tensor> bwd_cuda_input_only(
//                               int                  heads,
//                               torch::Tensor const& output_grads,
//                               torch::Tensor const& matmul2_results,
//                               torch::Tensor const& dropout_results,
//                               torch::Tensor const& attn_scores,
////                               const half* pad_mask,
//                               torch::Tensor const& input_lin_q_results,
//                               torch::Tensor const& input_lin_kv_results,
//                               torch::Tensor const& inputs_q,
//                               torch::Tensor const& inputs_kv,
//                               torch::Tensor const& input_weights_q,
//                               torch::Tensor const& input_weights_kv,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& dropout_mask,
//                               float                dropout_prob
//                                                  );
//std::vector<torch::Tensor> bwd_recompute_cuda(
//                               int                  heads,
//                               torch::Tensor const& output_grads,
////                               torch::Tensor const& matmul2_results,
////                               torch::Tensor const& dropout_results,
////                               torch::Tensor const& softmax_results,
////                               torch::Tensor const& input_lin_q_results,
////                               torch::Tensor const& input_lin_kv_results,
//                               torch::Tensor const& inputs_q,
//                               torch::Tensor const& inputs_kv,
//                               torch::Tensor const& input_weights_q,
//                               torch::Tensor const& input_weights_kv,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& input_biases_q,
//                               torch::Tensor const& input_biases_kv,
//                               torch::Tensor const& output_biases,
//                               torch::Tensor const& dropout_mask,
//                               torch::Tensor const& pad_mask,
//                               float                dropout_prob
//);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases_q,
                               torch::Tensor const& input_biases_kv,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(inputs_q.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_kv.dim()        == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights_q.dim()  == 2, "expected 2D tensor");
  AT_ASSERTM(input_weights_kv.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()   == 2, "expected 2D tensor");
  AT_ASSERTM(input_biases_q.dim()  == 1, "expected 2D tensor");
  AT_ASSERTM(input_biases_kv.dim() == 1, "expected 2D tensor");
  AT_ASSERTM(output_biases.dim()   == 1, "expected 2D tensor");

  AT_ASSERTM(inputs_q.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_kv.type().scalarType()        == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_q.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_kv.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  
//  AT_ASSERTM(pad_mask.dim()                     == 2,                    "expected 2D tensor");
  // AT_ASSERTM(pad_mask.type().scalarType()       == at::ScalarType::Byte, "Only BYTE is supported");
  auto lt_workspace = torch::empty({1 << 22}, inputs_q.type());
  
  return fwd_cuda(
                     is_training,
                     heads,
                     inputs_q,
                     inputs_kv,
                     input_weights_q,
                     input_weights_kv,
                     output_weights,
                     input_biases_q,
                     input_biases_kv,
                     output_biases,
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
//                               torch::Tensor const& pad_mask,
                               torch::Tensor const& input_lin_q_results,
                               torch::Tensor const& input_lin_kv_results,
                               torch::Tensor const& inputs_q,
                               torch::Tensor const& inputs_kv,
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(matmul2_results.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_results.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(attn_scores.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_q_results.dim()  == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_kv_results.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_q.dim()             == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_kv.dim()            == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights_q.dim()      == 2, "expected 2D tensor");
  AT_ASSERTM(input_weights_kv.dim()     == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()       == 2, "expected 2D tensor");
  AT_ASSERTM(dropout_mask.dim()         == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(matmul2_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(attn_scores.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_lin_q_results.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_lin_kv_results.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_q.type().scalarType()             == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_kv.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_q.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_kv.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()       == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_mask.type().scalarType()         == at::ScalarType::Byte, "Only BYTE is supported");

  auto lt_workspace = torch::empty({1 << 22}, inputs_q.type());

  return bwd_cuda(
                                 heads,
                                 output_grads,
                                 matmul2_results,
                                 dropout_results,
                                 attn_scores,
                                 input_lin_q_results,
                                 input_lin_kv_results,
                                 inputs_q,
                                 inputs_kv,
                                 input_weights_q,
                                 input_weights_kv,
                                 output_weights,
                                 dropout_mask,
                                 dropout_prob,
                                 lt_workspace
                                );
}
//
//
//std::vector<torch::Tensor> bwd_input_only(
//                               int                  heads,
//                               torch::Tensor const& output_grads,
//                               torch::Tensor const& matmul2_results,
//                               torch::Tensor const& dropout_results,
//                               torch::Tensor const& attn_scores,
//                               torch::Tensor const& input_lin_q_results,
//                               torch::Tensor const& input_lin_kv_results,
//                               torch::Tensor const& inputs_q,
//                               torch::Tensor const& inputs_kv,
//                               torch::Tensor const& input_weights_q,
//                               torch::Tensor const& input_weights_kv,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& dropout_mask,
//                               float                dropout_prob
//                                                  )
//{
//  AT_ASSERTM(output_grads.dim()         == 3, "expected 3D tensor");
//  AT_ASSERTM(matmul2_results.dim()      == 3, "expected 3D tensor");
//  AT_ASSERTM(dropout_results.dim()      == 3, "expected 3D tensor");
//  AT_ASSERTM(attn_scores.dim()      == 3, "expected 3D tensor");
//  AT_ASSERTM(input_lin_q_results.dim()  == 3, "expected 3D tensor");
//  AT_ASSERTM(input_lin_kv_results.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(inputs_q.dim()             == 3, "expected 3D tensor");
//  AT_ASSERTM(inputs_kv.dim()            == 3, "expected 3D tensor");
//  AT_ASSERTM(input_weights_q.dim()      == 2, "expected 2D tensor");
//  AT_ASSERTM(input_weights_kv.dim()     == 2, "expected 2D tensor");
//  AT_ASSERTM(output_weights.dim()       == 2, "expected 2D tensor");
//  AT_ASSERTM(dropout_mask.dim()         == 3, "expected 3D tensor");
//
//  AT_ASSERTM(output_grads.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(matmul2_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(dropout_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(attn_scores.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_lin_q_results.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_lin_kv_results.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(inputs_q.type().scalarType()             == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(inputs_kv.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_weights_q.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(input_weights_kv.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(output_weights.type().scalarType()       == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(dropout_mask.type().scalarType()         == at::ScalarType::Byte, "Only BYTE is supported");
//
//  return bwd_cuda_input_only(
//                                 heads,
//                                 output_grads,
//                                 matmul2_results,
//                                 dropout_results,
//                                 attn_scores,
//                                 input_lin_q_results,
//                                 input_lin_kv_results,
//                                 inputs_q,
//                                 inputs_kv,
//                                 input_weights_q,
//                                 input_weights_kv,
//                                 output_weights,
//                                 dropout_mask,
//                                 dropout_prob
//                                );
//}


//std::vector<torch::Tensor> bwd_recompute(
//                               int                  heads,
//                               torch::Tensor const& output_grads,
//                               torch::Tensor const& inputs_q,
//                               torch::Tensor const& inputs_kv,
//                               torch::Tensor const& input_weights_q,
//                               torch::Tensor const& input_weights_kv,
//                               torch::Tensor const& output_weights,
//                               torch::Tensor const& dropout_mask,
//                               torch::Tensor const& pad_mask,
//                               float                dropout_prob
//                                                  )
//{
//    AT_ASSERTM(output_grads.dim()         == 3, "expected 3D tensor");
//    AT_ASSERTM(inputs_q.dim()             == 3, "expected 3D tensor");
//    AT_ASSERTM(inputs_kv.dim()            == 3, "expected 3D tensor");
//    AT_ASSERTM(input_weights_q.dim()      == 2, "expected 2D tensor");
//    AT_ASSERTM(input_weights_kv.dim()     == 2, "expected 2D tensor");
//    AT_ASSERTM(output_weights.dim()       == 2, "expected 2D tensor");
//    AT_ASSERTM(dropout_mask.dim()         == 3, "expected 3D tensor");
//
//    AT_ASSERTM(output_grads.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
//    AT_ASSERTM(inputs_q.type().scalarType()             == at::ScalarType::Half, "Only HALF is supported");
//    AT_ASSERTM(inputs_kv.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
//    AT_ASSERTM(input_weights_q.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
//    AT_ASSERTM(input_weights_kv.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
//    AT_ASSERTM(output_weights.type().scalarType()       == at::ScalarType::Half, "Only HALF is supported");
//    AT_ASSERTM(dropout_mask.type().scalarType()         == at::ScalarType::Byte, "Only BYTE is supported");
//
//    return bwd_recompute_cuda(
//                                 heads,
//                                 output_grads,
//                                 inputs_q,
//                                 inputs_kv,
//                                 input_weights_q,
//                                 input_weights_kv,
//                                 output_weights,
//                                 dropout_mask,
//                                 pad_mask,
//                                 dropout_prob
//                                );
//}

} // end namespace cublas_gemmex
} // end namespace encdec 
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &multihead_attn::encdec_bias::cublaslt::fwd, "Encdec Multihead Attention Bias Forward.");
  m.def("backward", &multihead_attn::encdec_bias::cublaslt::bwd, "Encdec Multihead Attention Bias Backward.");
//  m.def("backward_input_only", &multihead_attn::encdec_bias::cublaslt::bwd_input_only, "Encdec Multihead Attention Bias Backward.");
//  m.def("backward_recompute", &multihead_attn::encdec::cublas_gemmex::bwd_recompute, "Encdec Multihead Attention Backward Recompute.");
}
