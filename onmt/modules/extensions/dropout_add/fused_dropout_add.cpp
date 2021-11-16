#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace fused_dropout_add {

// return O = dropout(X) + R and dropout mask M
std::vector<torch::Tensor> fwd_cuda(
                                   bool                 is_training,
                                   torch::Tensor const& input,
                                   torch::Tensor const& residual,
                                   float                dropout_prob
                                  );

// return gradX (gradR is the same as output grads, we can let pytorch handle it)
torch::Tensor bwd_cuda(
                                   torch::Tensor const& output_grads,
                                   torch::Tensor const& dropout_add_mask,
                                   float                dropout_prob
                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
                               bool                 is_training,
                               torch::Tensor const& input,
                               torch::Tensor const& residual,
                               float                dropout_prob
                                                 )
{
  // AT_ASSERTM(input.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(input.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(residual.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");

  return fwd_cuda(
                     is_training,
                     input,
                     residual,
                     dropout_prob
                    );
}

torch::Tensor bwd(
                               torch::Tensor const& output_grads,
                               torch::Tensor const& dropout_add_mask,
                               float                dropout_prob
                              )
{
    return bwd_cuda(
             output_grads,
             dropout_add_mask,
             dropout_prob
            );
}
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &multihead_attn::fused_dropout_add::fwd,
  "Fused Dropout and Residual Add connection.");
  m.def("backward", &multihead_attn::fused_dropout_add::bwd,
  "Fused Dropout and Residual Add Connection - Backward");
}