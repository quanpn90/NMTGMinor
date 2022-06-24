#include <torch/extension.h>

void multi_tensor_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale);

void multi_tensor_axpby_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float a,
  float b,
  int arg_to_check);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale,
  at::optional<bool> per_tensor_python);

void multi_tensor_adam_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int mode,
  const int bias_correction,
  const float weight_decay);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors");
  m.def("multi_tensor_axpby", &multi_tensor_axpby_cuda,
        "out = a*x + b*y for a list of contiguous tensors");
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_l2norm_scale", &multi_tensor_l2norm_scale_cuda,
        "Computes L2 norm for a list of contiguous tensors and does scaling");
  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
}