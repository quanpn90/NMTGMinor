#include <torch/extension.h>
#include <vector>

torch::Tensor blru_forward(
    torch::Tensor Lambda_exp,
    torch::Tensor Bu,
    torch::Tensor lengths,
    int direction);

std::vector<torch::Tensor> blru_backward(
    torch::Tensor grad_output,
    torch::Tensor Lambda_exp,
    torch::Tensor output,
    torch::Tensor lengths,
    int direction);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &blru_forward, "BLRU Forward");
    m.def("backward", &blru_backward, "BLRU Backward");
}