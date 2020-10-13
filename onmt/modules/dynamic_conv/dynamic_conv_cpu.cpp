#include <torch/torch.h>
#include <vector>

std::vector<float*> dynamic_conv_cpu_forward(
    float* input,
    float* filters,
    int padding_l);

std::vector<float*> dynamic_conv_cpu_backward(
    float* gradOutput,
    int padding_l,
    float* input,
    float* filters);

std::vector<float*> dynamic_conv_forward(
    float* input,
    float* filters,
    int padding_l) {

    return dynamic_conv_cpu_forward(input, filters, padding_l);
}

std::vector<float*> dynamic_conv_backward(
    float* gradOutput,
    int padding_l,
    float* input,
    float* filters) {

    return dynamic_conv_cpu_backward(gradOutput, padding_l, input, filters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dynamic_conv_forward, "dynamic_conv forward (CPU)");
    m.def("backward", &dynamic_conv_backward, "dynamic_conv backward (CPU)");
}