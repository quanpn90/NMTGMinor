#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/python.h>
//#include <ATen/native/cuda/block_reduce.cuh>



template <typename T>
torch::Tensor kernel_coefficient_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor out, T* a_ptr);

template <typename T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kernel_coefficient_backward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor dout,
    torch::Tensor da,
    torch::Tensor db,
    torch::Tensor dc,
    T* a_ptr);


torch::Tensor ssm_kernel_coefficient_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {

    const auto IC = a.size(0);
    const auto Q = a.size(1); // num heads
    const auto H = a.size(2); // input dim
    const auto N = a.size(3); // hidden dim
    const auto L = b.size(1); // seq length

    auto out = torch::empty({IC, Q, L, H}, torch::dtype(a.dtype()).device(a.device()));

    AT_DISPATCH_FLOATING_TYPES(a.type(), "ssm_kernel_coefficient_forward", [&] {

        auto result = kernel_coefficient_forward<scalar_t>(a, b, c, out, a.data_ptr<scalar_t>());
    });

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ssm_kernel_coefficient_backward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor dout) {

    const auto IC = a.size(0);
    const auto Q = a.size(1); // num heads
    const auto H = a.size(2); // input dim
    const auto N = a.size(3); // hidden dim
    const auto L = b.size(1); // seq length

    auto da = torch::empty({IC, Q, H, N}, torch::dtype(a.dtype()).device(a.device()));
    auto db = torch::empty({Q, L, H}, torch::dtype(b.dtype()).device(b.device()));
    auto dc = torch::empty({Q, N}, torch::dtype(c.dtype()).device(c.device()));


    AT_DISPATCH_FLOATING_TYPES(a.type(), "ssm_kernel_coefficient_backward", [&] {
        auto result = kernel_coefficient_backward<scalar_t>(a, b, c, dout, da, db, dc, a.data_ptr<scalar_t>());

    });
    return std::make_tuple(da, db, dc);


    }


 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("forward", &ssm_kernel_coefficient_forward, "Create Kernel efficiently in CUDA forward");
   m.def(
       "backward", &ssm_kernel_coefficient_backward, "Create Kernel efficiently in CUDA backward");
//   m.def(
//       "kernel_coefficient_backward_float", &kernel_coefficient_backward<float>);
//   m.def(
//       "kernel_coefficient_backward_double",
//       &kernel_coefficient_backward<double>);
 }