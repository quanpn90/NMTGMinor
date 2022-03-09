#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

#include <stdio.h>


template <typename T>
int linear_bias_forward_cuda(at::Tensor input, T *weight, at::Tensor bias,
int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace);

template <typename T>
int linear_bias_backward_cuda(bool compute_grad_input, T *input, T *weight, T *d_output, int in_features,
                              int batch_size, int out_features, T *d_weight, T *d_bias, T *d_input,  void *lt_workspace);

template <typename T>
int linear_bias_backward_input_only_cuda(T *input, T *weight, T *d_output, int in_features,
                              int batch_size, int out_features, T *d_input,  void *lt_workspace);


at::Tensor linear_bias_forward(at::Tensor input, at::Tensor weight, at::Tensor bias) {

  unsigned batch_size = 1;
  std::vector<int64_t> output_sizes;
  auto input_sizes = input.sizes();

  for (unsigned i=0; i < input_sizes.size() - 1 ; i++)  {
    batch_size = batch_size * input_sizes[i];
    output_sizes.push_back(input_sizes[i]);
  }

  auto in_features = input_sizes.back();
  int out_features = weight.size(0);
  output_sizes.push_back(out_features);

  // create output/workspace tensor
//  auto out = at::empty({batch_size, out_features}, input.type());
  auto out = at::empty(output_sizes, input.type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, input.type());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "linear_bias_forward", [&] {
    scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    scalar_t* b_ptr = bias.data_ptr<scalar_t>();
    auto result = linear_bias_forward_cuda<scalar_t>(
        input,
        w_ptr,
        bias,
        in_features,
        batch_size,
        out_features,
        out,
        (void*) (lt_workspace.data_ptr<scalar_t>()));
  });

  return {out};
}

std::vector<at::Tensor> linear_bias_backward(at::Tensor input, at::Tensor weight,
                                             at::Tensor d_output, bool compute_grad_input) {

  // compute batch size as the product of the first dimensions
  // -> more flexible input size
  unsigned batch_size = 1;
  auto input_sizes = input.sizes();
  for (unsigned i=0; i < input_sizes.size() - 1 ; i++)
    batch_size = batch_size * input_sizes[i];
  auto in_features = input_sizes.back();

  int out_features = weight.size(0);

  // create output/workspace tensor
  auto d_weight = at::empty({out_features, in_features}, input.type());

  auto d_bias = at::empty({out_features}, input.type());
//  auto d_input = at::empty({batch_size, in_features}, input.type());
  at::Tensor d_input;

  if (compute_grad_input)    {
    d_input = at::empty(input.sizes(), input.type());
  } else {
    d_input = at::empty({0}, input.type());
  }
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, input.type());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "linear_bias_backward", [&] {
    scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    scalar_t* d_b_ptr = d_bias.data_ptr<scalar_t>();
    auto result = linear_bias_backward_cuda<scalar_t>(
        compute_grad_input,
        input.data_ptr<scalar_t>(),
        w_ptr,
        d_output.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        out_features,
        d_weight.data_ptr<scalar_t>(),
        d_bias.data_ptr<scalar_t>(),
        d_input.data_ptr<scalar_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()));
  });

  return {d_input, d_weight, d_bias};
}


at::Tensor linear_bias_backward_input_only(at::Tensor input, at::Tensor weight, at::Tensor d_output) {

  // compute batch size as the product of the first dimensions
  // -> more flexible input size
  unsigned batch_size = 1;
  auto input_sizes = input.sizes();
  for (unsigned i=0; i < input_sizes.size() - 1 ; i++)
    batch_size = batch_size * input_sizes[i];
  auto in_features = input_sizes.back();

  int out_features = weight.size(0);

//  auto d_input = at::empty({batch_size, in_features}, input.type());
  auto d_input = at::empty(input.sizes(), input.type());
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, input.type());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "linear_bias_backward_input_only", [&] {
    scalar_t* w_ptr = weight.data_ptr<scalar_t>();
    auto result = linear_bias_backward_input_only_cuda<scalar_t>(
        input.data_ptr<scalar_t>(),
        w_ptr,
        d_output.data_ptr<scalar_t>(),
        in_features,
        batch_size,
        out_features,
        d_input.data_ptr<scalar_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()));
  });

  return d_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_bias_forward, "linear bias forward");
  m.def("backward", &linear_bias_backward, "linear bias backward");
  m.def("backward_input_only", &linear_bias_backward_input_only, "linear bias backward input only");
}