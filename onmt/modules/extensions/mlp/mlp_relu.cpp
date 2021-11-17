#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdio.h>

size_t get_mlp_reserved_space(int64_t batch_size, int num_layers, const int* output_features);

size_t get_mlp_activation_space(int64_t batch_size, int num_layers, const int* output_features);

template <typename T>
size_t get_mlp_bp_workspace_in_bytes(int batch_size, int num_layers, const int* output_features);

template <typename T>
int mlp_fp(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T** BPtr,
    T* Y,
    T* reserved_space,
    uint8_t* reserved_mask,
    void* lt_workspace,
    float p,
    bool store_dropout_mask);

template <typename T>
int mlp_bp(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    T* work_space,
    T* dX,
    T** dwPtr,
    T** dbPtr,
    bool requires_grad,
    float p);

template <typename T>
int mlp_bp_recompute(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    T** BPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    uint8_t* reserved_mask,
    T* work_space,
    T* dX,
    T** dwPtr,
    T** dbPtr,
    bool requires_grad,
    float p);

std::vector<at::Tensor> mlp_forward(float p, bool store_dropout_mask, std::vector<at::Tensor> inputs) {

  auto num_layers = inputs.size() - 1;
  num_layers /= 2;
  unsigned batch_size = 1;
  std::vector<int> output_features;
  for (int i = 0; i < num_layers; i++) {
    output_features.push_back(inputs[i + 1].size(0));
  }

  auto input_sizes = inputs[0].sizes();
  auto input_features = input_sizes.back();
  std::vector<int64_t> output_sizes;
  for (unsigned i=0; i < input_sizes.size() - 1 ; i++) {
    batch_size = batch_size * input_sizes[i];
    output_sizes.push_back(input_sizes[i]);
  }
  output_sizes.push_back(output_features.back());

  // if dropout is disabled then don't need to store anything
  if (p == 0.0) store_dropout_mask = false;

  auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  auto dmask_size    = 0;
  if (store_dropout_mask)
    dmask_size = get_mlp_activation_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
//  at::IntArrayRef output_size_array(output_sizes);
//  auto output_size_array = {input_sizes[0], input_sizes[1], output_features.back()};
  auto out = at::empty(output_sizes, inputs[0].type());

  auto reserved_space = at::empty({reserved_size}, inputs[0].type());

  auto act_options  = inputs[0].options().requires_grad(false);
  auto mask_options = act_options.dtype(torch::kUInt8);
  auto reserved_mask  = at::empty({dmask_size}, mask_options);  // for relu we don't need to keep the mask
  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  auto lt_workspace = at::empty({1 << 22}, inputs[0].type());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs[0].type(), "mlp_forward", [&] {
    std::vector<scalar_t*> w_ptr;
    std::vector<scalar_t*> b_ptr;
    for (int i = 0; i < num_layers; i++) {
      w_ptr.push_back(inputs[i + 1].data_ptr<scalar_t>());
      b_ptr.push_back(inputs[i + 1 + num_layers].data_ptr<scalar_t>());
    }
    auto result = mlp_fp<scalar_t>(
        inputs[0].data_ptr<scalar_t>(),
        input_features,
        batch_size,
        w_ptr.data(),
        num_layers,
        output_features.data(),
        b_ptr.data(),
        out.data_ptr<scalar_t>(),
        reserved_space.data_ptr<scalar_t>(),
        reserved_mask.data_ptr<uint8_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()),
        p,
        store_dropout_mask);
  });

  return {out, reserved_space, reserved_mask};
}

std::vector<at::Tensor> mlp_backward(
  float p,
  at::Tensor grad_o,
  std::vector<at::Tensor> fprop_outputs,
  std::vector<at::Tensor> inputs) {

  auto num_layers = inputs.size() - 1;
  num_layers /= 2;

  unsigned batch_size = 1;
  auto input_sizes = inputs[0].sizes();

  unsigned input_features = input_sizes.back();
  for (unsigned i=0; i < input_sizes.size() - 1 ; i++)
    batch_size = batch_size * input_sizes[i];

  bool requires_grad = inputs[0].requires_grad();

  std::vector<int> output_features;
  for (int i = 0; i < num_layers; i++) {
    output_features.push_back(inputs[i + 1].size(0));
  }
  // create outputs, length of inputs
  std::vector<at::Tensor> outputs;
  for (int i = 0; i < inputs.size(); i++) {
    outputs.push_back(at::empty(inputs[i].sizes(), inputs[i].type()));  // clone for testing now
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs[0].type(), "mlp_backward", [&] {
    std::vector<scalar_t*> w_ptr;
    for (int i = 0; i < num_layers; i++) {
      w_ptr.push_back(inputs[i + 1].data_ptr<scalar_t>());
    }
    std::vector<scalar_t*> outputs_ptr;
    for (int i = 0; i < inputs.size(); i++) {
      outputs_ptr.push_back(outputs[i].data_ptr<scalar_t>());
    }

    auto work_size =
        get_mlp_bp_workspace_in_bytes<scalar_t>(batch_size, num_layers, output_features.data());

    // auto work_space = at::empty({work_size*4}, at::kByte);
    auto work_space = at::empty({work_size / sizeof(scalar_t)}, inputs[0].type());

    auto result = mlp_bp<scalar_t>(
        inputs[0].data_ptr<scalar_t>(),
//        fprop_outputs[0].data_ptr<scalar_t>(),  // Y not necessary because at the output layer there is no activation
        input_features,
        batch_size,
        w_ptr.data(),
        num_layers,
        output_features.data(),
        grad_o.contiguous().data_ptr<scalar_t>(),
        fprop_outputs[1].data_ptr<scalar_t>(), // activations or reserved_space
        work_space.data_ptr<scalar_t>(),
        outputs_ptr[0],  //
        outputs_ptr.data() + 1,  // dweights
        outputs_ptr.data() + 1 + num_layers,  // dbiases
        requires_grad,
        p);
  });

  return outputs;
}
//
// Recompute version. Only requires input and grad out
std::vector<at::Tensor> mlp_backward_recompute(
  float p,
  at::Tensor grad_o,
  at::Tensor reserved_mask,
  std::vector<at::Tensor> inputs) {

  auto num_layers = inputs.size() - 1;
  num_layers /= 2;

  unsigned batch_size = 1;
  auto input_sizes = inputs[0].sizes();

  unsigned input_features = input_sizes.back();
  for (unsigned i=0; i < input_sizes.size() - 1 ; i++)
    batch_size = batch_size * input_sizes[i];

  bool requires_grad = inputs[0].requires_grad();

  std::vector<int> output_features;
  for (int i = 0; i < num_layers; i++) {
    output_features.push_back(inputs[i + 1].size(0));
  }
  // create outputs, length of inputs
  std::vector<at::Tensor> outputs;
  for (int i = 0; i < inputs.size(); i++) {
    // including input, weights and biases
    outputs.push_back(at::empty(inputs[i].sizes(), inputs[i].type()));  // clone for testing now
  }

  // instead of using activations, allocate similar to forward pass to recompute
  auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());
  auto reserved_space = at::empty({reserved_size}, inputs[0].type());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs[0].type(), "mlp_backward", [&] {
    std::vector<scalar_t*> w_ptr;
    std::vector<scalar_t*> b_ptr;
    for (int i = 0; i < num_layers; i++) {
      w_ptr.push_back(inputs[i + 1].data_ptr<scalar_t>());
      b_ptr.push_back(inputs[i + 1 + num_layers].data_ptr<scalar_t>());
    }
    std::vector<scalar_t*> outputs_ptr;
    for (int i = 0; i < inputs.size(); i++) {
      outputs_ptr.push_back(outputs[i].data_ptr<scalar_t>());
    }

    auto work_size =
        get_mlp_bp_workspace_in_bytes<scalar_t>(batch_size, num_layers, output_features.data());

    // auto work_space = at::empty({work_size*4}, at::kByte);
    auto work_space = at::empty({work_size / sizeof(scalar_t)}, inputs[0].type());

    auto result = mlp_bp_recompute<scalar_t>(
        inputs[0].data_ptr<scalar_t>(),
        input_features,
        batch_size,
        w_ptr.data(),
        b_ptr.data(),
        num_layers,
        output_features.data(),
        grad_o.contiguous().data_ptr<scalar_t>(),
        reserved_space.data_ptr<scalar_t>(),
        reserved_mask.data_ptr<uint8_t>(),
        work_space.data_ptr<scalar_t>(),
        outputs_ptr[0],
        outputs_ptr.data() + 1,
        outputs_ptr.data() + 1 + num_layers,
        requires_grad,
        p);
  });

  return outputs;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mlp_forward, "MLP forward");
  m.def("backward", &mlp_backward, "MLP backward");
  m.def("backward_recompute", &mlp_backward_recompute, "MLP backward");
}
