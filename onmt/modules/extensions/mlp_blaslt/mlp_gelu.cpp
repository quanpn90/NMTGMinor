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
    T* reserved_activations,
    uint8_t* reserved_mask,
    void* lt_workspace,
    float p);

template <typename T>
int mlp_bp(
    T* X,
    T* Y,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    T* reserved_activations,
    uint8_t* reserved_mask,
    T* work_space,
    T* dX,
    T** dwPtr,
    T** dbPtr,
    bool requires_grad,
    void* lt_workspace,
    float p);

template <typename T>
int mlp_bp_input_only(
    T* X,
    T* Y,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    T* reserved_activations,
    uint8_t* reserved_mask,
    T* work_space,
    T* dX,
    bool requires_grad,
    void* lt_workspace,
    float p);


std::vector<at::Tensor> mlp_forward(float p, std::vector<at::Tensor> inputs) {

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

  auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());
  size_t dmask_size = 0;

  if (p > 0.0)
    dmask_size    = get_mlp_activation_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto out = at::empty(output_sizes, inputs[0].type());

  auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  auto reserved_activations = at::empty({reserved_size}, inputs[0].type());

  auto act_options  = inputs[0].options().requires_grad(false);
  auto mask_options = act_options.dtype(torch::kUInt8);
  auto reserved_mask  = at::empty({dmask_size}, mask_options);  // for relu we don't need the mask for bwd
  auto lt_workspace = torch::empty({1 << 22}, inputs[0].type());

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
        reserved_activations.data_ptr<scalar_t>(),
        reserved_mask.data_ptr<uint8_t>(),
        (void*) (lt_workspace.data_ptr<scalar_t>()),
        p);
  });

  return {out, reserved_space, reserved_activations, reserved_mask};
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

  auto input_features = input_sizes.back();
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
    auto lt_workspace = torch::empty({1 << 22}, inputs[0].type());

    auto result = mlp_bp<scalar_t>(
        inputs[0].data_ptr<scalar_t>(),
        fprop_outputs[0].data_ptr<scalar_t>(),
        input_features,
        batch_size,
        w_ptr.data(),
        num_layers,
        output_features.data(),
        grad_o.contiguous().data_ptr<scalar_t>(),
        fprop_outputs[1].data_ptr<scalar_t>(),
        fprop_outputs[2].data_ptr<scalar_t>(),
        fprop_outputs[3].data_ptr<uint8_t>(),
        work_space.data_ptr<scalar_t>(),
        outputs_ptr[0],
        outputs_ptr.data() + 1,
        outputs_ptr.data() + 1 + num_layers,
        requires_grad,
        (void*) (lt_workspace.data_ptr<scalar_t>()),
        p);
  });

  return outputs;
}


std::vector<at::Tensor> mlp_backward_input_only(
  float p,
  at::Tensor grad_o,
  std::vector<at::Tensor> fprop_outputs,
  std::vector<at::Tensor> inputs) {

  auto num_layers = inputs.size() - 1;
  num_layers /= 2;

  unsigned batch_size = 1;
  auto input_sizes = inputs[0].sizes();

  auto input_features = input_sizes.back();
  for (unsigned i=0; i < input_sizes.size() - 1 ; i++)
    batch_size = batch_size * input_sizes[i];

  bool requires_grad = inputs[0].requires_grad();

  std::vector<int> output_features;
  for (int i = 0; i < num_layers; i++) {
    output_features.push_back(inputs[i + 1].size(0));
  }
  // create outputs, length of inputs
  std::vector<at::Tensor> outputs;
  for (int i = 0; i < 1; i++) {
//  for (int i = 0; i < inputs.size(); i++) {
    outputs.push_back(at::empty(inputs[i].sizes(), inputs[i].type()));  // clone for testing now
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs[0].type(), "mlp_backward", [&] {
    std::vector<scalar_t*> w_ptr;
    for (int i = 0; i < num_layers; i++) {
      w_ptr.push_back(inputs[i + 1].data_ptr<scalar_t>());
    }
    std::vector<scalar_t*> outputs_ptr;
//    for (int i = 0; i < inputs.size(); i++) {
    for (int i = 0; i < 1; i++) {   // the first element is gradInput
      outputs_ptr.push_back(outputs[i].data_ptr<scalar_t>());
    }

    auto work_size =
        get_mlp_bp_workspace_in_bytes<scalar_t>(batch_size, num_layers, output_features.data());

//    auto work_space = at::empty({work_size / sizeof(scalar_t)}, inputs[0].type());
    auto work_space = at::empty({work_size / sizeof(scalar_t)}, inputs[0].type());
    auto lt_workspace = torch::empty({1 << 22}, inputs[0].type());

    auto result = mlp_bp_input_only<scalar_t>(
        inputs[0].data_ptr<scalar_t>(),
        fprop_outputs[0].data_ptr<scalar_t>(),
        input_features,
        batch_size,
        w_ptr.data(),
        num_layers,
        output_features.data(),
        grad_o.contiguous().data_ptr<scalar_t>(),
        fprop_outputs[1].data_ptr<scalar_t>(), // linear - activation
        fprop_outputs[2].data_ptr<scalar_t>(), // linear
        fprop_outputs[3].data_ptr<uint8_t>(), // mask
        work_space.data_ptr<scalar_t>(),
        outputs_ptr[0],
//        outputs_ptr.data() + 1,
//        outputs_ptr.data() + 1 + num_layers,
        requires_grad,
        (void*) (lt_workspace.data_ptr<scalar_t>()),
        p);
  });

  return outputs;
}






PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mlp_forward, "MLP forward");
  m.def("backward", &mlp_backward, "MLP backward");
  m.def("backward_input_only", &mlp_backward_input_only, "MLP backward");
}