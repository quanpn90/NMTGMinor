#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

#include <stdio.h>


// Returns the reserved space (in elements) needed for the MLP
size_t get_mlp_reserved_linear_space(int batch_size, int num_layers, const int* output_features) {
  size_t res_space = 0;
  // Need to store output of every intermediate MLP - size equal to output_features[i] * batch_size
  // for all 'i' in [0, num_layers-1)
  for (int l = 0; l < num_layers; l++) {
    res_space += output_features[l] * batch_size;
  }
  return res_space;
}

// Returns the reserved space (in elements) needed for the MLP
size_t get_mlp_reserved_activation_space(int batch_size, int num_layers, const int* output_features) {
  size_t res_space = 0;
  // Need to store activations of every intermediate MLP except the last layer
  // for all 'i' in [0, num_layers-1)
  for (int l = 0; l < num_layers - 1; l++) {
    res_space += output_features[l] * batch_size;
  }
  return res_space;
}
