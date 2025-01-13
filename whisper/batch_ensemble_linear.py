import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEnsembleLinear(nn.Module):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BatchEnsembleLinear(nn.Linear):
        def __init__(self, in_features, out_features, num_ensembles,
                     bias=True, device=None, dtype=None,
                     init='constant'):
            # Initialize nn.Linear
            super(BatchEnsembleLinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
            self.num_ensembles = num_ensembles

            # Rank-1 factors for each ensemble member
            factory_kwargs = {"device": device, "dtype": dtype}
            self.r = nn.Parameter(torch.empty(num_ensembles, out_features, **factory_kwargs))  # Output modulation
            self.s = nn.Parameter(torch.empty(num_ensembles, in_features, **factory_kwargs))  # Input modulation

            # Initialize modulation parameters
            self.init_method = init
            self.reset_modulation_parameters()

    def reset_modulation_parameters(self):
        """
        Initializes the rank-1 modulation parameters (r and s).

        """

        if self.init_method == 'constant':
            nn.init.constant_(self.r, 1.0)
            nn.init.constant_(self.s, 1.0)
        elif self.init_method == 'normal_one':
            nn.init.normal_(self.r, mean=1., std=0.1)
            nn.init.normal_(self.s, mean=1., std=0.1)
        elif self.init_method == 'random_sign':
            nn.init.constant_(self.r, 1.)
            nn.init.constant_(self.s, 1.)
            with torch.no_grad():
                alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                alpha_coeff.mul_(2).add_(-1)
                gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                gamma_coeff.mul_(2).add_(-1)

                self.r *= alpha_coeff
                self.gamma *= gamma_coeff

        # nn.init.normal_(self.r, mean=0, std=0.1)
        # nn.init.normal_(self.s, mean=0, std=0.1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_ensembles={self.num_ensembles})"
        )

    def forward(self, x, training=True):
        if training:
            # Training input: [batch_size, length, hidden_size]
            batch_size, length, hidden_size = x.shape

            if batch_size == self.num_ensembles:
                # Case 1: B == E; directly modulate with r and s
                modulated_r = self.r.unsqueeze(1)  # Shape: [num_ensembles, 1, out_features]
                modulated_s = self.s.unsqueeze(1)  # Shape: [num_ensembles, 1, in_features]
                modulated_x = x * modulated_s  # Element-wise modulation
            else:
                # Case 2: B != E; randomize ensemble assignments
                ensemble_idx = torch.randint(0, self.num_ensembles, (batch_size,), device=x.device)
                modulated_r = self.r[ensemble_idx].unsqueeze(1)  # Shape: [batch_size, 1, out_features]
                modulated_s = self.s[ensemble_idx].unsqueeze(1)  # Shape: [batch_size, 1, in_features]
                modulated_x = x * modulated_s  # Element-wise modulation

            # Apply shared linear transformation and modulation
            shared_output = F.linear(modulated_x, self.weight)  # Shape: [batch_size, length, out_features]
            return shared_output * modulated_r + self.bias

        else:
            # Testing input: [num_ensembles, batch_size, length, hidden_size]
            num_ensembles, batch_size, length, hidden_size = x.shape

            if num_ensembles != self.num_ensembles:
                raise ValueError(
                    f"Expected num_ensembles={self.num_ensembles}, but got {num_ensembles}."
                )

            # Expand r and s for broadcasting
            r_expanded = self.r.unsqueeze(1).unsqueeze(1)  # Shape: [num_ensembles, 1, 1, out_features]
            s_expanded = self.s.unsqueeze(1).unsqueeze(1)  # Shape: [num_ensembles, 1, 1, in_features]

            # Modulate input and apply shared weights
            modulated_x = x * s_expanded  # Shape: [num_ensembles, batch_size, length, in_features]
            shared_output = F.linear(modulated_x,
                                     self.weight)  # Shape: [num_ensembles, batch_size, length, out_features]

            # Modulate output and return mean over ensembles
            modulated_output = shared_output * r_expanded + self.bias
            # Shape: [num_ensembles, batch_size, length, out_features]
            return modulated_output
