import torch
import torch.nn.functional as F
import torch.nn as nn
from onmt.modules.layer_norm import LayerNorm


c


class Adapter(torch.nn.Module):

    def __init__(self, input_dim, downsample_factor=2):
        self.input_dim = input_dim
        self.middle_dim = input_dim // downsample_factor
        super(Adapter, self).__init__()

        self.linear_in = nn.Linear(input_dim, self.middle_dim)
        self.linear_out = nn.Linear(self.middle_dim, input_dim)
        self.norm = LayerNorm(input_dim)

        self.fused = False
        from onmt.modules.mlp.mlp import mlp_relu_function
        if mlp_relu_function is not None:
            self.fused_function = mlp_relu_function
            self.fused = True
        self.reset_parameters()

    def reset_parameters(self):

        def normal_(data):
            # with FSDP, module params will be on CUDA, so we cast them back to CPU
            # so that the RNG is consistent with and without FSDP
            data.copy_(
                data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
            )
        with torch.no_grad():
            normal_(self.linear_in.weight.data)
            normal_(self.linear_out.weight.data)
            self.linear_in.bias.data.zero_()
            self.linear_out.bias.data.zero_()

    def forward(self, input):
        if self.fused:
            weights = [self.linear_in.weight, self.linear_out.weight]
            biases = [self.linear_in.bias, self.linear_out.bias]

            # seq_len, bsz, hidden_size = input.size(0), input.size(1), input.size(2)
            input_norm = self.norm(input)

            input = self.fused_function(0.0, False, input_norm,
                                        *weights, *biases)

            return input
        else:
            return self.linear_out(F.relu(self.linear_in(self.norm(input))))


class MultilingualAdapter(torch.nn.Module):

    def __init__(self, n_languages, input_size, downsample_factor=4):
        self.n_languages = n_languages
        self.input_size = input_size
        super(MultilingualAdapter, self).__init__()

        self.adapters = nn.ModuleList([Adapter(input_size, downsample_factor) for _ in range(self.n_languages)])

    def forward(self, input, lang=None, mixture=None):
        """
        :param input: tensor TxBxH
        :param lang: tensor size 1 (language for the batch)
        :param mixture: tensor size B x n_language (mixture for the minibatch)
        :return:
        """

        if lang is not None:
            assert mixture is None
            if lang.numel() != 1:
                print("Expected singled unit tensor, but get", lang.size())
            assert lang.numel() == 1
            adapter = self.adapters[lang.item()]

            return adapter(input)

        if mixture is not None:
            assert mixture.size(0) == input.size(1) and mixture.size(1) == self.n_languages
            outputs = list()

            for i in range(self.n_languages):
                # mixture size is [B x n_language]
                mixture_weight = mixture[:, i].unsqueeze(0).squeeze(-1)
                outputs.append(self.adapters[i](input)) * mixture_weight

            outputs = torch.stack(outputs).sum(0)  # n_languages x T x B x H
            outputs = torch.sum(outputs, 0, keepdim=False)  # -> T x B x H

        return outputs
