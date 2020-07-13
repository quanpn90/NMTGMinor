# coding=utf-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch REFORMER model. """

import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from onmt.modules.lsh_attention import LSHSelfAttention
from onmt.models.transformers import PrePostProcessing
from onmt.modules.linear import FeedForward
from typing import Callable, Dict, List, Optional, Tuple


def apply_chunking_to_forward(
    chunk_size: int, chunk_dim: int, forward_fn: Callable[..., torch.Tensor], *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size`
    over the dimension `chunk_dim`.
    It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the
    same result as not applying it.
    Args:
        chunk_size: int - the chunk size of a chunked tensor. `num_chunks` = `len(input_tensors[0]) / chunk_size`
        chunk_dim: int - the dimension over which the input_tensors should be chunked
        forward_fn: fn - the forward fn of the model
        input_tensors: tuple(torch.Tensor) - the input tensors of `forward_fn` which are chunked
    Returns:
        a Tensor with the same shape the foward_fn would have given if applied
    Examples::
        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states
        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.chunk_size_lm_head, self.seq_len_dim, self.forward_chunk, hidden_states)
    """

    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape
    assert all(
        input_tensor.shape == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compability
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


class ReformerEncoderLayer(nn.Module):

    def __init__(self, opt, death_rate=0.0):
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        d_model = opt.model_size
        p = opt.dropout

        super(ReformerEncoderLayer, self).__init__()

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        self.self_attention = LSHSelfAttention(opt)
        self.feedforward = FeedForward(opt.model_size, opt.inner_size, opt.dropout, opt.variational_dropout)

    def forward(self, input, attn_mask):

        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            query = self.preprocess_attn(input)
            out, _, _ = self.self_attention(query, attn_mask)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

        return input