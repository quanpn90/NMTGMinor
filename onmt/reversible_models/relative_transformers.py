import torch
import torch.nn as nn
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.linear import FeedForward
from onmt.modules.attention import MultiHeadAttention
from torch.autograd.function import Function
import sys
from torch.utils.checkpoint import get_device_states, set_device_states


class ReversibleEncoderFunction(Function):
    """
    Implementing the reversible encoder funcionality
    """

    @staticmethod
    def forward(ctx, hidden_states, position_encodings, layers, attn_mask):

        attn_output, hidden_states = torch.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            # forward pass in the layer
            attn_output, hidden_states = layer(
                attn_output, hidden_states, position_encodings, attn_mask
            )

        # attach params to ctx for backward

        # why should we detach here? because Y1 Y2 were built within torch.no_grad()
        # so cutting the backward from these variables seems unnecessary

        # save_for_backward will release memory more efficiently
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach(), position_encodings)
        # ctx.save_for_backward(attn_output, hidden_states)
        ctx.layers = layers
        ctx.attn_mask = attn_mask

        # summing the outputs for the encoder result
        with torch.no_grad():
            output = attn_output + hidden_states

        return output

    @staticmethod
    def backward(ctx, grad_hidden_states):

        # print(grad_hidden_states.sum())
        # grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)
        grad_attn_output = grad_hidden_states

        # retrieve params from ctx
        attn_output, hidden_states, position_encodings = ctx.saved_tensors
        layers = ctx.layers
        attn_mask = ctx.attn_mask

        for idx, layer in enumerate(layers[::-1]):
            # backprop
            attn_output, hidden_states, grad_attn_output, grad_hidden_states = layer.backward_pass(
                attn_output, hidden_states, grad_attn_output, grad_hidden_states, attn_mask
            )

        grad_hidden_states = torch.cat([grad_attn_output, grad_hidden_states], dim=-1)

        # we do not need the gradients for the position encodings
        return grad_hidden_states, None, None, None

