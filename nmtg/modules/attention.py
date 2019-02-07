from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F

from nmtg.modules.dropout import StaticDropout
from nmtg.modules.linear import XavierLinear
from nmtg.modules.masking import MaskedFunction
from nmtg.sequence_generator import IncrementalState


class MultiHeadAttention(nn.Module):
    """
    Applies multi-head attentions to inputs (query, key, value)
    Args:
        model_dim:      dimension of model
        num_heads:      number of heads
        dropout:        dropout probability
        static_dropout: whether to use static dropout
        batch_first:    whether the inputs (and outputs) are batch first or time first

    Params:
        query_projection:  FC layer to project query, model_dim x (num_heads x head_dim)
        key_projection:    FC layer to project key,   model_dim x (num_heads x head_dim)
        value_projection:  FC layer to project value, model_dim x (num_heads x head_dim)
        out_projection: FC layer to concat and project multiheads, model_dim x (num_heads x head_dim)

    Inputs Shapes:
        query: batch_size x len_query x model_dim
        key:   batch_size x len_key x model_dim
        value: batch_size x len_key x model_dim
        If not batch_first, swap the batch_size and len arguments

        mask:  batch_size x len_query x len_key or broadcastable, regardless of batch_first

    Outputs Shapes:
        out:      batch_size x len_query x model_dim
        If not batch_first, batch and len_query are swapped

    Note:
        batch_first=False is very slightly slower (<10%)
    """

    def __init__(self, model_dim, num_heads, dropout=0.1, batch_first=False, masked_layers=False):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.batch_first = batch_first
        self.masked_layers = masked_layers

        assert model_dim % num_heads == 0

        self.head_dim = model_dim // num_heads

        self.query_projection = MaskedFunction(XavierLinear(model_dim, model_dim, bias=False))
        self.key_projection = MaskedFunction(XavierLinear(model_dim, model_dim, bias=False))
        self.value_projection = MaskedFunction(XavierLinear(model_dim, model_dim, bias=False))

        self.out_projection = MaskedFunction(XavierLinear(model_dim, model_dim, bias=False))

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attention_bias=None, query_mask=None, value_mask=None, static_kv=False,
                incremental_state: IncrementalState = None):
        """
        Perform the forward pass
        :param query: (len_query x batch_size x model_dim) | (batch_size x len_query x model_dim)
        :param key: (len_key x batch_size x model_dim) | (batch_size x len_key x model_dim)
        :param value: (len_key x batch_size x model_dim) | (batch_size x len_key x model_dim)
        :param attention_bias: (batch_size x len_query x len_key) or broadcastable
        :param query_mask (len_query x batch_size) | (batch_size x len_query)
        :param value_mask (len_key x batch_size) | (batch_size x len_key)
        :param static_kv: If true, do not recalculate keys and values, they have not changed since
            the last timestep (used for self-attention when incremental_state is None)
        :param incremental_state: Instance of incremental_state for step-wise decoding
        :return: (len_query x batch_size x model_dim) | (batch_size x len_query x model_dim)

        Implementation notes Felix 2019-02-02:
        This is mostly the same as the fairseq implementation.
        I simplified the q/k/v projection. I tested and found no time benefit to calculating
        the projection in one go. If there is one, it is cancelled out by having to call
        .contiguous() afterwards.
        This implementation is very slightly (~5%) faster than Quan's previous one, which may
        be purely due to not using Bottle (aka MaskedFunction). The batch_first implementation
        Follows the procedure from tensor2tensor and is very slightly slower than length first
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        if not self.masked_layers:
            query_mask = None
            value_mask = None
        elif qkv_same:
            value_mask = query_mask

        if self.batch_first:
            batch_size, len_query, _ = query.size()
        else:
            len_query, batch_size, _ = query.size()

        if incremental_state is not None:
            saved_state = incremental_state.get(self, 'attn_state', {})
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        # Project inputs to multi-heads
        # Projection has the same dimensionality as input, gets split into heads later
        q = self.query_projection(query, query_mask)
        k = self.key_projection(key, value_mask) if key is not None else None
        v = self.value_projection(value, value_mask) if value is not None else None
        q *= self.head_dim ** -0.5

        # prepare the shape for applying softmax
        if self.batch_first:
            # batch_size x num_heads x seq_len x head_dim
            q = q.view(batch_size, len_query, self.num_heads, self.head_dim).transpose(1, 2) \
                .contiguous().view(batch_size * self.num_heads, len_query, self.head_dim)
            if k is not None:
                k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) \
                    .contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
            if v is not None:
                v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) \
                    .contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
        else:
            # batch_size*num_heads x seq_len x head_dim
            q = q.view(len_query, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            if k is not None:
                k = k.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
            if v is not None:
                v = v.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(batch_size * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(batch_size * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(batch_size, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(batch_size, self.num_heads, -1, self.head_dim)
            incremental_state.set(self, 'attn_state', saved_state)

        len_key = k.size(1)

        # get dotproduct softmax attns for each head
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights.view(batch_size, self.num_heads, len_query, len_key)

        # FP16 support: cast to float and back
        attn_weights = attn_weights.float()

        if attention_bias is not None:
            attn_weights += attention_bias.unsqueeze(1)

        attn_weights = F.softmax(attn_weights, dim=-1).type_as(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # apply attns on value
        attn_weights = attn_weights.view(batch_size * self.num_heads, len_query, len_key)
        out = torch.bmm(attn_weights, v)  # batch_size*num_heads x len_query x head_dim
        if self.batch_first:
            out = out.view(batch_size, self.num_heads, len_query, self.head_dim).transpose(1, 2)\
                .contiguous().view(batch_size, len_query, self.model_dim)
        else:
            out = out.transpose(0, 1).contiguous().view(len_query, batch_size, self.model_dim)

        out = self.out_projection(out, query_mask)

        return out

    def reorder_incremental_state(self, incremental_state: IncrementalState, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = incremental_state.get(self, 'attn_state', {})
        for k in input_buffer.keys():
            input_buffer[k] = input_buffer[k].index_select(0, new_order)
        incremental_state.set(self, 'attn_state', input_buffer)
