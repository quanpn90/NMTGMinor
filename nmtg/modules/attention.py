from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F

from nmtg.models.encoder_decoder import IncrementalModule
from nmtg.modules.linear import XavierLinear
from nmtg.modules.masking import MaskedFunction
from nmtg.sequence_generator import IncrementalState


class MultiHeadAttention(IncrementalModule):
    """
    Applies multi-head attentions to inputs (query, key, value)
    Args:
        model_dim:      dimension of model
        num_heads:      number of heads
        dropout:        dropout probability
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

    def forward(self, query, key, value, attention_bias=None, query_mask=None, value_mask=None):
        """
        Perform the forward pass
        :param query: (len_query x batch_size x model_dim) | (batch_size x len_query x model_dim)
        :param key: (len_key x batch_size x model_dim) | (batch_size x len_key x model_dim)
        :param value: (len_key x batch_size x model_dim) | (batch_size x len_key x model_dim)
        :param attention_bias: (batch_size x len_query x len_key) or broadcastable
        :param query_mask (len_query x batch_size) | (batch_size x len_query)
        :param value_mask (len_key x batch_size) | (batch_size x len_key)
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

        if not self.masked_layers:
            query_mask = None
            value_mask = None
        elif qkv_same:
            value_mask = query_mask

        if self.batch_first:
            batch_size, len_query, _ = query.size()
        else:
            len_query, batch_size, _ = query.size()

        q, k, v = self._project_inputs(query, key, value, query_mask, value_mask, batch_size, len_query)

        out = self._attention(q, k, v, attention_bias, query_mask, batch_size, len_query)

        return out

    def _step(self, query, key, value, incremental_state: IncrementalState, attention_bias=None,
              query_mask=None, value_mask=None, static_kv=False,):
        """
        static_kv: If true, do not recalculate keys and values, they have not changed since
            the last timestep (used for self-attention when incremental_state is None)
        incremental_state: Instance of incremental_state for step-wise decoding
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        if not self.masked_layers:
            query_mask = None
            value_mask = None
        elif qkv_same:
            value_mask = query_mask

        batch_size = query.size(0 if self.batch_first else 1)

        saved_state = incremental_state.get(self, 'attn_state', {})
        if 'prev_key' in saved_state:
            # previous time steps are cached - no need to recompute
            # key and value if they are static
            if static_kv:
                assert kv_same and not qkv_same
                key = value = None

        q, k, v = self._project_inputs(query, key, value, query_mask, value_mask, batch_size, 1)

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

        out = self._attention(q, k, v, attention_bias, query_mask, batch_size, 1)

        return out

    def _project_inputs(self, query, key, value, query_mask, value_mask, batch_size, len_query):
        # Project inputs to multi-heads
        # Projection has the same dimensionality as input, gets split into heads later
        q = self.query_projection(query, query_mask)
        k = self.key_projection(key, value_mask) if key is not None else None
        v = self.value_projection(value, value_mask) if value is not None else None
        q *= self.head_dim ** -0.5

        # prepare the shape for applying softmax
        q = self._split_heads(q, batch_size, len_query)
        if k is not None:
            k = self._split_heads(k, batch_size, -1)
        if v is not None:
            v = self._split_heads(v, batch_size, -1)
        return q, k, v

    def _attention(self, q, k, v, attention_bias, query_mask, batch_size, len_query):
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
        out = self._join_heads(out, batch_size, len_query)
        out = self.out_projection(out, query_mask)
        return out

    def _split_heads(self, tensor, batch_size, seq_len):
        if self.batch_first:
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) \
                .contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        else:
            return tensor.view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

    def _join_heads(self, out, batch_size, len_query):
        if self.batch_first:
            return out.view(batch_size, self.num_heads, len_query, self.head_dim).transpose(1, 2) \
                .contiguous().view(batch_size, len_query, self.model_dim)
        else:
            return out.transpose(0, 1).contiguous().view(len_query, batch_size, self.model_dim)

    def reorder_incremental_state(self, incremental_state: IncrementalState, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = incremental_state.get(self, 'attn_state', {})
        for k in input_buffer.keys():
            input_buffer[k] = input_buffer[k].index_select(0, new_order)
        incremental_state.set(self, 'attn_state', input_buffer)
