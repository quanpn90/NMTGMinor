import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt
import torch.nn.functional as F
from onmt.modules.bottle import Bottle
from onmt.modules.static_dropout import StaticDropout
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import group_linear


class MultiHeadAttention(nn.Module):
    """Applies multi-head attentions to inputs (query, key, value)
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity

    Params:
        fc_query:  FC layer to project query, d_model x (h x d_head)
        fc_key:    FC layer to project key,   d_model x (h x d_head)
        fc_value:  FC layer to project value, d_model x (h x d_head)
        fc_concat: FC layer to concat and project multiheads, d_model x (h x d_head)

    Inputs Shapes:
        query: batch_size x len_query x d_model
        key:   batch_size x len_key x d_model
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable

    Outputs Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key

    """

    def __init__(self, h, d_model, attn_p=0.1, static=False, share=3):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d = d_model
        self.share = share

        assert d_model % h == 0

        self.d_head = d_model // h
        self.fc_query = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_key = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_value = Bottle(Linear(d_model, h * self.d_head, bias=False))
        self.fc_concat = Bottle(Linear(h * self.d_head, d_model, bias=False))

        self.sm = nn.Softmax(dim=-1)

        if static:
            self.attn_dropout = StaticDropout(attn_p)
        else:
            self.attn_dropout = nn.Dropout(attn_p)

    def forward(self, query, key, value, mask,
                incremental=False, incremental_cache=None):

        len_query, b = query.size(0), query.size(1)

        # batch_size*h x len_query x d_head
        # project inputs to multi-heads
        if self.share == 1:
            shared_qkv = group_linear(
                [self.fc_query.function.linear, self.fc_key.function.linear, self.fc_value.function.linear], query)
            proj_query, proj_key, proj_value = shared_qkv.chunk(3, dim=-1)

            # In incremental case: we concatenate the previously computed (mapped) states to the proj_key and proj_v
            if incremental:
                if incremental_cache is not None and 'k' in incremental_cache and 'v' in incremental_cache:
                    proj_key = torch.cat([incremental_cache['k'], proj_key], dim=0)  # time first
                    incremental_cache['k'] = proj_key
                    proj_value = torch.cat([incremental_cache['v'], proj_value], dim=0)  # time first
                    incremental_cache['v'] = proj_value
                    len_key, b_ = proj_key.size(0), proj_key.size(1)
                else:
                    if incremental_cache is None:
                        incremental_cache = dict()
                    incremental_cache['k'] = proj_key
                    incremental_cache['v'] = proj_value

        elif self.share == 2:

            # This function will have to change in the future for Transformer XL
            proj_query = self.fc_query(query)  # batch_size x len_query x h*d_head
            shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
            proj_key, proj_value = shared_kv.chunk(2, dim=-1)

            if incremental:
                if incremental_cache is not None and 'c_k' in incremental_cache and 'c_v' in incremental_cache:
                    proj_key = incremental_cache['c_k']
                    proj_value = incremental_cache['c_v']
                else:
                    if incremental_cache is None:
                        incremental_cache = dict()
                    shared_kv = group_linear([self.fc_key.function.linear, self.fc_value.function.linear], key)
                    proj_key, proj_value = shared_kv.chunk(2, dim=-1)
                    incremental_cache['c_k'] = proj_key
                    incremental_cache['c_v'] = proj_value

        else:
            proj_query = self.fc_query(query)
            proj_key = self.fc_key(key)  # batch_size x len_key x h*d_head
            proj_value = self.fc_value(value)  # batch_size x len_key x h*d_head

        q, k, v = proj_query, proj_key, proj_value
        len_key, b_ = k.size(0), k.size(1)
        # prepare the shape for applying softmax
        q = q.contiguous().view(len_query, b * self.h, self.d_head).transpose(0, 1)
        k = k.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)
        v = v.contiguous().view(len_key, b * self.h, self.d_head).transpose(0, 1)

        q = q * (self.d_head ** -0.5)

        # get dotproduct softmax attns for each head
        attns = torch.bmm(q, k.transpose(1, 2))  # batch_size*h x len_query x len_key

        attns = attns.view(b, self.h, len_query, len_key)
        mask_ = mask.unsqueeze(-3)
        # FP16 support: cast to float and back
        attns = attns.float().masked_fill_(mask_, -float('inf')).type_as(attns)
        attns = F.softmax(attns.float(), dim=-1).type_as(attns)
        # return mean attention from all heads as coverage
        coverage = torch.mean(attns, dim=1)
        attns = self.attn_dropout(attns)
        attns = attns.view(b * self.h, len_query, len_key)

        # apply attns on value
        out = torch.bmm(attns, v)  # batch_size*h x len_query x d_head
        out = out.transpose(0, 1).contiguous().view(len_query, b, self.d)

        out = self.fc_concat(out)

        return out, coverage, incremental_cache
