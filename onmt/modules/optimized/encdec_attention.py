import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .encdec_attention_func import encdec_attn_func


class EncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, num_heads, embed_dim, attn_drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = attn_drop
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = False
        self.scaling = self.head_dim ** -0.5  # this value is hardcoded in the "fast" implementation

        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.register_parameter('in_proj_bias_q', None)
        self.register_parameter('in_proj_bias_kv', None)
        self.in_proj_bias_q = None
        self.in_proj_bias_kv = None
        self.out_proj_bias = None

        self.attn_func = encdec_attn_func

        self.reset_parameters()
        self.autograd = False

    def convert_autograd(self):

        if self.autograd:
            return

        with torch.no_grad():

            self.autograd = True
            self.linear_q = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.linear_kv = torch.nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False)
            self.out_linear = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)

            self.linear_q.weight.copy_(self.in_proj_weight_q)
            self.linear_kv.weight.copy_(self.in_proj_weight_kv)
            self.out_linear.weight.copy_(self.out_proj_weight)

            del self.in_proj_weight_q
            del self.in_proj_weight_kv
            del self.out_proj_weight

    def reset_parameters(self, init='normal'):
        if init == 'normal':  # xavier normal
            std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            nn.init.normal_(self.in_proj_weight_q, 0.0, std_)
            nn.init.normal_(self.in_proj_weight_kv, 0.0, std_)
            nn.init.normal_(self.out_proj_weight, 0.0, std_)
        else:  # xavier uniform
            std_ = math.sqrt(6.0 / (self.embed_dim + self.embed_dim))
            nn.init.uniform_(self.in_proj_weight_q, -std_, std_)
            nn.init.uniform_(self.in_proj_weight_kv, -std_, std_)
            nn.init.uniform_(self.out_proj_weight, -std_, std_)

    def forward(self, query, key, value, attn_mask=None, incremental=False, incremental_cache=None):

        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False

        if self.autograd:

            assert not self.training
            mask = attn_mask

            if mask is not None:
                # Self Attention Pad Mask
                mask = mask.to(torch.bool)

                if len(mask.shape) == 3:
                    mask = mask.unsqueeze(1)  # for the head dimension
                else:
                    mask = mask.unsqueeze(1).unsqueeze(2)  # for the head and query dimension

            len_q = query.size(0)
            len_k = key.size(0)
            bsz = query.size(1)
            heads = self.num_heads
            head_dim = self.head_dim
            scale_t = torch.tensor([head_dim ** -0.5])

            input_lin_q_results = self.linear_q(query)
            queries = input_lin_q_results.view(len_q, bsz * heads, head_dim)

            if incremental and ('c_k' in incremental_cache and 'c_v' in incremental_cache):
                keys = incremental_cache['c_k']
                values = incremental_cache['c_v']
                keys = keys.view(len_k, bsz * heads, head_dim)
                values = values.view(len_k, bsz * heads, head_dim)
            else:
                input_lin_kv_results = self.linear_kv(key)

                input_lin_kv_results = input_lin_kv_results.view(len_k, bsz * heads, 2, head_dim)

                keys = input_lin_kv_results[:, :, 0, :]
                values = input_lin_kv_results[:, :, 1, :]
                if incremental:
                    keys = keys.contiguous().view(len_k, bsz, heads * head_dim)
                    values = values.contiguous().view(len_k, bsz, heads * head_dim)

                    incremental_cache['c_k'] = keys
                    incremental_cache['c_v'] = values

                    keys = keys.view(len_k, bsz * heads, head_dim)
                    values = values.view(len_k, bsz * heads, head_dim)

            matmul1_results = torch.matmul(queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2))
            matmul1_results.mul_(scale_t[0])

            if mask is not None:
                matmul1_results = matmul1_results.view(bsz, heads, len_q, len_k)
                # after unsqueezing the mask should have size [bsz x 1 x 1 x seql_k]
                matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
                matmul1_results = matmul1_results.view(bsz * heads, len_q, len_k)

            softmax_results = F.softmax(matmul1_results, dim=-1, dtype=torch.float32).type_as(matmul1_results)
            matmul2_results = torch.matmul(softmax_results, values.transpose(0, 1)).transpose(0, 1)
            matmul2_results = matmul2_results.contiguous().view(len_q, bsz, self.embed_dim)
            outputs = self.out_linear(matmul2_results)

            return outputs, softmax_results

        else:

            outputs, coverage = self.attn_func(time_masking, is_training,
                                                self.num_heads, query, key,
                                                self.in_proj_weight_q, self.in_proj_weight_kv,
                                                self.out_proj_weight, attn_mask, self.dropout,
                                                incremental, incremental_cache,
                                                False, True)  # double precision False and return coverage True

        return outputs, coverage

