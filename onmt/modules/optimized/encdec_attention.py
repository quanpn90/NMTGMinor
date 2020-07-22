import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .encdec_attention_func import encdec_attn_func

if hasattr(torch._C, '_jit_set_profiling_executor'):
    torch._C._jit_set_profiling_executor(False)
if hasattr(torch._C, '_jit_set_profiling_mode'):
    torch._C._jit_set_profiling_mode(False)


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
        try:
            # the fast one requires apex and does not work with incremental so careful
            from apex.contrib.multihead_attn.fast_encdec_multihead_attn_func import fast_encdec_attn_func
            self.attn_func_fast = fast_encdec_attn_func
            self.optimized = 2

        except ModuleNotFoundError as e:
            # print(e)
            # print("Cannot use fast self-attention implementation")
            self.optimized = 2
            self.attn_func_fast = None

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight_q)
        # in_proj_weight_kv has shape [2 * hidden, hidden] but it should be
        # initialized like a [hidden, hidden] matrix.
        # sqrt(6 / (hidden + hidden)) / sqrt(6 / (2 * hidden + hidden)) = sqrt(1.5)
        # therefore xavier_uniform gain should be set to sqrt(1.5).
        # nn.init.xavier_uniform_(self.in_proj_weight_kv, gain=math.sqrt(1.5))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
        nn.init.normal_(self.in_proj_weight_q, 0.0, std_)
        nn.init.normal_(self.in_proj_weight_kv, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)

    def forward(self, query, key, value, attn_mask=None, incremental=False, incremental_cache=None):

        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False
        len_key = key.size(0)

        if not query.is_cuda:
            return self.forward_autograd(time_masking, is_training,
                                         self.num_heads, query, key,
                                         self.in_proj_weight_q, self.in_proj_weight_kv,
                                         self.out_proj_weight, attn_mask, self.dropout,
                                         incremental, incremental_cache)

        if self.optimized == 1 and (self.training and not incremental) and len_key <= 1024:
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask.squeeze(1)
                attn_mask = attn_mask.byte()

            outputs = self.attn_func_fast(time_masking, is_training, self.num_heads, query, key,
                                          self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight,
                                          attn_mask, self.dropout)

            coverage = None

        # during evaluation we use the python binding which is safer ....
        else:
            outputs, coverage, = self.attn_func(time_masking, is_training,
                                                self.num_heads, query, key,
                                                self.in_proj_weight_q, self.in_proj_weight_kv,
                                                self.out_proj_weight, attn_mask, self.dropout,
                                                incremental, incremental_cache)

        # TODO: add incremental cache

        return outputs, coverage

    def forward_autograd(self, time_masking, is_training, heads, inputs_q, inputs_kv,
                         input_weights_q, input_weights_kv, output_weights, mask,
                         dropout, incremental, incremental_cache):

        heads_t = torch.tensor([heads])
        # dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs_q.size(2) // heads
        scale_t = torch.tensor([head_dim ** -0.5])

        bsz, len_q, len_k = inputs_q.size(1), inputs_q.size(0), inputs_kv.size(0)

        # TODO: add incremental cache

        input_lin_q_results = torch.mm(inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                                       input_weights_q.transpose(0, 1))
        input_lin_q_results = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1), input_weights_q.size(0))
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads, head_dim)

        if incremental and ('c_k' in incremental_cache and 'c_v' in incremental_cache):
            keys = incremental_cache['c_k']
            values = incremental_cache['c_v']
            keys = keys.view(len_k, bsz * heads, head_dim)
            values = values.view(len_k, bsz * heads, head_dim)
            input_lin_kv_results = torch.stack([keys, values], dim=-2)
        else:
            input_lin_kv_results = torch.mm(inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)),
                                            input_weights_kv.transpose(0, 1))
            input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1),
                                                             input_weights_kv.size(0))
            input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1) * heads, 2, head_dim)
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
        # matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2),
        #                                 beta=0.0, alpha=scale_t[0])

        if mask is not None:
            # Self Attention Time Mask
            mask = mask.to(torch.bool)

            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)  # for the head dimension
            else:
                mask = mask.unsqueeze(1).unsqueeze(2)  # for the head and query dimension

            batches, seql_q, seql_k = matmul1_results.size()
            bsz = int(batches / heads)
            matmul1_results = matmul1_results.view(bsz, heads, seql_q, seql_k)
            mask = mask.to(torch.bool)
            # after unsqueezing the mask should have size [bsz x 1 x 1 x seql_k]
            matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            matmul1_results = matmul1_results.view(bsz * heads, seql_q, seql_k)

        dtype_ = torch.float32
        softmax_results = F.softmax(matmul1_results.double(), dim=-1).type_as(matmul1_results)
        # softmax_results = F.softmax(matmul1_results.float(), dim=-1).type_as(matmul1_results)

        dropout_results = softmax_results
        dropout_mask = null_tensor

        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1))
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs_q.size(0), inputs_q.size(1),
                                                                            inputs_q.size(2))

        # Output Linear GEMM
        # Input1: (activations) [seql_q, seqs, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ] transpose(0,1)
        # Output:               [ seql_q, seqs, embed_dim ]
        # GEMM: ( seql_q*seqs x embed_dim ) x ( embed_dim x embed_dim ) = ( seql_q*seqs x embed_dim )
        outputs = torch.mm(matmul2_results.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                           output_weights.transpose(0, 1))
        outputs = outputs.view(inputs_q.size(0), inputs_q.size(1), output_weights.size(0))

        return outputs.detach(), softmax_results.detach()
