import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .relative_self_attention_func import relative_self_attn_func
from .relative_self_attention_func import RelativeShift


class RelativeSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., learnable_pos=False, max_pos=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True
        self.learnable_pos = learnable_pos
        self.autograd = False

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if self.learnable_pos:
            # If using learnable position embeddings, then assign embeddings for 2N + 1 max positions
            # (embeddings are shared across heads)
            assert max_pos >= 1
            self.pos_emb = nn.Embedding(2 * max_pos + 1, self.head_dim)
            self.pos_proj_weight, self.pos_proj_bias = None, None
        else:
            # Using sin/cos position encodings which are linearly projected to head_dim (seperately per head)
            self.pos_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.pos_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

        self.reset_parameters()
        self.attn_func = relative_self_attn_func

    def convert_autograd(self):

        if self.autograd:
            return

        self.autograd = True

        with torch.no_grad():

            self.in_linear = torch.nn.Linear(self.embed_dim, 3 * self.embed_dim)
            self.out_linear = torch.nn.Linear(self.embed_dim, self.embed_dim)

            if not self.learnable_pos:
                self.pos_linear = torch.nn.Linear(self.embed_dim, self.embed_dim)
                self.pos_linear.weight.copy_(self.pos_proj_weight)
                self.pos_linear.bias.copy_(self.pos_proj_bias)
                del self.pos_proj_weight
                del self.pos_proj_bias

            self.in_linear.weight.copy_(self.in_proj_weight)
            self.in_linear.bias.copy_(self.in_proj_bias)

            self.out_linear.weight.copy_(self.out_proj_weight)
            self.out_linear.bias.copy_(self.out_proj_bias)

            del self.in_proj_weight
            del self.out_proj_weight
            del self.in_proj_bias
            del self.out_proj_bias

    def reset_parameters(self, init='normal'):
        if init == 'normal':  # xavier normal
            std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            nn.init.normal_(self.in_proj_weight, 0.0, std_)
            nn.init.normal_(self.out_proj_weight, 0.0, std_)
            if self.pos_proj_weight is not None:
                nn.init.normal_(self.pos_proj_weight, 0.0, std_)

        else:
            std_ = math.sqrt(6.0 / (self.embed_dim + self.embed_dim))
            nn.init.uniform_(self.in_proj_weight, -std_, std_)
            nn.init.uniform_(self.out_proj_weight, -std_, std_)
            if self.pos_proj_weight is not None:
                nn.init.uniform_(self.pos_proj_weight, -std_, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)
        if self.pos_proj_bias is not None:
            nn.init.constant_(self.pos_proj_bias, 0.)

        nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        nn.init.normal_(self.r_r_bias, 0.0, 0.02)

    def forward(self, input, pos, key_padding_mask=None, attn_mask=None, mems=None,
                incremental=False, incremental_cache=None):
        """
        :param input: [T x B x H]
        :param pos: [T x 1 x H] or [T x T x H]
        :param key_padding_mask: [1 x T x B]
        :param attn_mask: [T x T]
        :param mems:
        :param incremental:
        :param incremental_cache:
        :return:
        """

        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
            if len(mask.shape) == 3:
                # [1 x T x B] -> [B x T]
                mask = mask.squeeze(0).transpose(0, 1)
        elif attn_mask is not None:
            mask = attn_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(-1)
        else:
            mask = None

        is_training = self.training

        if self.learnable_pos:
            # [len_q x len_k] -> [len_q x len_k x head_dim]
            pos = self.pos_emb(pos)

        if self.autograd:
            assert not self.training, "Auto-grad mode only used in Evaluation (for Quantization)."
            bsz = input.size(1)
            heads = self.num_heads
            head_dim = self.head_dim
            len_q = input.size(0)
            len_k = len_q
            input_lin_results = self.in_linear(input)
            scale_t = torch.tensor([head_dim ** -0.5])
            use_time_mask = attn_mask is not None

            if mask is not None:
                mask = mask.to(torch.bool)
                # Self Attention Time Mask
                if use_time_mask:
                    assert (len(mask.size()) == 2), "Timing mask is not 2D!"
                    mask = mask.unsqueeze(0).unsqueeze(0)
                # Key Padding Mask
                else:
                    mask = mask.unsqueeze(1).unsqueeze(2)

            if not self.learnable_pos:
                pos_lin_results = self.pos_linear(pos)
                r_head_k = pos_lin_results.view(pos.size(0), bsz * self.num_heads, self.head_dim)

            input_lin_results = input_lin_results.view(input.size(0), input.size(1) * self.num_heads, 3, self.head_dim)
            queries = input_lin_results[:, :, 0, :]
            keys = input_lin_results[:, :, 1, :]
            values = input_lin_results[:, :, 2, :]

            if incremental:
                # We have to change the heads x head_dim first and then concat to the T dim
                # bsz is changed during translation due to beam search
                # during translation we want to keep the actual T dim in MM as 1 constantly
                keys = keys.reshape(len_q, bsz, heads * head_dim)
                values = values.reshape(len_q, bsz, heads * head_dim)

                if 'k' in incremental_cache and 'v' in incremental_cache:
                    keys = torch.cat([incremental_cache['k'], keys], dim=0)  # time first
                    incremental_cache['k'] = keys
                    values = torch.cat([incremental_cache['v'], values], dim=0)  # time first
                    incremental_cache['v'] = values
                else:
                    incremental_cache['k'] = keys
                    incremental_cache['v'] = values

                keys = keys.view(-1, bsz * heads, head_dim)
                values = values.view(-1, bsz * heads, head_dim)
                # re-update len_k to be the newly updated length of the keys
                len_k = keys.size(0)

            rw_head_q = queries.view(len_q, bsz, heads, head_dim) + self.r_w_bias
            rw_head_q = rw_head_q.view(len_q, bsz * heads, head_dim)
            matmul_ac = torch.bmm(rw_head_q.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2)).mul_(scale_t[0])

            rr_head_q = queries.view(len_q, bsz, heads, head_dim) + self.r_r_bias
            rr_head_q = rr_head_q.view(len_q, bsz * heads, head_dim)

            if not self.learnable_pos:
                matmul_bd = torch.matmul(rr_head_q.transpose(0, 1), r_head_k.transpose(0, 1).transpose(1, 2)) \
                    .mul_(scale_t[0])
                matmul_bd = RelativeShift.forward(matmul_bd, True, False)
                matmul_bd = matmul_bd[:, :, :len_k]
                attn_score = matmul_ac + matmul_bd
            else:
                matmul_ac.transpose(0, 1).baddbmm_(rr_head_q, pos.transpose(1, 2), beta=1.0, alpha=scale_t[0])
                attn_score = matmul_ac

            if mask is not None:
                attn_score.view(bsz, heads, len_q, len_k).masked_fill_(mask, float('-inf'))

            softmax_results = F.softmax(attn_score, dim=-1)
            matmul2_results = torch.bmm(softmax_results, values.transpose(0, 1)).transpose(0, 1)
            matmul2_results = matmul2_results.contiguous().view(len_q, bsz, self.embed_dim)
            outputs = self.out_linear(matmul2_results)

            return outputs, softmax_results

        else:

            outputs, coverage = self.attn_func(input, pos, attn_mask is not None, is_training, self.num_heads,
                                               self.in_proj_weight, self.out_proj_weight, self.pos_proj_weight,
                                               self.in_proj_bias, self.out_proj_bias, self.pos_proj_bias,
                                               self.r_w_bias, self.r_r_bias,
                                               mask, self.dropout,
                                               incremental, incremental_cache, False,
                                               self.learnable_pos, True)
            # last Falses are double precision, learnable_embedding and return coverage

            return outputs, coverage
