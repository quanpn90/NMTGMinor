import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math

from ..optimized.relative_self_attention_func import relative_self_attn_func


class MFWRelativeSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., n_languages=1,
                 rank=1, use_multiplicative=False, weight_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True
        self.use_multiplicative = use_multiplicative
        self.weight_drop = weight_drop

        self.in_proj_weight = Parameter(torch.Tensor(embed_dim, 3 * embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.pos_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        self.pos_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.r_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, 3 * embed_dim))
        self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.r_p = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_p = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        if use_multiplicative:
            self.rm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, 3 * embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.rm_p = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_p = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

        self.reset_parameters()
        self.attn_func = relative_self_attn_func

    def reset_parameters(self, init='normal'):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)

        if init == 'normal':  # xavier normal
            std_ = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            nn.init.normal_(self.in_proj_weight, 0.0, std_)
            nn.init.normal_(self.out_proj_weight, 0.0, std_)
            nn.init.normal_(self.pos_proj_weight, 0.0, std_)

        else:
            std_ = math.sqrt(6.0 / (self.embed_dim + self.embed_dim))
            nn.init.uniform_(self.in_proj_weight, -std_, std_)
            nn.init.uniform_(self.out_proj_weight, -std_, std_)
            nn.init.uniform_(self.pos_proj_weight, -std_, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.constant_(self.pos_proj_bias, 0.)

        nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        nn.init.normal_(self.r_r_bias, 0.0, 0.02)

        nn.init.normal_(self.r_i, 0.0, 0.02)
        nn.init.normal_(self.s_i, 0.0, 0.02)
        nn.init.normal_(self.r_p, 0.0, 0.02)
        nn.init.normal_(self.s_p, 0.0, 0.02)
        nn.init.normal_(self.r_o, 0.0, 0.02)
        nn.init.normal_(self.s_o, 0.0, 0.02)

        if self.use_multiplicative:
            with torch.no_grad():
                self.rm_i.bernoulli_(0.5).mul_(-2).add_(1)
                self.sm_i.bernoulli_(0.5).mul_(-2).add_(1)
                self.rm_o.bernoulli_(0.5).mul_(-2).add_(1)
                self.sm_o.bernoulli_(0.5).mul_(-2).add_(1)
                self.rm_p.bernoulli_(0.5).mul_(-2).add_(1)
                self.sm_p.bernoulli_(0.5).mul_(-2).add_(1)

    def forward(self, input, pos, indices=None, key_padding_mask=None, attn_mask=None, mems=None,
                incremental=False, incremental_cache=None):

        if indices.size(0) == 1 and len(indices.shape) == 1:
            r_i = torch.index_select(self.r_i, 0, indices).squeeze(0)
            s_i = torch.index_select(self.s_i, 0, indices).squeeze(0)
            r_p = torch.index_select(self.r_p, 0, indices).squeeze(0)
            s_p = torch.index_select(self.s_p, 0, indices).squeeze(0)
            r_o = torch.index_select(self.r_o, 0, indices).squeeze(0)
            s_o = torch.index_select(self.s_o, 0, indices).squeeze(0)
        else:
            print(indices.size(), input.size())
            raise NotImplementedError

        # weight dropout
        in_proj_weight = F.dropout(self.in_proj_weight, p=self.dropout, training=self.training)
        pos_proj_weight = F.dropout(self.pos_proj_weight, p=self.dropout, training=self.training)
        out_proj_weight = F.dropout(self.out_proj_weight, p=self.dropout, training=self.training)

        if self.use_multiplicative:
            rm_i = torch.index_select(self.rm_i, 0, indices).squeeze(0)
            sm_i = torch.index_select(self.sm_i, 0, indices).squeeze(0)
            rm_p = torch.index_select(self.rm_p, 0, indices).squeeze(0)
            sm_p = torch.index_select(self.sm_p, 0, indices).squeeze(0)
            rm_o = torch.index_select(self.rm_o, 0, indices).squeeze(0)
            sm_o = torch.index_select(self.sm_o, 0, indices).squeeze(0)

            in_proj_weight = in_proj_weight * torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
            pos_proj_weight = pos_proj_weight * torch.bmm(rm_p.unsqueeze(-1), sm_p.unsqueeze(1)).sum(dim=0)
            out_proj_weight = out_proj_weight * torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

        in_proj_weight = in_proj_weight + torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
        pos_proj_weight = pos_proj_weight + torch.bmm(r_p.unsqueeze(-1), s_p.unsqueeze(1)).sum(dim=0)
        out_proj_weight = out_proj_weight + torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

        if key_padding_mask is not None:
            assert (attn_mask is None), "ERROR attn_mask and key_padding_mask should not be both defined!"
            mask = key_padding_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(0).transpose(0, 1)
        elif attn_mask is not None:
            mask = attn_mask
            if len(mask.shape) == 3:
                mask = mask.squeeze(-1)
        else:
            mask = None

        is_training = self.training

        outputs, coverage = self.attn_func(input, pos, attn_mask is not None, is_training, self.num_heads,
                                           in_proj_weight.t(), out_proj_weight.t(), pos_proj_weight.t(),
                                           self.in_proj_bias, self.out_proj_bias, self.pos_proj_bias,
                                           self.r_w_bias, self.r_r_bias,
                                           mask, self.dropout,
                                           incremental, incremental_cache, False, False)
        # last False is double precision

        return outputs, coverage


if __name__ == "__main__":
    bsz = 4
    seq_len_q = 4
    seq_len_kv = 7
    embed_dim = 32
    n_heads = 4
    output_size = 32
    n_languages = 7


    class TestNetwork(nn.Module):

        def __init__(self):
            super(TestNetwork, self).__init__()
            self.func = relative_self_attn_func

            self.n_heads = n_heads

        def forward(self, q, r, input_weights, output_weights, pos_weights,
                    input_biases, output_biases, pos_biases,
                    r_i, s_i, r_o, s_o, r_p, s_p,
                    r_w_bias, r_r_bias):

            use_time_mask = False
            mask = None
            is_training = True
            incremental = False
            incremental_cache = None
            double_precision = True
            dropout_prob = 0.0
            heads = self.n_heads

            output, coverage = self.func(q, r, use_time_mask, is_training, heads,
                                         input_weights, output_weights, pos_weights,
                                         input_biases, output_biases, pos_biases,
                                         r_i, s_i, r_o, s_o, r_p, s_p,
                                         r_w_bias, r_r_bias,
                                         mask, dropout_prob,
                                         incremental, incremental_cache, double_precision)

            return output


    r_w_bias = nn.Parameter(torch.Tensor(n_heads, embed_dim//n_heads)).double().cuda()
    r_r_bias = nn.Parameter(torch.Tensor(n_heads, embed_dim//n_heads)).double().cuda()

    in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim)).double().cuda()
    pos_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim)).double().cuda()
    out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim)).double().cuda()

    in_proj_bias = Parameter(torch.Tensor(3 * embed_dim)).double().cuda()
    pos_proj_bias = Parameter(torch.Tensor(embed_dim)).double().cuda()
    out_proj_bias = Parameter(torch.Tensor(embed_dim)).double().cuda()

    r_i = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    s_i = torch.nn.Parameter(torch.Tensor(bsz, 3 * embed_dim)).double().cuda()
    r_p = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    s_p = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    r_o = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    s_o = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()

    std_ = math.sqrt(2.0 / (embed_dim + embed_dim))
    nn.init.normal_(in_proj_weight, 0.0, std_)
    nn.init.normal_(pos_proj_weight, 0.0, std_)
    nn.init.normal_(out_proj_weight, 0.0, std_)
    nn.init.normal_(r_w_bias, 0.0, std_)
    nn.init.normal_(r_r_bias, 0.0, std_)

    torch.nn.init.constant_(in_proj_bias, 0.0)
    torch.nn.init.constant_(out_proj_bias, 0.0)
    torch.nn.init.constant_(pos_proj_bias, 0.0)

    with torch.no_grad():
        r_i.bernoulli_(0.5).mul_(-2).add_(1)
        s_i.bernoulli_(0.5).mul_(-2).add_(1)
        r_p.bernoulli_(0.5).mul_(-2).add_(1)
        s_p.bernoulli_(0.5).mul_(-2).add_(1)
        r_o.bernoulli_(0.5).mul_(-2).add_(1)
        s_o.bernoulli_(0.5).mul_(-2).add_(1)

    model = TestNetwork()

    q = torch.randn((seq_len_q, bsz, embed_dim), requires_grad=True)
    r = torch.randn((seq_len_kv, bsz, embed_dim), requires_grad=False)

    model = model.double().cuda()

    q = q.double().cuda()
    r = r.double().cuda()

    print("Gradchecking ...")
    torch.autograd.gradcheck(model, (q, r, in_proj_weight, out_proj_weight, pos_proj_weight,
                                     in_proj_bias, out_proj_bias, pos_proj_bias,
                                     r_i, s_i, r_o, s_o, r_p, s_p,
                                     r_w_bias, r_r_bias))
    print("Gradcheck successful!!!")