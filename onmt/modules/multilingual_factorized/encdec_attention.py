import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math

from ..optimized.encdec_attention_func import encdec_attn_func


class MFWEncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, num_heads, embed_dim, attn_drop=0.,
                 n_languages=1, rank=1, use_multiplicative=False, weight_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = attn_drop
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = False
        self.scaling = self.head_dim ** -0.5  # this value is hardcoded in the "fast" implementation
        self.use_multiplicative = use_multiplicative
        self.weight_drop = weight_drop

        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.r_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.r_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, 2 * embed_dim))
        self.s_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
        self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        if use_multiplicative:
            self.rm_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_q = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.rm_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, 2 * embed_dim))
            self.sm_kv = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, embed_dim))

        self.in_proj_bias_q = None
        self.in_proj_bias_kv = None
        self.out_proj_bias = None

        self.attn_func = encdec_attn_func

        try:
            # the fast one requires apex and does not work with incremental so careful
            from apex.contrib.multihead_attn.fast_encdec_multihead_attn_func import fast_encdec_attn_func
            self.attn_func_fast = fast_encdec_attn_func
            self.optimized = 1

        except ModuleNotFoundError as e:
            self.optimized = 2
            self.attn_func_fast = None

        self.reset_parameters()

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

        # torch.nn.init.constant_(self.in_proj_bias_q, 0.0)
        # torch.nn.init.constant_(self.in_proj_bias_kv, 0.0)
        # torch.nn.init.constant_(self.out_proj_bias, 0.0)
        nn.init.normal_(self.r_q, 0.0, math.sqrt(0.02))
        nn.init.normal_(self.s_q, 0.0, math.sqrt(0.02))
        nn.init.normal_(self.r_kv, 0.0, math.sqrt(0.02))
        nn.init.normal_(self.s_kv, 0.0, math.sqrt(0.02))
        nn.init.normal_(self.r_o, 0.0, math.sqrt(0.02))
        nn.init.normal_(self.s_o, 0.0, math.sqrt(0.02))

        if self.use_multiplicative:
            with torch.no_grad():
                self.rm_q.bernoulli_(0.5).mul_(-2).add_(1)
                self.sm_q.bernoulli_(0.5).mul_(-2).add_(1)
                self.rm_kv.bernoulli_(0.5).mul_(-2).add_(1)
                self.sm_kv.bernoulli_(0.5).mul_(-2).add_(1)
                self.rm_o.bernoulli_(0.5).mul_(-2).add_(1)
                self.sm_o.bernoulli_(0.5).mul_(-2).add_(1)

    def forward(self, query, key, value, src_indices=None, tgt_indices=None, attn_mask=None,
                incremental=False, incremental_cache=None):

        indices = tgt_indices
        n_languages = self.r_q.size(0)
        bsz = query.size(1)

        if indices.size(0) == 1 and len(indices.shape) == 1:
            r_q = torch.index_select(self.r_q, 0, indices).squeeze(0)
            s_q = torch.index_select(self.s_q, 0, src_indices).squeeze(0)
            r_kv = torch.index_select(self.r_kv, 0,  indices).squeeze(0)
            s_kv = torch.index_select(self.s_kv, 0, src_indices).squeeze(0)
            r_o = torch.index_select(self.r_o, 0, indices).squeeze(0)
            s_o = torch.index_select(self.s_o, 0, src_indices).squeeze(0)
        else:
            print(indices.size(), input.size())
            raise NotImplementedError

        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False
        len_key = key.size(0)

        # dropping the main weights during training
        in_proj_weight_q = F.dropout(self.in_proj_weight_q, p=self.weight_drop, training=self.training)
        in_proj_weight_kv = F.dropout(self.in_proj_weight_kv, p=self.weight_drop, training=self.training)
        out_proj_weight = F.dropout(self.out_proj_weight, p=self.weight_drop, training=self.training)

        if self.use_multiplicative:
            # multiply main weights with extra weights
            rm_q = torch.index_select(self.rm_q, 0, indices).squeeze(0)
            sm_q = torch.index_select(self.sm_q, 0, src_indices).squeeze(0)
            rm_kv = torch.index_select(self.rm_kv, 0, indices).squeeze(0)
            sm_kv = torch.index_select(self.sm_kv, 0, src_indices).squeeze(0)
            rm_o = torch.index_select(self.rm_o, 0, indices).squeeze(0)
            sm_o = torch.index_select(self.sm_o, 0, src_indices).squeeze(0)

            in_proj_weight_q = in_proj_weight_q * torch.bmm(rm_q.unsqueeze(-1), sm_q.unsqueeze(1)).sum(dim=0)
            in_proj_weight_kv = in_proj_weight_kv * torch.bmm(rm_kv.unsqueeze(-1), sm_kv.unsqueeze(1)).sum(dim=0)
            out_proj_weight = out_proj_weight * torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

        # adding main weights with extra weights
        # sum(dim=0) sums over the rank dimension
        in_proj_weight_q = in_proj_weight_q + torch.bmm(r_q.unsqueeze(-1), s_q.unsqueeze(1)).sum(dim=0)
        in_proj_weight_kv = in_proj_weight_kv + torch.bmm(r_kv.unsqueeze(-1), s_kv.unsqueeze(1)).sum(dim=0)
        out_proj_weight = out_proj_weight + torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

        if self.optimized == 1 and (self.training and not incremental) and len_key <= 1024 \
                and query.is_cuda and in_proj_weight_q.dtype == torch.half:
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask.squeeze(1)
                attn_mask = attn_mask.byte()

            outputs = self.attn_func_fast(time_masking, is_training, self.num_heads,
                                          query, key.type_as(in_proj_weight_q),
                                          in_proj_weight_q, in_proj_weight_kv, out_proj_weight,
                                          attn_mask, self.dropout)

            coverage = None

        # during evaluation we use the python binding which is safer ....
        else:
            outputs, coverage, = self.attn_func(time_masking, is_training,
                                                self.num_heads, query, key,
                                                in_proj_weight_q, in_proj_weight_kv,
                                                out_proj_weight, attn_mask, self.dropout,
                                                incremental, incremental_cache)

        # TODO: add incremental cache

        return outputs, coverage


if __name__ == "__main__":
    bsz = 4
    seq_len_q = 4
    seq_len_kv = 4
    embed_dim = 32
    n_heads = 4
    output_size = 32
    n_languages = 7


    class TestNetwork(nn.Module):

        def __init__(self):
            super(TestNetwork, self).__init__()
            self.func = EncdecAttnFunc.apply

            self.n_heads = 4

        def forward(self, q, kv, input_weights_q, input_weights_kv, output_weights,
                    input_biases_q, input_biases_kv, output_biases,
                    r_q, s_q, r_kv, s_kv):
            use_time_mask = False
            mask = None
            is_training = True
            incremental = False
            incremental_cache = None
            double_precision = True
            dropout_prob = 0.0
            heads = self.n_heads
            #
            # use_time_mask, is_training, heads, inputs_q, inputs_kv,
            # input_weights_q, input_weights_kv, output_weights,
            # input_biases_q, input_biases_kv, output_biases,
            # r_q, s_q, r_kv, s_kv, r_o, s_o,
            # mask, dropout_prob,
            # incremental, incremental_cache, double_precision

            output, coverage = self.func(use_time_mask, is_training, heads, q, kv,
                                         input_weights_q, input_weights_kv, output_weights,
                                         input_biases_q, input_biases_kv, output_biases,
                                         r_q, s_q, r_kv, s_kv,
                                         mask, dropout_prob,
                                         incremental, incremental_cache, double_precision)

            return output


    in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim)).double().cuda()
    in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim, embed_dim)).double().cuda()
    out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim)).double().cuda()

    in_proj_bias_q = Parameter(torch.Tensor(embed_dim)).double().cuda()
    in_proj_bias_kv = Parameter(torch.Tensor(2 * embed_dim)).double().cuda()
    out_proj_bias = Parameter(torch.Tensor(embed_dim)).double().cuda()

    r_q = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    s_q = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    r_kv = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    s_kv = torch.nn.Parameter(torch.Tensor(bsz, 2 * embed_dim)).double().cuda()
    # r_o = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()
    # s_o = torch.nn.Parameter(torch.Tensor(bsz, embed_dim)).double().cuda()

    std_ = math.sqrt(2.0 / (embed_dim + embed_dim))
    nn.init.normal_(in_proj_weight_q, 0.0, std_)
    nn.init.normal_(in_proj_weight_kv, 0.0, std_)
    nn.init.normal_(out_proj_weight, 0.0, std_)

    torch.nn.init.constant_(in_proj_bias_q, 0.0)
    torch.nn.init.constant_(in_proj_bias_kv, 0.0)
    torch.nn.init.constant_(out_proj_bias, 0.0)

    with torch.no_grad():
        r_q.bernoulli_(0.5).mul_(-2).add_(1)
        s_q.bernoulli_(0.5).mul_(-2).add_(1)
        r_kv.bernoulli_(0.5).mul_(-2).add_(1)
        s_kv.bernoulli_(0.5).mul_(-2).add_(1)
        r_o.bernoulli_(0.5).mul_(-2).add_(1)
        s_o.bernoulli_(0.5).mul_(-2).add_(1)

    # model = BEEncdecMultiheadAttn(n_heads, embed_dim, 0.0, n_languages)
    # model = Batchn_languagesLinear(embed_dim, output_size, n_languages)
    model = TestNetwork()

    q = torch.randn((seq_len_q, bsz, embed_dim), requires_grad=True)
    kv = torch.randn((seq_len_kv, bsz, embed_dim), requires_grad=True)

    model = model.double().cuda()

    q = q.double().cuda()
    kv = kv.double().cuda()

    print("Gradchecking ...")
    torch.autograd.gradcheck(model, (q, kv, in_proj_weight_q, in_proj_weight_kv, out_proj_weight,
                                     in_proj_bias_q, in_proj_bias_kv, out_proj_bias,
                                     r_q, s_q, r_kv, s_kv, r_o, s_o))


