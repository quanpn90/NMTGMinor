import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math


# from onmt.constants import double_precision
# from .batch_ensemble_linear import BatchEnsembleMM as mm

class BatchEnsembleMM(object):

    @staticmethod
    def forward(x, weight, bias, ensemble_r, ensemble_s):
        """
        :param x: [T x B x H]
        :param weight: [H_out x H]
        :param bias: [H_out]
        :param ensemble_r: [B x H]
        :param ensemble_s: [B x H_out]
        :return:
        """
        bsz, len_x, hin = x.size(1), x.size(0), x.size(2)
        hout = weight.size(0)

        assert bsz == ensemble_s.size(0)
        # assert ensemble * bsz_per_ensemble == bsz, "Mini-batch must divide evenly to the ensembles"

        # element-wise [T x B x H] \times [B x H]
        x_r = torch.mul(x, ensemble_r)

        # GEMM No Bias. Otherwise use addmm
        x_mm = torch.mm(x_r.view(-1, hin), weight.transpose(0, 1))
        x_mm = x_mm.view(len_x, bsz, hout)

        # element-wise [T x B x Hout] \times [B x Hout]
        x_s = torch.mul(x_mm, ensemble_s)

        # add bias
        x_s = torch.add(x_s, bias)

        # we need to store the intermediate results for the backward pass
        return x_s, x_mm, x_r

    # maybe we need some allocated memory as well
    @staticmethod
    def backward(grad_y, x, x_r, x_mm, weight, ensemble_r, ensemble_s):
        bsz, len_x, hin = x.size(1), x.size(0), x.size(2)
        hout = x_mm.size(-1)

        grad_bias = torch.sum(grad_y, (0, 1))
        grad_s = grad_y

        # backprop through the last element-wise multiplication
        grad_ensemble_s = torch.mul(grad_s, x_mm)
        grad_ensemble_s = torch.sum(grad_ensemble_s, dim=0)

        # backprop through the MM
        grad_mm = torch.mul(grad_s, ensemble_s)
        grad_mm = grad_mm.view(-1, hout)
        grad_r = torch.mm(grad_mm, weight).view(len_x, bsz, hin)
        # GEMM: [hout x bsz] \times [bsz x hin]
        grad_weight = torch.mm(grad_mm.transpose(0, 1), x_r.view(-1, hin))

        # back prop through the first element-wise multiplication
        # element-wise [len_x, bsz, hin] \cdot [bsz, hin]
        grad_x = torch.mul(grad_r, ensemble_r)
        # grad ensemble r
        grad_ensemble_r = torch.mul(grad_r, x)
        grad_ensemble_r = torch.sum(grad_ensemble_r, dim=0)

        return grad_x, grad_weight, grad_bias, grad_ensemble_r, grad_ensemble_s


mm = BatchEnsembleMM


class EncdecAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, ensemble,
                inputs_q, inputs_kv,
                input_weights_q, input_weights_kv, output_weights,
                input_biases_q, input_biases_kv, output_biases,
                r_q, s_q, r_kv, s_kv,
                mask, dropout_prob,
                incremental, incremental_cache, double_precision):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs_q.size(2) // heads
        scale_t = torch.tensor([head_dim ** -0.5])

        bsz, len_q, len_k = inputs_q.size(1), inputs_q.size(0), inputs_kv.size(0)
        if not is_training:
            bsz = bsz // ensemble

        # TODO: add incremental cache

        # Linear Projection Q
        q, q_mm, q_r = mm.forward(inputs_q, input_weights_q, input_biases_q, r_q, s_q)

        if not is_training:
            q = q.view(len_q, ensemble, bsz, q.size(-1))
            q = torch.mean(q, dim=1)
        # input_lin_q_results = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1), input_weights_q.size(0))
        # print(q.size())
        queries = q.view(q.size(0), q.size(1) * heads, head_dim)

        # Linear Projection KV
        if incremental and ('c_k' in incremental_cache and 'c_v' in incremental_cache):
            keys = incremental_cache['c_k']
            values = incremental_cache['c_v']
            keys = keys.view(len_k, bsz * heads, head_dim)
            values = values.view(len_k, bsz * heads, head_dim)
            kv = torch.stack([keys, values], dim=-2)
        else:
            kv, kv_mm, kv_r = mm.forward(inputs_kv, input_weights_kv, input_biases_kv, r_kv, s_kv)
            if not is_training:
                kv = kv.view(kv.size(0), ensemble, kv.size(1) // ensemble, kv.size(-1))
                kv = torch.mean(kv, dim=1)
            kv = kv.view(kv.size(0), kv.size(1) * heads, 2, head_dim)
            keys = kv[:, :, 0, :]
            values = kv[:, :, 1, :]
            if incremental:
                keys = keys.contiguous().view(len_k, bsz, heads * head_dim)
                values = values.contiguous().view(len_k, bsz, heads * head_dim)

                incremental_cache['c_k'] = keys
                incremental_cache['c_v'] = values

                keys = keys.view(len_k, bsz * heads, head_dim)
                values = values.view(len_k, bsz * heads, head_dim)


        # Matmul1 Batched GEMMs
        # The output tensor is specified prior to the Batch GEMM because baddbmm requires its specification
        # baddbmm is used to apply the scale parameter via the Batched GEMM's alpha parameter instead of
        # a separate elementwise operation.
        # Input1: (Queries) [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Input2: (Keys)    [seql_k, seqs*heads, head_dim] transpose(0,1)
        # output:           [seqs*heads, seql_q, seql_k]
        # GEMM: Per batch: ( seql_q x head_dim ) x ( head_dim x seql_k ) = ( seql_q x seql_k )
        if queries.is_cuda:
            matmul1_results = torch.empty((queries.size(1), queries.size(0), keys.size(0)), dtype=queries.dtype,
                                          device=queries.device)
            matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0, 1),
                                            keys.transpose(0, 1).transpose(1, 2),
                                            out=matmul1_results, beta=0.0, alpha=scale_t[0])
        else:
            matmul1_results = torch.matmul(queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2))
            matmul1_results.mul_(scale_t[0])

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

        dtype_ = torch.float64 if double_precision else torch.float32
        softmax_results = F.softmax(matmul1_results, dim=-1, dtype=dtype_).type_as(matmul1_results)
        # softmax_results = F.softmax(matmul1_results.float(), dim=-1).type_as(matmul1_results)

        # Dropout - is not executed for inference
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=(1. - dropout_prob_t[0]))
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor

        # Matmul2 Batched GEMMs
        # The output tensor specification is needed here to specify the non-standard output.
        # Given that pytorch cannot currently perform autograd with an output tensor specified,
        # this requires a backward pass specified.
        # Input1: from_softmax [seqs*heads, seql_q, seql_k]
        # Input2: (values)     [seql_v, seqs*heads, head_dim] transpose(0,1)
        # Output:              [seql_q, seqs*heads, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_q x seql_k ) x ( seql_k x head_dim ) = (seql_q x head_dim)
        if queries.is_cuda:
            matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)),
                                          dtype=dropout_results.dtype, device=dropout_results.device).transpose(1, 0)
            matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        else:
            matmul2_results = torch.matmul(dropout_results, values.transpose(0, 1))
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(len_q, bsz, inputs_q.size(2))


        # # Output Linear GEMM
        # # Input1: (activations) [seql_q, seqs, embed_dim=heads*head_dim]
        # # Input2: (weights)     [ embed_dim, embed_dim ] transpose(0,1)
        # # Output:               [ seql_q, seqs, embed_dim ]
        # # GEMM: ( seql_q*seqs x embed_dim ) x ( embed_dim x embed_dim ) = ( seql_q*seqs x embed_dim )
        # outputs = torch.mm(matmul2_results.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
        #                    output_weights.transpose(0, 1))
        # outputs = outputs.view(inputs_q.size(0), inputs_q.size(1), output_weights.size(0))
        # Output Linear Projection
        o_input = matmul2_results
        # o, o_mm, o_r = mm.forward(o_input, output_weights, output_biases, r_o, s_o)
        o = torch.addmm(output_biases,
                        o_input.view(len_q * bsz, o_input.size(2)),
                        output_weights.transpose(0, 1),
                        beta=1., alpha=1.)

        outputs = o.view(len_q, bsz, output_weights.size(0))

        ctx.save_for_backward(heads_t,
                              scale_t,
                              matmul2_results,
                              dropout_results,
                              softmax_results,
                              q, q_mm, q_r,
                              kv, kv_mm, kv_r,
                              inputs_q,
                              inputs_kv,
                              input_weights_q, input_biases_q, r_q, s_q,
                              input_weights_kv, input_biases_kv, r_kv, s_kv,
                              output_weights, output_biases,
                              dropout_mask,
                              dropout_prob_t)

        # return o.detach()
        with torch.no_grad():
            softmax_results = softmax_results.new(*softmax_results.size()).copy_(softmax_results)

        return outputs.detach(), softmax_results

    @staticmethod
    def backward(ctx, output_grads, softmax_grads):

        heads_t, scale_t, matmul2_results, dropout_results, softmax_results \
            , q, q_mm, q_r, kv, kv_mm, kv_r \
            , inputs_q, inputs_kv \
            , input_weights_q, input_biases_q, r_q, s_q \
            , input_weights_kv, input_biases_kv, r_kv, s_kv \
            , output_weights, output_biases \
            , dropout_mask, dropout_prob_t \
            = ctx.saved_tensors

        head_dim = inputs_q.size(2) // heads_t[0]

        # Slice out k,v from one big Input Linear output (should only impact meta data, no copies!)
        # Batch sizes and heads are combined to make the batch of the Batched GEMM
        # input_lin_kv_results: [seql_k, bsz, heads(16), 2, head_dim(64)]
        # input_lin_kv_results: [seql_k, batches=bsz*heads, 2, head_dim]
        queries = q.view(inputs_q.size(0), inputs_q.size(1) * heads_t[0], head_dim)
        kv = kv.view(inputs_kv.size(0), inputs_kv.size(1) * heads_t[0], 2, head_dim)
        keys = kv[:, :, 0, :]
        values = kv[:, :, 1, :]

        # Slice out k,v from one big set of gradients entering the input linear's bprop
        # (should only impact meta data, no copies!)
        # The gradients are identical in size to the Input Linear outputs.
        # The tensor is declared before hand to properly slice out query, key, and value grads.
        kv_grads = torch.empty_like(kv)
        queries_grads = torch.empty_like(queries)
        keys_grads = kv_grads[:, :, 0, :]
        values_grads = kv_grads[:, :, 1, :]

        # Output Linear Projection
        o_input = matmul2_results

        # output_lin_grads, output_weights_grads, output_biases_grads, r_o_grads, s_o_grads \
        #     = mm.backward(output_grads, o_input, o_r, o_mm, output_weights, r_o, s_o)
        output_lin_grads = torch.mm(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        output_weights_grads = torch.mm(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1),
            matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        output_biases_grads = torch.sum(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1) * heads_t[0],
                                                 head_dim).transpose(0, 1)

        # Matmul2 - DGRAD1
        # Input1: (data grads)  [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Input2: (activations) [seql_k, seqs*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [seqs*heads, seql_q, seql_k]
        # GEMM: Per batch: ( seql_q x head_dim ) x ( head_dim x seql_k ) = ( seql_q x seql_k )

        # print(output_lin_grads.size(), values.size())
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))

        # Matmul2 - DGRAD2
        # Input1: (data grads)  [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Input2: (activations) [seql_k, seqs*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [seqs*heads, seql_q, seql_k]
        # GEMM: Per batch: ( seql_q x head_dim ) x ( head_dim x seql_k ) = ( seql_q x seql_k )

        torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))

        # Mask and Scaling for Dropout (not a publically documented op)
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))

        # Softmax Grad (not a publically documented op)
        softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)

        # Matmul1 - DGRAD1
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k]
        # Input2: (activations) [seql_k, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_q, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_q x seql_k ) x ( seql_k x head_dim ) = ( seql_q x head_dim )
        queries_grads = torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1),
                                      out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        # Matmul1 - DGRAD2
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k] transpose(1,2)
        # Input2: (activations) [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_k, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_k x seql_q ) x ( seql_q x head_dim ) = ( seql_k x head_dim )

        torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1),
                      out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])

        # Input Q Linear GEMM - DGRAD

        # input1: (data grads) [seql_q, seqs, embed_dim(1024)]
        # input2: (weights)    [embed_dim (1024), embed_dim (1024)]
        # output:              [seql_q, seqs, embed_dim]
        # GEMM: ( (seql_q*seqs) x embed_dim ) x ( embed_dim x embed_dim ) = (seql_q*seqs x embed_dim)
        queries_grads = queries_grads.transpose(0, 1).view(inputs_q.size(0), inputs_q.size(1), heads_t[0] * head_dim)

        # print("Reached 2 here")
        # print(queries_grads.size(), q_r.size(), q_mm.size())
        inputs_q_grads, input_weights_q_grads, input_biases_q_grads, r_q_grads, s_q_grads \
            = mm.backward(queries_grads, inputs_q, q_r, q_mm, input_weights_q, r_q, s_q)

        kv_grads = kv_grads.view(inputs_kv.size(0), inputs_kv.size(1), heads_t[0] * 2 * head_dim)

        inputs_kv_grads, input_weights_kv_grads, input_biases_kv_grads, r_kv_grads, s_kv_grads \
            = mm.backward(kv_grads, inputs_kv, kv_r, kv_mm, input_weights_kv, r_kv, s_kv)

        return None, None, None, None \
            , inputs_q_grads, inputs_kv_grads \
            , input_weights_q_grads, input_weights_kv_grads, output_weights_grads \
            , input_biases_q_grads, input_biases_kv_grads, output_biases_grads \
            , r_q_grads, s_q_grads, r_kv_grads, s_kv_grads \
            , None, None, None, None, None


# encdec_attn_func = EncdecAttnFunc.apply

class BEEncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, num_heads, embed_dim, attn_drop=0., ensemble=1):
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

        self.in_proj_bias_q = Parameter(torch.Tensor(embed_dim))
        self.in_proj_bias_kv = Parameter(torch.Tensor(2 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.r_q = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        self.s_q = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        self.r_kv = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        self.s_kv = torch.nn.Parameter(torch.Tensor(ensemble, 2 * embed_dim))
        # self.r_o = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        # self.s_o = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))

        self.attn_func = EncdecAttnFunc.apply

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

        torch.nn.init.constant_(self.in_proj_bias_q, 0.0)
        torch.nn.init.constant_(self.in_proj_bias_kv, 0.0)
        torch.nn.init.constant_(self.out_proj_bias, 0.0)

        with torch.no_grad():
            self.r_q.bernoulli_(0.5).mul_(-2).add_(1)
            self.s_q.bernoulli_(0.5).mul_(-2).add_(1)
            self.r_kv.bernoulli_(0.5).mul_(-2).add_(1)
            self.s_kv.bernoulli_(0.5).mul_(-2).add_(1)
            # self.r_o.bernoulli_(0.5).mul_(-2).add_(1)
            # self.s_o.bernoulli_(0.5).mul_(-2).add_(1)

    def forward(self, query, key, value, attn_mask=None, incremental=False, incremental_cache=None,
                indices=None, double_precision=False):

        assert value is key, "ERROR: Keys and values must be the same."

        is_training = self.training
        time_masking = False
        len_key = key.size(0)

        ensemble = self.r_q.size(0)
        bsz = query.size(1)

        if is_training:
            if indices is None:
                    with torch.no_grad():
                        indices = torch.arange(0, bsz, device=query.device, dtype=torch.long)
                        indices = torch.remainder(indices, ensemble)

            r_q = torch.index_select(self.r_q, 0, indices)
            s_q = torch.index_select(self.s_q, 0, indices)

            r_kv = torch.index_select(self.r_kv, 0, indices)
            s_kv = torch.index_select(self.s_kv, 0, indices)
            #
            # r_o = torch.index_select(self.r_o, 0, indices)
            # s_o = torch.index_select(self.s_o, 0, indices)
        else:
            query = query.repeat(1, ensemble, 1)
            key = key.repeat(1, ensemble, 1)
            # attn_mask = attn_mask.repeat(ensemble, 1, 1)
            r_q = self.r_q.repeat(bsz, 1).view(bsz, ensemble, self.r_q.size(-1)).\
                        transpose(0, 1).contiguous().view(-1, self.r_q.size(-1))
            s_q = self.s_q.repeat(bsz, 1).view(bsz, ensemble, self.s_q.size(-1)).\
                        transpose(0, 1).contiguous().view(-1, self.s_q.size(-1))
            r_kv = self.r_kv.repeat(bsz, 1).view(bsz, ensemble, self.r_kv.size(-1)).\
                        transpose(0, 1).contiguous().view(-1, self.r_kv.size(-1))
            s_kv = self.s_kv.repeat(bsz, 1).view(bsz, ensemble, self.s_kv.size(-1)).\
                        transpose(0, 1).contiguous().view(-1, self.s_kv.size(-1))
            # r_o = self.r_o.repeat(bsz, 1).view(bsz, ensemble, self.r_o.size(-1)).\
            #             transpose(0, 1).contiguous().view(-1, self.r_o.size(-1))
            # s_o = self.s_o.repeat(bsz, 1).view(bsz, ensemble, self.s_o.size(-1)).\
            #             transpose(0, 1).contiguous().view(-1, self.r_o.size(-1))

        outputs, coverage = self.attn_func(time_masking, is_training, self.num_heads, ensemble,
                                           query, key,
                                           self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight,
                                           self.in_proj_bias_q, self.in_proj_bias_kv, self.out_proj_bias,
                                           r_q, s_q, r_kv, s_kv, attn_mask, self.dropout,
                                           incremental, incremental_cache, double_precision)

        return outputs, coverage


if __name__ == "__main__":
    bsz = 4
    seq_len_q = 4
    seq_len_kv = 4
    embed_dim = 32
    n_heads = 4
    output_size = 32
    ensemble = 7


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

    # model = BEEncdecMultiheadAttn(n_heads, embed_dim, 0.0, ensemble)
    # model = BatchEnsembleLinear(embed_dim, output_size, ensemble)
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
