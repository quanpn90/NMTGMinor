import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math


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
    def backward(grad_y, x, x_r, x_mm, weight, ensemble_r, ensemble_s, need_grad_x=True):
        bsz, len_x, hin = x.size(1), x.size(0), x.size(2)
        hout = x_mm.size(-1)

        grad_bias = grad_y
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
        if need_grad_x :
            grad_x = torch.mul(grad_r, ensemble_r)
        else:
            grad_x = None
        # grad ensemble r
        grad_ensemble_r = torch.mul(grad_r, x)
        grad_ensemble_r = torch.sum(grad_ensemble_r, dim=0)

        return grad_x, grad_weight, grad_bias, grad_ensemble_r, grad_ensemble_s


mm = BatchEnsembleMM


class RelativeShiftFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, batch_first, emb_last):
        assert len(x.shape) == 3, "Input must have 3 dimensions B x len_q x len_r!"

        ctx.batch_first = batch_first
        ctx.emb_last = emb_last
        return RelativeShift.forward(x, batch_first, emb_last)

    @staticmethod
    def backward(ctx, grad_x):
        batch_first = ctx.batch_first
        emb_last = ctx.emb_last
        return RelativeShift.backward(grad_x, batch_first, emb_last), False, False


class RelativeShift(object):

    @staticmethod
    def forward(x, batch_first, emb_last):
        assert len(x.shape) == 3, "Input must have 3 dimensions B x len_q x len_r or len_q x len_r x demb!"
        assert (batch_first or emb_last) and not(batch_first and emb_last), \
            "Batch first and Embedding last must be mutually exclusive"

        if batch_first:
            bsz = x.size(0)
            zero_pad = torch.zeros((bsz, x.size(1), 1),
                                   device=x.device, dtype=x.dtype)

            # padded into [T x T+1 x (B x H)]
            x_padded = torch.cat([zero_pad, x], dim=2)

            # view into [T+1 x T x (BxH)]
            x_view = x_padded.view(bsz, x.size(2) + 1, x.size(1))

            # remove the first collumn
            x = x_view[:, 1:].view_as(x)
        else:
            raise NotImplementedError

        return x

    @staticmethod
    def backward(grad_x, batch_first, emb_last):

        if batch_first:
            bsz = grad_x.size(0)
            len_q, len_r = grad_x.size(1), grad_x.size(2)

            grad_x_view = grad_x.view(bsz, len_r, len_q)

            zero_pad = torch.zeros((bsz, 1, len_q), device=grad_x.device, dtype=grad_x.dtype)

            # grad_x should have size B x len_q x len_r
            # x_view should have size B x len_q+1 x len_r

            # put the zeros into the missing gradients
            grad_x_view = torch.cat([zero_pad, grad_x_view], dim=1)
            # print(grad_x_view.size())
            grad_x_padded = grad_x_view.view(bsz, len_q, len_r + 1)

            # because the first index in the padded dim was from zero_pad
            grad_output = grad_x_padded[:, :, 1:]
        else:
            raise NotImplementedError

        return grad_output


class RelativeSelfAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, pos, use_time_mask, is_training, heads, ensemble,
                input_weights, output_weights, pos_weights,
                input_biases, output_biases, pos_biases,
                r_i, s_i, r_p, s_p,
                r_w_bias, r_r_bias,
                mask, dropout_prob,
                incremental, incremental_cache,
                double_precision):
        """
        :param double_precision: ops at float64, only for debugging
        :param ctx: context object to stash information for backward
        :param inputs: input hidden states [len_q x batch_size x hidden]
        :param pos: [len_k x 1 x hidden]
        :param use_time_mask: bool, if we use the causal mask for decoder
        :param is_training: training state, for dropout
        :param heads: number of heads
        :param input_weights: weight matrix [hidden x 3*hidden]
        :param output_weights: output weight [hidden x hidden]
        :param input_biases: bias [3*hidden]
        :param output_biases: output bias [bias]
        :param pos_biases:
        :param pos_weights:
        :param r_w_bias:
        :param r_r_bias:
        :param mask: None or [B x T] or [T x T]
        :param dropout_prob:
        :param incremental:
        :param incremental_cache:
        :return:
        """

        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs.size(2) // heads
        scale_t = torch.tensor([head_dim ** -0.5])

        bsz, len_q = inputs.size(1), inputs.size(0)
        len_r = pos.size(0)  # r can be longer than query, i.e for bidirectional attention we need 2k+1 positions
        len_k = len_q  # because of self-attention
        if not is_training:
            bsz = bsz // ensemble

        if pos.size(1) == 1:
            pos = pos.repeat(1, bsz, 1)  # to T x B x H

        # # Input Linear GEMM
        # # input1: (activations) [len_q, bsz, hidden]
        # # input2: (weights)     [hidden*3 (3072), hidden (1024)] (transpose [0,1])
        # # output:               [len_q, bsz, hidden*3]
        # # GEMM: ( (len_q*bsz) x embed_dim ) x ( embed_dim x embed_dim*3 ) = (len_q*bsz x embed_dim*3)
        qkv, qkv_mm, qkv_r = mm.forward(inputs, input_weights, input_biases, r_i, s_i)
        if not is_training:
            qkv = qkv.view(len_q, ensemble, bsz, qkv.size(-1))
            qkv = torch.mean(qkv, dim=1)

        rpos, rpos_mm, rpos_r = mm.forward(pos, pos_weights, pos_biases, r_p, s_p)
        if not is_training:
            rpos = rpos.view(len_r, ensemble, bsz, rpos.size(-1))
            rpos = torch.mean(rpos, dim=1)

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # input_lin_results: [len_q, bsz, heads(16), 3, head_dim(64)]
        # input_lin_results: [len_q, batches=bsz*heads, 3, head_dim]
        qkv = qkv.view(len_q, bsz * heads, 3, head_dim)
        queries = qkv[:, :, 0, :]
        keys = qkv[:, :, 1, :]
        values = qkv[:, :, 2, :]

        r_head_k = rpos.view(pos.size(0), bsz * heads, head_dim)  # T x BxH x D

        if incremental:
            # We have to change the heads x head_dim first and then concat to the T dim
            # bsz is changed during translation due to beam search
            # during translation we want to keep the actual T dim in MM as 1 constantly
            keys = keys.contiguous().view(len_q, bsz, heads * head_dim)
            values = values.contiguous().view(len_q, bsz, heads * head_dim)
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

        # Relative Attention from here:
        # r_w_bias size: head * head_dim
        rw_head_q = queries.view(len_q, bsz, heads, head_dim) + r_w_bias  #
        rw_head_q = rw_head_q.view(len_q, bsz * heads, head_dim)

        # matmul1 batched GEMMs
        # queries+bias: [len_q, bsz*heads, head_dim] transpose(0, 1)
        # keys: [len_k, bsz*heads, head_dim] transpose(0, 1)
        if queries.is_cuda:
            matmul_ac = torch.empty((bsz * heads, queries.size(0), keys.size(0)), dtype=queries.dtype,
                                    device=rw_head_q.device)
            matmul_ac = torch.baddbmm(matmul_ac, rw_head_q.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2),
                                      out=matmul_ac, beta=0.0, alpha=scale_t[0])
        else:
            matmul_ac = torch.bmm(rw_head_q.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2)).mul_(scale_t[0])

        rr_head_q = queries.view(len_q, bsz, heads, head_dim) + r_r_bias  #
        rr_head_q = rr_head_q.view(len_q, bsz * heads, head_dim)

        # matmul2 batched GEMMs
        # queries+bias: [len_q, bsz*heads, head_dim] transpose(0, 1)
        # rel_positions: [len_r, bsz*heads, head_dim] transpose(0, 1)
        if queries.is_cuda:
            matmul_bd = torch.empty((bsz * heads, queries.size(0), len_r), dtype=queries.dtype,
                                    device=rw_head_q.device)
            matmul_bd = torch.baddbmm(matmul_bd, rr_head_q.transpose(0, 1), r_head_k.transpose(0, 1).transpose(1, 2),
                                      out=matmul_bd, beta=0.0, alpha=scale_t[0])
        else:
            matmul_bd = torch.matmul(rr_head_q.transpose(0, 1), r_head_k.transpose(0, 1).transpose(1, 2)) \
                .mul_(scale_t[0])

        # shift so that the relative positions are aligned
        # the first element will have 0 -1 ... -n relative positions compared to other elements
        # the last element will have  n-1 n-2 ...  0
        matmul_bd = RelativeShift.forward(matmul_bd, True, False)

        # if len_r is longer than len_k, then we need to take the first len_k positions only
        matmul_bd = matmul_bd[:, :, :len_k]

        attn_score = matmul_ac + matmul_bd  # both AC and BD are scaled with scale_t before in baddbmm
        # attn_score should have size [bsz*heads, len_q, len_k] for now
        if mask is not None:
            # Self Attention Time Mask
            if use_time_mask:
                assert (len(mask.size()) == 2), "Timing mask is not 2D!"
                # assert (mask.size(0) == mask.size(1)), "Sequence length should match!"
                mask = mask.to(torch.bool)
                attn_score = attn_score.masked_fill_(mask, float('-inf'))
            # Key Padding Mask
            else:
                batches, len_q, seql_k = attn_score.size()
                bsz = int(batches / heads)
                attn_score = attn_score.view(bsz, heads, len_q, seql_k)
                mask = mask.to(torch.bool)
                attn_score = attn_score.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                attn_score = attn_score.view(bsz * heads, len_q, seql_k)

        dtype_ = torch.float64 if double_precision else torch.float32
        softmax_results = F.softmax(attn_score, dim=-1, dtype=dtype_).type_as(attn_score)

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
        # Input1: from_softmax [bsz*heads, len_q, seql_k]
        # Input2: (values)     [seql_v, bsz*heads, head_dim] transpose(0,1)
        # Output:              [len_q, bsz*heads, head_dim] transpose(0,1)
        # GEMM: Per batch: ( len_q x seql_k ) x ( seql_k x head_dim ) = (len_q x head_dim)
        matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)),
                                      dtype=dropout_results.dtype, device=queries.device).transpose(1, 0)
        torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(len_q, bsz, inputs.size(2))

        # Output Linear GEMM
        # Input1: (activations) [len_q, bsz, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ] transpose(0,1)
        # Output:               [ len_q, bsz, embed_dim ]
        # GEMM: ( len_q*bsz x embed_dim ) x ( embed_dim x embed_dim ) = ( len_q*bsz x embed_dim )
        # outputs = torch.addmm(output_biases,
        #                       matmul2_results.view(inputs.size(0) * inputs.size(1), inputs.size(2)),
        #                       output_weights.transpose(0, 1),
        #                       beta=1., alpha=1.)
        #
        # outputs = outputs.view(inputs.size(0), inputs.size(1), output_weights.size(0))
        o_input = matmul2_results
        # o, o_mm, o_r = mm.forward(o_input, output_weights, output_biases, r_o, s_o)
        # outputs = o
        outputs = torch.addmm(output_biases,
                              matmul2_results.view(len_q * bsz, inputs.size(2)),
                              output_weights.transpose(0, 1),
                              beta=1., alpha=1.)

        outputs = outputs.view(len_q, bsz, output_weights.size(0))

        ctx.save_for_backward(heads_t,
                              scale_t,
                              matmul2_results,  #
                              dropout_results,
                              softmax_results,
                              qkv, qkv_mm, qkv_r,
                              rpos_r, rpos_mm,
                              rw_head_q, rr_head_q,
                              inputs, pos, r_head_k,
                              input_weights, pos_weights, output_weights,
                              r_i, s_i, r_p, s_p,
                              r_w_bias, r_r_bias,
                              dropout_mask,
                              dropout_prob_t)

        # with torch.no_grad():
        #     coverage = softmax_results.new(*softmax_results.size()).copy_(softmax_results)
        coverage = softmax_results

        return outputs.detach(), coverage
        # return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads, softmax_grads):
        # def backward(ctx, output_grads):
        """
        :param ctx:
        :param output_grads: gradients w.r.t the outputs
        :param softmax_grads: unncessary except we use the attention weights somewhere
        :return:
        """
        heads_t, \
            scale_t, \
            matmul2_results, \
            dropout_results, \
            softmax_results, \
            qkv, qkv_mm, qkv_r, \
            rpos_r, rpos_mm, \
            rw_head_q, rr_head_q, \
            inputs, pos, r_head_k, \
            input_weights, pos_weights, output_weights, \
            r_i, s_i, r_p, s_p, \
            r_w_bias, r_r_bias, \
            dropout_mask, \
            dropout_prob_t = ctx.saved_tensors

        head_dim = inputs.size(2) // heads_t[0]
        len_q, bsz = inputs.size(0), inputs.size(1)
        len_r = pos.size(0)

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # input_lin_results: [len_q, bsz, heads(16), 3, head_dim(64)]
        # input_lin_results: [len_q, batches=bsz*heads, 3, head_dim]
        qkv = qkv.view(inputs.size(0), inputs.size(1) * heads_t[0], 3, head_dim)
        queries = qkv[:, :, 0, :]
        keys = qkv[:, :, 1, :]
        values = qkv[:, :, 2, :]

        # The tensor is declared before hand to properly slice out query, key, and value grads.
        qkv_grads = torch.empty_like(qkv)
        queries_grads = qkv_grads[:, :, 0, :]
        keys_grads = qkv_grads[:, :, 1, :]
        values_grads = qkv_grads[:, :, 2, :]

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
        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.size(1) * heads_t[0], head_dim).transpose(0, 1)

        output_biases_grads = torch.sum(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)

        # Matmul2 - DGRAD1
        # Input1: (data grads)  [len_q, bsz*heads, head_dim] transpose(0,1)
        # Input2: (activations) [seql_k, bsz*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [bsz*heads, len_q, seql_k]
        # GEMM: Per batch: ( len_q x head_dim ) x ( head_dim x seql_k ) = ( len_q x seql_k )
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))
        # Matmul2 - DGRAD2
        # Input1: (data grads)  [len_q, bsz*heads, head_dim] transpose(0,1)
        # Input2: (activations) [seql_k, bsz*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [bsz*heads, len_q, seql_k]
        # GEMM: Per batch: ( len_q x head_dim ) x ( head_dim x seql_k ) = ( len_q x seql_k )
        torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))

        # print("Reached here")

        # Mask and Scaling for Dropout (not a publically documented op)
        if dropout_prob_t[0] > 0.0:
            dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))
        else:
            dropout_grads = matmul2_dgrad1

        # Softmax Grad (not a publically documented op)
        softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)
        attn_score_grads = softmax_grads
        # the grads are evenly distributed to AC and BD
        matmul_ac_grads = attn_score_grads

        # Matmul1 - DGRAD1
        # Input1: (data grads)  [bsz*heads, len_q, seql_k]
        # Input2: (activations) [seql_k, bsz*heads, head_dim] transpose(0,1)
        # Output:               [bsz*heads, len_q, head_dim] transpose(0,1)
        # GEMM: Per batch: ( len_q x seql_k ) x ( seql_k x head_dim ) = ( len_q x head_dim )
        torch.baddbmm(queries_grads.transpose(0, 1), matmul_ac_grads, keys.transpose(0, 1),
                      out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])

        queries_grads_ac = queries_grads
        r_w_bias_grads = torch.sum(queries_grads_ac.view(len_q, bsz, heads_t[0], -1), dim=[0, 1])  # heads * head_dim

        matmul_bd_grads = attn_score_grads

        if len_r > len_q:  # if we cut off the BDs from before, then put the zero gradients behind
            grad_cut = matmul_bd_grads.new_zeros((matmul_bd_grads.size(0), matmul_bd_grads.size(1), len_r - len_q))
            matmul_bd_grads = torch.cat([matmul_bd_grads, grad_cut], dim=-1)

        # backprop through the shifting
        matmul_bd_grads = RelativeShift.backward(matmul_bd_grads, True, False)

        # Matmul1 - DGRAD1
        # Input1: (matmul_bd_grads)  [bsz*heads, len_q, seql_k]
        # Input2: (r_head_k) [len_q, bsz*heads, head_dim] transpose(0,1)
        # Output:               [bsz*heads, len_q, head_dim] transpose(0,1)
        # GEMM: Per batch: ( len_q x seql_k ) x ( seql_k x head_dim ) = ( len_q x head_dim )
        queries_grads_bd = queries_grads.new_empty(*queries_grads.size())
        torch.baddbmm(queries_grads_bd.transpose(0, 1), matmul_bd_grads, r_head_k.transpose(0, 1),
                      out=queries_grads_bd.transpose(0, 1), beta=0.0, alpha=scale_t[0])

        # len_q x batch*heads x d_head
        r_r_bias_grads = torch.sum(queries_grads_bd.view(len_q, bsz, heads_t[0], -1), dim=[0, 1])

        # add the gradients from bd to queries
        queries_grads.add_(queries_grads_bd)

        # # MatmulAC - DGAD2
        # Input1: (data grads)  [bsz*heads, len_q, seql_k] transpose(1,2)
        # Input2: (rw_head_q) [bsz*heads, head_dim, len_q] transpose(0,1)
        # Output:               [seql_k, bsz*heads, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_k x len_q ) x ( len_q x head_dim ) = ( seql_k x head_dim )
        torch.baddbmm(keys_grads.transpose(0, 1), matmul_ac_grads.transpose(1, 2),
                      rw_head_q.transpose(0, 1), out=keys_grads.transpose(0, 1),
                      beta=0.0, alpha=scale_t[0])

        # MatmulBD - DGRAD2
        # Input1: (data grads)  [bsz*heads, len_q, len_r] transpose(1,2)
        # Input2: (rr_head_q) [len_q, bsz*heads, head_dim] transpose(0,1)
        # Output:  r_head_k  [len_r, bsz*heads, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_k x len_q ) x ( len_q x head_dim ) = ( seql_k x head_dim )
        r_head_k_grad = r_head_k.new_empty((len_r, bsz * heads_t[0], head_dim))
        # rr_head_q = queries.view(len_q, bsz, heads_t[0], head_dim) + r_r_bias  #
        # rr_head_q = rr_head_q.view(len_q, bsz * heads_t[0], head_dim)
        torch.baddbmm(r_head_k_grad.transpose(0, 1), matmul_bd_grads.transpose(1, 2).contiguous(),
                      rr_head_q.transpose(0, 1), out=r_head_k_grad.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        # r_head_k_grad = torch.matmul(matmul_bd_grads.transpose(1, 2), rr_head_q.transpose(0, 1))

        r_head_k_grad = r_head_k_grad.view(len_r, bsz, heads_t[0] * head_dim)
        # Input Linear GEMM - DGRAD
        # input1: (data grads) [len_q, bsz, 3*embed_dim(3072)]
        # input2: (weights)    [embed_dim*3 (3072), embed_dim (1024)]
        # output:              [len_q, bsz, embed_dim]
        # GEMM: ( (len_q*bsz) x 3*embed_dim ) x ( 3*embed_dim x embed_dim ) = (len_q*bsz x embed_dim)
        qkv_grads = qkv_grads.view(inputs.size(0), inputs.size(1), heads_t[0] * 3 * head_dim)

        input_grads, input_weights_grads, input_biases_grads, r_i_grads, s_i_grads = \
            mm.backward(qkv_grads, inputs, qkv_r, qkv_mm, input_weights, r_i, s_i)

        _, pos_weights_grads, pos_biases_grads, r_p_grads, s_p_grads = \
            mm.backward(r_head_k_grad, pos, rpos_r, rpos_mm, pos_weights, r_p, s_p, need_grad_x=False)

        return input_grads, None, None, None, None, None, \
               input_weights_grads, output_weights_grads, pos_weights_grads, \
               input_biases_grads, output_biases_grads, pos_biases_grads, \
               r_i_grads, s_i_grads, r_p_grads, s_p_grads, \
               r_w_bias_grads, r_r_bias_grads, \
               None, None, None, None, None


relative_self_attn_func = RelativeSelfAttnFunc.apply


class BERelativeSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., ensemble=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = True

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.pos_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        self.pos_proj_bias = Parameter(torch.Tensor(embed_dim))

        self.r_i = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        self.s_i = torch.nn.Parameter(torch.Tensor(ensemble, 3 * embed_dim))
        # self.r_o = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        # self.s_o = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        self.r_p = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))
        self.s_p = torch.nn.Parameter(torch.Tensor(ensemble, embed_dim))

        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

        self.reset_parameters()
        self.attn_func = RelativeSelfAttnFunc.apply

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

        with torch.no_grad():
            self.r_i.bernoulli_(0.5).mul_(-2).add_(1)
            self.s_i.bernoulli_(0.5).mul_(-2).add_(1)
                # self.r_o.bernoulli_(0.5).mul_(-2).add_(1)
                # self.s_o.bernoulli_(0.5).mul_(-2).add_(1)
            self.r_p.bernoulli_(0.5).mul_(-2).add_(1)
            self.s_p.bernoulli_(0.5).mul_(-2).add_(1)

    def forward(self, input, pos, key_padding_mask=None, attn_mask=None, indices=None, mems=None,
                incremental=False, incremental_cache=None, double_precision=False):

        bsz = input.size(1)
        ensemble = self.r_i.size(0)

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

        if self.training:
            if indices is None:
                with torch.no_grad():
                    indices = torch.arange(0, bsz, device=input.device, dtype=torch.long)
                    indices = torch.remainder(indices, ensemble)

            r_i = torch.index_select(self.r_i, 0, indices)
            s_i = torch.index_select(self.s_i, 0, indices)
            # r_o = torch.index_select(self.r_o, 0, indices)
            # s_o = torch.index_select(self.s_o, 0, indices)
            r_p = torch.index_select(self.r_p, 0, indices)
            s_p = torch.index_select(self.s_p, 0, indices)
        else:
            input = input.repeat(1, ensemble, 1)
            pos = pos.repeat(1, ensemble, 1)
            # if key_padding_mask is not None:
            #     mask = mask.repeat(ensemble, 1)
            r_i = self.r_i.repeat(bsz, 1).view(bsz, ensemble, self.r_i.size(-1)). \
                transpose(0, 1).contiguous().view(-1, self.r_i.size(-1))
            s_i = self.s_i.repeat(bsz, 1).view(bsz, ensemble, self.s_i.size(-1)). \
                transpose(0, 1).contiguous().view(-1, self.s_i.size(-1))
            r_p = self.r_p.repeat(bsz, 1).view(bsz, ensemble, self.r_p.size(-1)). \
                transpose(0, 1).contiguous().view(-1, self.r_p.size(-1))
            s_p = self.s_p.repeat(bsz, 1).view(bsz, ensemble, self.s_p.size(-1)). \
                transpose(0, 1).contiguous().view(-1, self.s_p.size(-1))
            # r_o = self.r_o.repeat(bsz, 1).view(bsz, ensemble, self.r_o.size(-1)). \
            #     transpose(0, 1).contiguous().view(-1, self.r_o.size(-1))
            # s_o = self.s_o.repeat(bsz, 1).view(bsz, ensemble, self.s_o.size(-1)). \
            #     transpose(0, 1).contiguous().view(-1, self.r_o.size(-1))

        is_training = self.training

        outputs, coverage = self.attn_func(input, pos, attn_mask is not None, is_training, self.num_heads, ensemble,
                                           self.in_proj_weight, self.out_proj_weight, self.pos_proj_weight,
                                           self.in_proj_bias, self.out_proj_bias, self.pos_proj_bias,
                                           r_i, s_i, r_p, s_p,
                                           self.r_w_bias, self.r_r_bias,
                                           mask, self.dropout,
                                           incremental, incremental_cache, double_precision)
        # last False is double precision

        return outputs, coverage


if __name__ == "__main__":
    bsz = 4
    seq_len_q = 4
    seq_len_kv = 7
    embed_dim = 32
    n_heads = 4
    output_size = 32
    ensemble = 7


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