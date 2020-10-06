import torch
import torch.nn.functional as F
import apex.amp as amp


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
        assert (batch_first or emb_last) and not (batch_first and emb_last), \
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
    def forward(ctx, inputs, pos, use_time_mask, is_training, heads,
                input_weights, output_weights, pos_weights,
                input_biases, output_biases, pos_biases,
                r_w_bias, r_r_bias,
                mask, dropout_prob,
                incremental, incremental_cache,
                double_precision, add_position):
        """
        :param add_position: bool
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

        if pos.size(1) == 1:
            pos = pos.repeat(1, bsz, 1)  # to T x B x H

        # Input Linear GEMM
        # input1: (activations) [len_q, bsz, hidden]
        # input2: (weights)     [hidden*3 (3072), hidden (1024)] (transpose [0,1])
        # output:               [len_q, bsz, hidden*3]
        # GEMM: ( (len_q*bsz) x embed_dim ) x ( embed_dim x embed_dim*3 ) = (len_q*bsz x embed_dim*3)
        input_lin_results = torch.addmm(input_biases,
                                        inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)),
                                        input_weights.transpose(0, 1),
                                        beta=1., alpha=1.)

        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))

        pos_lin_results = torch.addmm(pos_biases,
                                      pos.view(pos.size(0) * pos.size(1), pos.size(2)),
                                      pos_weights.transpose(0, 1),
                                      beta=1., alpha=1.)

        pos_lin_results = pos_lin_results.view(pos.size(0), pos.size(1), pos_weights.size(0))

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # input_lin_results: [len_q, bsz, heads(16), 3, head_dim(64)]
        # input_lin_results: [len_q, batches=bsz*heads, 3, head_dim]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]

        r_head_k = pos_lin_results.view(pos.size(0), bsz * heads, head_dim)  # T x BxH x D

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
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs.size(0), inputs.size(1),
                                                                            inputs.size(2))

        # if add_position:
        #     # matmul3 batched GEMMs
        #     # input1: [bsz*heads, len_q, len_q] > [len_q, BH, len_q]
        #     # input2: [len_q len_q head_dim]  > [len_q, len_q, head_dim]
        #     # output: [len_q BH len_q head_dim]
        #     # input2: [bsz*heads, len_k, head_dim]
        #     matmul3_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)),
        #                                   dtype=dropout_results.dtype, device=queries.device).transpose(1, 0)

        """
        0 -1 -2 -3 -4 -5
        1 0 -1 -2 -3 -4
        2 1 0 -1 -2 -3
        3 2 1 0 -1 -2
        4 3 2 1 0 -1
        5 4 3 2 1 0
        """
        # else:
        #     matmul3_results = None

        # Output Linear GEMM
        # Input1: (activations) [len_q, bsz, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ] transpose(0,1)
        # Output:               [ len_q, bsz, embed_dim ]
        # GEMM: ( len_q*bsz x embed_dim ) x ( embed_dim x embed_dim ) = ( len_q*bsz x embed_dim )
        outputs = torch.addmm(output_biases,
                              matmul2_results.view(inputs.size(0) * inputs.size(1), inputs.size(2)),
                              output_weights.transpose(0, 1),
                              beta=1., alpha=1.)

        outputs = outputs.view(inputs.size(0), inputs.size(1), output_weights.size(0))

        ctx.save_for_backward(heads_t,
                              scale_t,
                              matmul2_results,  #
                              dropout_results,
                              softmax_results,
                              input_lin_results,
                              pos_lin_results,
                              rw_head_q, rr_head_q,
                              inputs, pos, r_head_k,
                              input_weights, pos_weights,
                              output_weights,
                              r_w_bias, r_r_bias,
                              dropout_mask,
                              dropout_prob_t)

        ctx.add_position = add_position

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
        input_lin_results, pos_lin_results, \
        rw_head_q, rr_head_q, \
        inputs, pos, r_head_k, \
        input_weights, pos_weights, \
        output_weights, \
        r_w_bias, r_r_bias, \
        dropout_mask, \
        dropout_prob_t = ctx.saved_tensors

        head_dim = inputs.size(2) // heads_t[0]
        len_q, bsz = inputs.size(0), inputs.size(1)
        len_r = pos.size(0)

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # input_lin_results: [len_q, bsz, heads(16), 3, head_dim(64)]
        # input_lin_results: [len_q, batches=bsz*heads, 3, head_dim]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads_t[0], 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]

        # The tensor is declared before hand to properly slice out query, key, and value grads.
        input_lin_results_grads = torch.empty_like(input_lin_results)
        queries_grads = input_lin_results_grads[:, :, 0, :]
        keys_grads = input_lin_results_grads[:, :, 1, :]
        values_grads = input_lin_results_grads[:, :, 2, :]

        # Output Linear GEMM - DGRAD
        # Input1: (data grads)  [len_q, bsz, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ]
        # Output:               [ len_q, bsz, embed_dim ]
        # GEMM: ( len_q*bsz x embed_dim ) x ( embed_dim x embed_dim ) = ( len_q*bsz x embed_dim )
        output_lin_grads = torch.mm(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        # Output Linear GEMM - WGRAD
        # Input1: (data grads)  [len_q*bsz, embed_dim=heads*head_dim] transpose(0,1)
        # Input2: (activations) [len_q*bsz, embed_dim ]
        # Output:               [ len_q, bsz, embed_dim ]
        # GEMM: ( embed_dim x len_q*bsz ) x ( len_q*bsz x embed_dim ) = ( embed_dim x embed_dim )
        output_weight_grads = torch.mm(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1),
            matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.size(1) * heads_t[0], head_dim).transpose(0, 1)

        output_bias_grads = torch.sum(
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

        r_head_k_grad = r_head_k_grad.view(len_r, bsz, heads_t[0], head_dim).view(len_r * bsz, heads_t[0] * head_dim)
        # Input Linear GEMM - DGRAD
        # input1: (data grads) [len_q, bsz, 3*embed_dim(3072)]
        # input2: (weights)    [embed_dim*3 (3072), embed_dim (1024)]
        # output:              [len_q, bsz, embed_dim]
        # GEMM: ( (len_q*bsz) x 3*embed_dim ) x ( 3*embed_dim x embed_dim ) = (len_q*bsz x embed_dim)
        input_lin_results_grads = input_lin_results_grads.view(inputs.size(0) * inputs.size(1),
                                                               heads_t[0] * 3 * head_dim)
        input_grads = torch.mm(input_lin_results_grads, input_weights)
        input_grads = input_grads.view(inputs.size(0), inputs.size(1), inputs.size(2))
        # Input Linear GEMM - WGRAD
        # input1: (data grads)  [len_q*bsz, 3*embed_dim(3072)]
        # input2: (activations) [len_q*bsz, embed_dim(1024)]
        # output:               [3*embed_dim, embed_dim]
        # GEMM: ( 3*embed_dim x len_q*bsz ) x ( len_q*bsz x embed_dim ) = (3*embed_dim x embed_dim)
        input_weight_grads = torch.mm(input_lin_results_grads.transpose(0, 1),
                                      inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)))

        input_bias_grads = torch.sum(input_lin_results_grads, 0)

        # Input Linear GEMM - WGRAD
        # input1: r_head_k_grad  [len_q*bsz, embed_dim]
        # input2: (pos)          [len_q*bsz, embed_dim]
        # output:               [3*embed_dim, embed_dim]
        # GEMM: ( 3*embed_dim x len_q*bsz ) x ( len_q*bsz x embed_dim ) = (3*embed_dim x embed_dim)
        pos_weight_grads = torch.mm(r_head_k_grad.transpose(0, 1),
                                    pos.view(pos.size(0) * pos.size(1), pos.size(2)))

        pos_bias_grads = torch.sum(r_head_k_grad, 0)

        return input_grads, None, None, None, None, input_weight_grads, output_weight_grads, pos_weight_grads, \
               input_bias_grads, output_bias_grads, pos_bias_grads, r_w_bias_grads, r_r_bias_grads, \
               None, None, None, None, None, None


@amp.half_function
def relative_self_attn_func(input, pos, use_mask, is_training, num_heads,
                            in_proj_weight, out_proj_weight, pos_proj_weight,
                            in_proj_bias, out_proj_bias, pos_proj_bias,
                            r_w_bias, r_r_bias,
                            mask, dropout,
                            incremental, incremental_cache, something, another):

    output, coverage = RelativeSelfAttnFunc.apply(input, pos, use_mask, is_training, num_heads,
                                                  in_proj_weight, out_proj_weight, pos_proj_weight,
                                                  in_proj_bias, out_proj_bias, pos_proj_bias,
                                                  r_w_bias, r_r_bias,
                                                  mask, dropout,
                                                  incremental, incremental_cache, something, another)

    return output, coverage
