import torch
import torch.nn.functional as F

try:
    import apex.amp as amp
    from apex.amp import half_function
except (ModuleNotFoundError, ImportError) as e:
    amp = None
    from .compat import half_function

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .compat import custom_fwd, custom_bwd

try:
    import mask_softmax_dropout_cuda
except (ModuleNotFoundError, ImportError) as e:
    mask_softmax_dropout_cuda = None


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
            # Refer to the variables in the forward to track the gradients
            bsz = grad_x.size(0)
            len_q, len_r = grad_x.size(1), grad_x.size(2)

            grad_x_view = grad_x.view(bsz, len_r, len_q)

            zero_pad = torch.zeros((bsz, 1, len_q), device=grad_x.device, dtype=grad_x.dtype)

            # grad_x should have size B x len_q x len_r
            # x_view should have size B x len_q+1 x len_r

            # put the zeros into the missing gradients
            grad_x_view = torch.cat([zero_pad, grad_x_view], dim=1)
            grad_x_padded = grad_x_view.view(bsz, len_q, len_r + 1)

            # because the first index in the padded dim was from zero_pad
            grad_output = grad_x_padded[:, :, 1:]
        else:
            raise NotImplementedError

        return grad_output


class RelativeSelfAttnFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, inputs, pos, use_time_mask, is_training, heads,
                input_weights, output_weights, pos_weights,
                input_biases, output_biases, pos_biases,
                r_w_bias, r_r_bias,
                mask, dropout_prob,
                incremental, incremental_cache,
                double_precision, learnable_pos, return_coverage):
        """
        :param return_coverage:
        :param learnable_pos:
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
        ctx.double_precision = double_precision

        bsz, len_q = inputs.size(1), inputs.size(0)
        len_r = pos.size(0)  # r can be longer than query, i.e for bidirectional attention we need 2k+1 positions
        len_k = len_q  # because of self-attention

        if pos.size(1) == 1 and not learnable_pos:
            pos = pos.repeat(1, bsz, 1)  # we have to use repeat instead of expand here because mm needs contiguous

        # Input Linear GEMM
        # input1: (activations) [len_q, bsz, hidden]
        # input2: (weights)     [hidden*3 (3072), hidden (1024)] (transpose [0,1])
        # output:               [len_q, bsz, hidden*3]
        # GEMM: ( (len_q*bsz) x embed_dim ) x ( embed_dim x embed_dim*3 ) = (len_q*bsz x embed_dim*3)
        input_lin_results = torch.addmm(input_biases,
                                        inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)),
                                        input_weights.transpose(0, 1),
                                        beta=1., alpha=1.)

        # reshape [len_q*bsz, embed_dim*3 -> len_q x bsz x embed_dim*3]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))

        if not learnable_pos:
            pos_lin_results = torch.addmm(pos_biases,
                                          pos.view(pos.size(0) * pos.size(1), pos.size(2)),
                                          pos_weights.transpose(0, 1),
                                          beta=1., alpha=1.)

            pos_lin_results = pos_lin_results.view(pos.size(0), pos.size(1), pos_weights.size(0))

            r_head_k = pos_lin_results.view(pos.size(0), bsz * heads, head_dim)  # T x BxH x D
        else:
            # pos_lin_results = pos.view(pos.size(0), bsz * heads, head_dim)  # T x BxH x D
            # r_head_k = pos_lin_results
            pos_lin_results = None
            r_head_k = None

        # Slice out q,k,v from one big Input Linear output (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # input_lin_results: [len_q, bsz, heads(16), 3, head_dim(64)]
        # input_lin_results: [len_q, batches=bsz*heads, 3, head_dim]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
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
        # Relative Attention from here:
        # r_w_bias size: head * head_dim
        rw_head_q = queries.view(len_q, bsz, heads, head_dim) + r_w_bias  #
        rw_head_q = rw_head_q.view(len_q, bsz * heads, head_dim)

        # matmul_ac batched GEMMs
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

        if not learnable_pos:
            if queries.is_cuda:
                # matmul2 batched GEMMs
                # queries+bias: [len_q, bsz*heads, head_dim] transpose(0, 1)
                # rel_positions: [len_r, bsz*heads, head_dim] transpose(0, 1)
                matmul_bd = torch.empty((bsz * heads, queries.size(0), len_r), dtype=queries.dtype,
                                        device=rw_head_q.device)
                matmul_bd = torch.baddbmm(matmul_bd, rr_head_q.transpose(0, 1),
                                          r_head_k.transpose(0, 1).transpose(1, 2),
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
        else:
            # if queries.is_cuda:
            #     # matmul2 batched GEMMs
            #     # queries+bias: [len_q, bsz*heads, head_dim]
            #     # rel_positions: [len_q, len_k, head_dim] transpose(1, 2)
            #     matmul_bd = torch.empty((queries.size(0), bsz * heads, keys.size(0)), dtype=queries.dtype,
            #                             device=queries.device)
            #     torch.baddbmm(matmul_bd, rr_head_q, pos.transpose(1, 2), out=matmul_bd, beta=0.0, alpha=scale_t[0])
            #     matmul_bd = matmul_bd.transpose(0, 1)
            #
            # else:
            matmul_bd = torch.matmul(rr_head_q, pos.transpose(1, 2)).transpose(0, 1).mul_(scale_t[0])
            # no need to shift in this case

        attn_score = matmul_ac + matmul_bd  # both AC and BD are scaled with scale_t before in baddbmm
        # attn_score should have size [bsz*heads, len_q, len_k] for now

        if mask is not None:
            # Self Attention Time Mask
            if use_time_mask:
                assert (len(mask.size()) == 2), "Timing mask is not 2D!"
                # assert (mask.size(0) == mask.size(1)), "Sequence length should match!"
                mask = mask.to(torch.bool)
            # Key Padding Mask
            else:
                batches, len_q, seql_k = attn_score.size()
                bsz = int(batches / heads)
                attn_score = attn_score.view(bsz, heads, len_q, seql_k)
                mask = mask.to(torch.bool)
                mask = mask.unsqueeze(1).unsqueeze(2)

            attn_score.masked_fill_(mask, float('-inf'))
            attn_score = attn_score.view(bsz * heads, len_q, len_k)

        if not (mask_softmax_dropout_cuda is not None and len_k <= 2048 and attn_score.is_cuda) or double_precision:

            dtype_ = torch.float64 if double_precision else torch.float32
            softmax_results = F.softmax(attn_score, dim=-1).type_as(attn_score)

            # Dropout - is not executed for inference
            if is_training:
                dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=(1. - dropout_prob_t[0]))
            else:
                dropout_results = softmax_results
                dropout_mask = null_tensor
            ctx.fused_softmax_dropout = False
        else:
            # Fused Softmax and Dropout
            # ASSERTED To produce the same result with F.softmax
            dropout_mask, softmax_results = \
                mask_softmax_dropout_cuda.forward(is_training, heads, attn_score, dropout_prob_t[0])

            # In this fused (inplace) dropout-softmax, "softmax_results" is mask-dropped already
            # We need to store the attn_score to recompute the softmax later in the backward pass
            dropout_results = softmax_results
            softmax_results = attn_score
            ctx.fused_softmax_dropout = True

        nan_mask = null_tensor
        # nan_mask = torch.isnan(softmax_results)
        # if nan_mask.any():
        #     softmax_results.masked_fill_(nan_mask, 0)

        # Matmul2 Batched GEMMs
        # The output tensor specification is needed here to specify the non-standard output.
        # Given that pytorch cannot currently perform autograd with an output tensor specified,
        # this requires a backward pass specified.
        # Input1: from_softmax [bsz*heads, len_q, seql_k]
        # Input2: (values)     [seql_v, bsz*heads, head_dim] transpose(0,1)
        # Output:              [bsz*heads, len_q, head_dim]
        # GEMM: Per batch: ( len_q x seql_k ) x ( seql_k x head_dim ) = (len_q x head_dim)

        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1)).transpose(0, 1)
        if learnable_pos:
            # Input1: from_softmax [bsz*heads, len_q, seql_k].transpose(0, 1)
            # Input2: R [len_q, len_k, head_dim]
            # Output: [ len_q, bsz*heads, head_dim]
            torch.baddbmm(matmul2_results, dropout_results.transpose(0, 1), pos, beta=1.0, alpha=1.0,
                          out=matmul2_results)

        matmul2_results = matmul2_results.contiguous().view(inputs.size(0), inputs.size(1), inputs.size(2))

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
                              matmul2_results,
                              dropout_results,
                              softmax_results,
                              input_lin_results,
                              pos_lin_results,
                              rw_head_q, rr_head_q,
                              inputs, pos, r_head_k,
                              input_weights, pos_weights,
                              output_weights,
                              r_w_bias, r_r_bias,
                              dropout_mask, nan_mask,
                              dropout_prob_t)

        ctx.learnable_pos = learnable_pos
        ctx.return_coverage = return_coverage

        if return_coverage:
            return (outputs, dropout_results)
        else:
            return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, *output_grads):
        """
        :param ctx:
        :param output_grads: gradients w.r.t the outputs
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
        dropout_mask, nan_mask, \
        dropout_prob_t = ctx.saved_tensors

        learnable_pos = ctx.learnable_pos
        if ctx.return_coverage:
            output_grads, softmax_grads = output_grads
        else:
            output_grads = output_grads[0]

        head_dim = inputs.size(2) // heads_t[0]
        len_q, bsz = inputs.size(0), inputs.size(1)
        len_k = len_q

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
        # Input1: (data grads)  [bsz*heads, len_q,  head_dim]
        # Input2: (activations) [seql_k, bsz*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [bsz*heads, len_q, seql_k]
        # GEMM: Per batch: ( len_q x head_dim ) x ( head_dim x seql_k ) = ( len_q x seql_k )
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))
        # Matmul2 - DGRAD2
        # Input2: (data grads)  [bsz*heads, len_q,  head_dim]
        # Input1: (activations) [bsz*heads, len_q, len_k] transpose(1,2)
        # Output:               [bsz*heads, len_k, head_dim]
        # GEMM: Per batch: ( len_k x len_q ) x ( len_q x head_dim ) = ( len_k x head_dim )
        torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))

        if learnable_pos:
            # Add the gradients from the positions to the attention matrix
            # Input1: (data grads)  [bsz*heads, len_q,  head_dim].transpose(0, 1)
            # Input2: (rpositions) [len_q, len_k, head_dim].transpose(1,2)
            # Output:               [bsz*heads, len_q, seql_k].transpose(0, 1)
            torch.baddbmm(matmul2_dgrad1.transpose(0, 1), output_lin_grads.transpose(0, 1), pos.transpose(1, 2),
                          beta=1.0, alpha=1.0, out=matmul2_dgrad1.transpose(0, 1))
            # Input2: (data grads)  [bsz*heads, len_q,  head_dim].transpose(0, 1)
            # Input1: (activations) [bsz*heads, len_q, len_k] transpose(0,1).transpose(1,2)
            # Output:               [len_q, len_k, head_dim]
            pos_grads = torch.bmm(dropout_results.transpose(0, 1).transpose(1, 2), output_lin_grads.transpose(0, 1))

        if ctx.fused_softmax_dropout:
            assert (mask_softmax_dropout_cuda is not None)
            # softmax_results here is actually softmax input to recompute softmax (attn_score above)
            softmax_grads = mask_softmax_dropout_cuda.backward_recompute(heads_t, matmul2_dgrad1,
                                                                         softmax_results, dropout_mask,
                                                                         dropout_prob_t[0])

            # ASSERTED to have the same grads with recomputing softmax and then calling torch.softmax_backward_data

        else:
            if mask_softmax_dropout_cuda is not None and len_k <= 1024 \
                    and matmul2_dgrad1.is_cuda and not ctx.double_precision:
                softmax_grads = mask_softmax_dropout_cuda.backward(heads_t[0], matmul2_dgrad1, softmax_results,
                                                                   dropout_mask, dropout_prob_t[0])

            else:
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

        if not learnable_pos:

            if len_r > len_q:  # if we cut off the BDs from before, then put the zero gradients behind
                grad_cut = matmul_bd_grads.new_zeros((matmul_bd_grads.size(0), matmul_bd_grads.size(1), len_r - len_q))
                matmul_bd_grads = torch.cat([matmul_bd_grads, grad_cut], dim=-1)

            # backprop through the shifting
            matmul_bd_grads = RelativeShift.backward(matmul_bd_grads, True, False)

            # MatmulBD - DGRAD1
            # Input1: (matmul_bd_grads)  [bsz*heads, len_q, seql_k]
            # Input2: (r_head_k) [len_q, bsz*heads, head_dim] transpose(0,1)
            # Output:               [bsz*heads, len_q, head_dim] transpose(0,1)
            # GEMM: Per batch: ( len_q x seql_k ) x ( seql_k x head_dim ) = ( len_q x head_dim )
            queries_grads_bd = queries_grads.new_empty(*queries_grads.size())
            torch.baddbmm(queries_grads_bd.transpose(0, 1), matmul_bd_grads, r_head_k.transpose(0, 1),
                          out=queries_grads_bd.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        else:
            # MatmulBD - DGRAD1
            # Input1: (matmul_bd_grads)  [bsz*heads, len_q, len_k] transpose(0,1)
            # Input2: (pos) [len_q, len_k, head_dim]
            # Output:               [len_q, bsz*heads, head_dim]
            # GEMM: Per batch: ( bsz*heads x len_k ) x ( len_k x head_dim ) = ( bsz*heads x head_dim )
            # queries_grads_bd = queries_grads.new_empty(*queries_grads.size())
            # torch.baddbmm(queries_grads_bd, matmul_bd_grads.transpose(0, 1), pos,
            #               out=queries_grads_bd, beta=0.0, alpha=scale_t[0])
            queries_grads_bd = torch.bmm(matmul_bd_grads.transpose(0, 1), pos).mul_(scale_t[0])

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

        if not learnable_pos:
            # MatmulBD - DGRAD2
            # Input1: (data grads)  [bsz*heads, len_q, len_r] transpose(1,2)
            # Input2: (rr_head_q) [len_q, bsz*heads, head_dim] transpose(0,1)
            # Output:  r_head_k  [len_r, bsz*heads, head_dim] transpose(0,1)
            # GEMM: Per batch: ( seql_k x len_q ) x ( len_q x head_dim ) = ( seql_k x head_dim )
            r_head_k_grad = r_head_k.new_empty((len_r, bsz * heads_t[0], head_dim))
            torch.baddbmm(r_head_k_grad.transpose(0, 1), matmul_bd_grads.transpose(1, 2).contiguous(),
                          rr_head_q.transpose(0, 1), out=r_head_k_grad.transpose(0, 1), beta=0.0, alpha=scale_t[0])

            r_head_k_grad = r_head_k_grad.view(len_r, bsz, heads_t[0], head_dim). \
                view(len_r * bsz, heads_t[0] * head_dim)

            pos_weight_grads = torch.mm(r_head_k_grad.transpose(0, 1),
                                        pos.view(pos.size(0) * pos.size(1), pos.size(2)))

            pos_bias_grads = torch.sum(r_head_k_grad, 0)
            pos_grads = None
        else:
            pos_weight_grads, pos_bias_grads = None, None
            # MatmulBD - DGRAD2
            # Input1: (data grads)  [bsz*heads, len_q, len_k] transpose(0,1),(1,2) -> [len_q, len_k, bsz*heads]
            # Input2: (rr_head_q) [len_q, bsz*heads, head_dim]
            # Output:  pos_grads  [len_q, len_k, head_dim]
            # GEMM: Per batch: ( len_k x bsz ) x ( bsz x head_dim ) = ( len_k x head_dim )
            torch.baddbmm(pos_grads, matmul_bd_grads.transpose(0, 1).transpose(1, 2).contiguous(),
                          rr_head_q, out=pos_grads, beta=1.0, alpha=scale_t[0])

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

        return input_grads, pos_grads, None, None, None, input_weight_grads, output_weight_grads, pos_weight_grads, \
               input_bias_grads, output_bias_grads, pos_bias_grads, r_w_bias_grads, r_r_bias_grads, \
               None, None, None, None, None, None, None


@half_function
def relative_self_attn_func(input, pos, use_mask, is_training, num_heads,
                            in_proj_weight, out_proj_weight, pos_proj_weight,
                            in_proj_bias, out_proj_bias, pos_proj_bias,
                            r_w_bias, r_r_bias,
                            mask, dropout,
                            incremental, incremental_cache,
                            double_precision, learnable_pos, return_coverage):
    return RelativeSelfAttnFunc.apply(input, pos, use_mask, is_training, num_heads,
                                      in_proj_weight, out_proj_weight, pos_proj_weight,
                                      in_proj_bias, out_proj_bias, pos_proj_bias,
                                      r_w_bias, r_r_bias,
                                      mask, dropout,
                                      incremental, incremental_cache,
                                      double_precision, learnable_pos, return_coverage)


# TODO: write test function
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=32,
                        help='Size of embedding / transformer hidden')
    parser.add_argument('-gpu', default=0, type=int,
                        help="Seed for deterministic runs.")

    test_function = relative_self_attn_func

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

    opt.layers = 2
    opt.variational_dropout = False
    opt.dropout = 0.0
    opt.attn_dropout = 0.0
    opt.n_heads = 4
    opt.inner_size = 16
    opt.head_dim = opt.model_size // opt.n_heads


    class Parameters(torch.nn.Module):

        def __init__(self, model_size=16, heads=1):
            self.model_size = model_size
            self.heads = heads
            self.head_dim = model_size // heads
            # self.function = RelativeShiftFunction.apply

            self.in_proj_weight = torch.Tensor(3 * model_size, model_size)
            self.out_proj_weight = torch.Tensor(model_size, model_size)
            self.pos_proj_weight = torch.Tensor(model_size, model_size)

            self.in_proj_bias = torch.Tensor(3 * model_size)
            self.out_proj_bias = torch.Tensor(model_size)
            self.pos_proj_bias = torch.Tensor(model_size)

            self.r_w_bias = torch.Tensor(self.heads, self.head_dim)
            self.r_r_bias = torch.Tensor(self.heads, self.head_dim)
            self.reset_parameters()

        def reset_parameters(self):
            std_ = 0.02
            torch.nn.init.normal_(self.in_proj_weight, 0.0, std_)
            torch.nn.init.normal_(self.out_proj_weight, 0.0, std_)
            torch.nn.init.normal_(self.pos_proj_weight, 0.0, std_)

            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj_bias, 0.)
            torch.nn.init.constant_(self.pos_proj_bias, 0.)

            torch.nn.init.normal_(self.r_w_bias, 0.0, std_)
            torch.nn.init.normal_(self.r_r_bias, 0.0, std_)

    class TestAttention(torch.nn.Module):

        def __init__(self, test_function, model_size=16, heads=1):
            super().__init__()
            self.model_size = model_size
            self.heads = heads
            self.head_dim = model_size // heads

            self.function = test_function

        def forward(self, input, pos, in_proj_weight, out_proj_weight, pos_proj_weight,
                    in_proj_bias, out_proj_bias, pos_proj_bias, r_w_bias, r_r_bias,
                    mask, learnable_embedding=False):
            use_time_mask = False
            is_training = True
            dropout = 0.0
            double_precision = True
            return_coverage = False

            return self.function(input, pos, use_time_mask, is_training, self.heads,
                                 in_proj_weight, out_proj_weight, pos_proj_weight,
                                 in_proj_bias, out_proj_bias, pos_proj_bias,
                                 r_w_bias, r_r_bias,
                                 mask, dropout,
                                 False, None,  # For the incremental stuff
                                 double_precision, learnable_embedding, return_coverage)   # double precision set to true


    bsz = 4
    len_q = 5
    len_r = 15

    input_states = torch.randn(*(len_q, bsz, opt.model_size)).double().cuda()
    input_states.requires_grad = True
    pos = torch.randn(*(len_r, 1, opt.model_size)).double().cuda()
    net = TestAttention(test_function, model_size=opt.model_size, heads=opt.n_heads)

    parameters = Parameters(opt.model_size, opt.n_heads)

    in_proj_weight = parameters.in_proj_weight.double().cuda()
    out_proj_weight = parameters.out_proj_weight.double().cuda()
    pos_proj_weight = parameters.pos_proj_weight.double().cuda()

    in_proj_bias = parameters.in_proj_bias.double().cuda()
    out_proj_bias = parameters.out_proj_bias.double().cuda()
    pos_proj_bias = parameters.pos_proj_bias.double().cuda()

    r_w_bias = parameters.r_w_bias.double().cuda()
    r_r_bias = parameters.r_r_bias.double().cuda()

    in_proj_weight.requires_grad = True
    out_proj_weight.requires_grad = True
    pos_proj_weight.requires_grad = True
    in_proj_bias.requires_grad = True
    out_proj_bias.requires_grad = True
    pos_proj_bias.requires_grad = True

    r_w_bias.requires_grad = True
    r_r_bias.requires_grad = True

    mask = None  # input_states.new(*(bsz, len_q)).fill_(0).bool()
    # mask.requires_grad = False
    learnable_pe = False

    print("gradchecking start.")

    torch.autograd.gradcheck(net, (input_states, pos, in_proj_weight, out_proj_weight, pos_proj_weight,
                                   in_proj_bias, out_proj_bias, pos_proj_bias, r_w_bias, r_r_bias,
                                   mask, learnable_pe))

    print("gradchecking completed.")

    pos = torch.randn(*(len_q, len_q, opt.head_dim)).double().cuda()
    pos.requires_grad = True
    learnable_pe = True

    print("gradchecking w/ learnable position encodings start.")
    net.forward(input_states, pos, in_proj_weight, out_proj_weight, pos_proj_weight,
                                   in_proj_bias, out_proj_bias, pos_proj_bias, r_w_bias, r_r_bias,
                                   mask, learnable_pe)
    torch.autograd.gradcheck(net, (input_states, pos, in_proj_weight, out_proj_weight, pos_proj_weight,
                                   in_proj_bias, out_proj_bias, pos_proj_bias, r_w_bias, r_r_bias,
                                   mask, learnable_pe),
                             eps=1e-6, atol=1e-5, rtol=1e-3)

    print("gradchecking w/ learnable position encodings completed.")
