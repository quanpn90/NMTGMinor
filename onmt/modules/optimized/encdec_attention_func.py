import torch
import torch.nn.functional as F
from onmt.constants import double_precision
import apex.amp as amp


class EncdecAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv,
                input_weights_q, input_weights_kv, output_weights,
                mask, dropout_prob,
                incremental, incremental_cache):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs_q.size(2) // heads
        scale_t = torch.tensor([head_dim ** -0.5])

        bsz, len_q, len_k = inputs_q.size(1), inputs_q.size(0), inputs_kv.size(0)

        # TODO: add incremental cache

        # Input Linear GEMM Q
        # input1: (activations) [seql_q, bsz, embed_dim(1024)]
        # input2: (weights)     [embed_dim (1024), embed_dim (1024)] (transpose [0,1])
        # output:               [seql_q, bsz, embed_dim]
        # GEMM: ( (seql_q*seqs) x embed_dim ) x ( embed_dim x embed_dim ) = (seql_q*seqs x embed_dim)
        input_lin_q_results = torch.mm(inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                                       input_weights_q.transpose(0, 1))
        input_lin_q_results = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1), input_weights_q.size(0))
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads, head_dim)
        # Input Linear GEMM KV
        # input1: (activations) [seql_k, bsz, embed_dim(1024)]
        # input2: (weights)     [embed_dim*2 (2048), embed_dim (1024)] (transpose [0,1])
        # output:               [seql_k, bsz, embed_dim*2]
        # GEMM: ( (seql_k*seqs) x embed_dim ) x ( embed_dim x embed_dim*2 ) = (seql_k*seqs x embed_dim*2)

        # Slice out k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM

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

        ctx.save_for_backward(heads_t,
                              scale_t,
                              matmul2_results,
                              dropout_results,
                              softmax_results,
                              input_lin_q_results,
                              input_lin_kv_results,
                              inputs_q,
                              inputs_kv,
                              input_weights_q,
                              input_weights_kv,
                              output_weights,
                              dropout_mask,
                              dropout_prob_t)

        return outputs.detach(), softmax_results.detach()

    @staticmethod
    def backward(ctx, output_grads, softmax_grads):

        heads_t, scale_t, matmul2_results, dropout_results, softmax_results \
            , input_lin_q_results, input_lin_kv_results \
            , inputs_q, inputs_kv \
            , input_weights_q, input_weights_kv, output_weights \
            , dropout_mask, dropout_prob_t \
            = ctx.saved_tensors

        head_dim = inputs_q.size(2) // heads_t[0]

        # Slice out k,v from one big Input Linear output (should only impact meta data, no copies!)
        # Batch sizes and heads are combined to make the batch of the Batched GEMM
        # input_lin_kv_results: [seql_k, bsz, heads(16), 2, head_dim(64)]
        # input_lin_kv_results: [seql_k, batches=bsz*heads, 2, head_dim]
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads_t[0], head_dim)
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1) * heads_t[0], 2, head_dim)
        keys = input_lin_kv_results[:, :, 0, :]
        values = input_lin_kv_results[:, :, 1, :]

        # Slice out k,v from one big set of gradients entering the input linear's bprop
        # (should only impact meta data, no copies!)
        # The gradients are identical in size to the Input Linear outputs.
        # The tensor is declared before hand to properly slice out query, key, and value grads.
        input_lin_kv_results_grads = torch.empty_like(input_lin_kv_results)
        queries_grads = torch.empty_like(queries)
        keys_grads = input_lin_kv_results_grads[:, :, 0, :]
        values_grads = input_lin_kv_results_grads[:, :, 1, :]

        # Output Linear GEMM - DGRAD
        # Input1: (data grads)  [seql_q, bsz, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ]
        # Output:               [ seql_q, seqs, embed_dim ]
        # GEMM: ( seql_q*seqs x embed_dim ) x ( embed_dim x embed_dim ) = ( seql_q*seqs x embed_dim )
        output_lin_grads = torch.mm(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        # Output Linear GEMM - WGRAD
        # Input1: (data grads)  [seql_q*seqs, embed_dim=heads*head_dim] transpose(0,1)
        # Input2: (activations) [seql_q*seqs, embed_dim ]
        # Output:               [ seql_q, seqs, embed_dim ]
        # GEMM: ( embed_dim x seql_q*seqs ) x ( seql_q*seqs x embed_dim ) = ( embed_dim x embed_dim )
        output_weight_grads = torch.mm(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1),
            matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1) * heads_t[0],
                                                 head_dim).transpose(0, 1)

        # Matmul2 - DGRAD1
        # Input1: (data grads)  [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Input2: (activations) [seql_k, seqs*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [seqs*heads, seql_q, seql_k]
        # GEMM: Per batch: ( seql_q x head_dim ) x ( head_dim x seql_k ) = ( seql_q x seql_k )
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))
        # Matmul2 - DGRAD2
        # Input1: (data grads)  [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Input2: (activations) [seql_k, seqs*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [seqs*heads, seql_q, seql_k]
        # GEMM: Per batch: ( seql_q x head_dim ) x ( head_dim x seql_k ) = ( seql_q x seql_k )
        values_grads = torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))

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
        queries_grads = queries_grads.transpose(0, 1).view(inputs_q.size(0) * inputs_q.size(1), heads_t[0] * head_dim)
        input_q_grads = torch.mm(queries_grads, input_weights_q)
        input_q_grads = input_q_grads.view(inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))
        # Input KV Linear GEMM - DGRAD
        # input1: (data grads) [seql_k, seqs, 2*embed_dim(2048)]
        # input2: (weights)    [embed_dim*2 (2048), embed_dim (1024)]
        # output:              [seql_k, seqs, embed_dim]
        # GEMM: ( (seql_k*seqs) x 2*embed_dim ) x ( 2*embed_dim x embed_dim ) = (seql_k*seqs x embed_dim)
        # the elements of values and query grads are already stored in (shared) query_grads and values_grads
        input_lin_kv_results_grads = input_lin_kv_results_grads.view(inputs_kv.size(0) * inputs_kv.size(1),
                                                                     heads_t[0] * 2 * head_dim)
        input_kv_grads = torch.mm(input_lin_kv_results_grads, input_weights_kv)
        input_kv_grads = input_kv_grads.view(inputs_kv.size(0), inputs_kv.size(1), inputs_kv.size(2))
        # Input Q Linear GEMM - WGRAD
        # input1: (data grads)  [seql_q*seqs, embed_dim(1024)]
        # input2: (activations) [seql_q*seqs, embed_dim(1024)]
        # output:               [embed_dim, embed_dim]
        # GEMM: ( embed_dim x seql_q*seqs ) x ( seql_q*seqs x embed_dim ) = (embed_dim x embed_dim)
        input_weight_q_grads = torch.mm(queries_grads.transpose(0, 1),
                                        inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)))
        # Input KV Linear GEMM - WGRAD
        # input1: (data grads)  [seql_k*seqs, 2*embed_dim(2048)]
        # input2: (activations) [seql_k*seqs, embed_dim(1024)]
        # output:               [2*embed_dim, embed_dim]
        # GEMM: ( 2*embed_dim x seql_k*seqs ) x ( seql_k*seqs x embed_dim ) = (2*embed_dim x embed_dim)
        input_weight_kv_grads = torch.mm(input_lin_kv_results_grads.transpose(0, 1),
                                         inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)))

        return None, None, None \
            , input_q_grads, input_kv_grads \
            , input_weight_q_grads, input_weight_kv_grads, output_weight_grads \
            , None, None, None, None


@amp.half_function
def encdec_attn_func(time_masking, is_training,
                     num_heads, query, key,
                     in_proj_weight_q, in_proj_weight_kv,
                     out_proj_weight, attn_mask, dropout,
                     incremental, incremental_cache):
    output, coverage = EncdecAttnFunc.apply(time_masking, is_training,
                                            num_heads, query, key,
                                            in_proj_weight_q, in_proj_weight_kv,
                                            out_proj_weight, attn_mask, dropout,
                                            incremental, incremental_cache)

    return output, coverage


@amp.half_function
def fast_self_attn_func(time_masking, is_training, num_heads, query, key,
                        in_proj_weight_q, in_proj_weight_kv,
                        out_proj_weight,
                        attn_mask, dropout):

    try:
        from apex.contrib.multihead_attn.fast_encdec_multihead_attn_func import fast_encdec_attn_func
    except ModuleNotFoundError as e:
        print("Cannot use fast self-attention implementation")

    return fast_encdec_attn_func(time_masking, is_training, num_heads, query, key,
                                 in_proj_weight_q, in_proj_weight_kv,
                                 out_proj_weight,
                                 attn_mask, dropout)
