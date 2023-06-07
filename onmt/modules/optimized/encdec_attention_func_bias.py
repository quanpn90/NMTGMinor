"""
Encoder-Decoder multi-head attention.
Code is heavily adapted from apex
https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/multihead_attn
"""

import torch
import torch.nn.functional as F

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .compat import custom_fwd, custom_bwd

try:
    import encdec_multihead_attn_bias_cuda
except (ModuleNotFoundError, ImportError) as e:
    encdec_multihead_attn_bias_cuda = None

try:
    import encdec_multihead_attn_bias_blaslt
except (ModuleNotFoundError, ImportError) as e:
    encdec_multihead_attn_bias_blaslt = None


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0


# only 1 term this time
def apply_rotary_pos_emb(q, cos, sin):
    return (q * cos) + (rotate_half(q) * sin)


def rotate_backward(dx):
    dx2, dx1 = dx[..., :dx.shape[-1] // 2], dx[..., dx.shape[-1] // 2:]
    return torch.cat((dx1, -dx2), dim=dx1.ndim - 1)


class EncdecAttnBiasFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd()
    def forward(ctx, recompute, is_training, heads, inputs_q, inputs_kv,
                input_weights_q, input_weights_kv, output_weights,
                input_bias_q, input_bias_kv, output_bias,
                mask, dropout_prob,
                incremental, incremental_cache,
                rotary_pos_enc, pos_emb_q, pos_emb_k,
                low_precision, return_coverage):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([]).to(inputs_q.device)
        head_dim = inputs_q.size(2) // heads
        scale_t = torch.tensor([head_dim ** -0.5])
        use_mask = (mask is not None)

        bsz, len_q, len_k = inputs_q.size(1), inputs_q.size(0), inputs_kv.size(0)
        ctx.incremental = incremental
        ctx.fused_softmax_dropout = False
        ctx.fused_all = False
        ctx.len_q = len_q
        ctx.len_k = len_k
        ctx.low_precision = low_precision
        ctx.return_coverage = return_coverage
        ctx.recompute = recompute
        ctx.rotary_pos_enc = rotary_pos_enc

        if encdec_multihead_attn_bias_cuda is not None and not incremental and len_k <= 2048 \
                and inputs_q.type() == 'torch.cuda.HalfTensor' and not rotary_pos_enc and low_precision:

            mask_ = mask
            # print("[DEBUGGING] FAST CUDA ENCDEC ATTENTION BIAS")
            # mask = mask.half() * -10000
            mask = mask.unsqueeze(1).unsqueeze(2).bool()

            cuda_module = encdec_multihead_attn_bias_blaslt if encdec_multihead_attn_bias_blaslt is not None \
                else encdec_multihead_attn_bias_cuda

            if encdec_multihead_attn_bias_blaslt is None:
                ctx.recompute = False

            input_lin_q_results, input_lin_kv_results, \
                attn_scores, dropout_results, dropout_mask, \
                matmul2_results, outputs \
                = cuda_module.forward(is_training, heads, inputs_q, inputs_kv,
                                      input_weights_q, input_weights_kv,
                                      output_weights, input_bias_q, input_bias_kv, output_bias,
                                      mask, dropout_prob)

            sinq, cosq, = null_tensor, null_tensor
            sink, cosk, = null_tensor, null_tensor

            if ctx.recompute:
                del matmul2_results, dropout_results, attn_scores, input_lin_q_results, input_lin_kv_results,

                ctx.save_for_backward(heads_t,
                                      scale_t,
                                      null_tensor, null_tensor, null_tensor, null_tensor, null_tensor,
                                      inputs_q,
                                      inputs_kv,
                                      input_weights_q,
                                      input_weights_kv,
                                      input_bias_q,
                                      input_bias_kv,
                                      output_weights,
                                      dropout_mask, mask,
                                      dropout_prob_t,
                                      sinq, cosq, sink, cosk)

                dropout_results = null_tensor
            else:
                ctx.save_for_backward(heads_t,
                                      scale_t,
                                      matmul2_results,
                                      dropout_results,
                                      attn_scores,
                                      input_lin_q_results,
                                      input_lin_kv_results,
                                      inputs_q,
                                      inputs_kv,
                                      input_weights_q,
                                      input_weights_kv,
                                      input_bias_q,
                                      input_bias_kv,
                                      output_weights,
                                      dropout_mask, mask,
                                      dropout_prob_t,
                                      sinq, cosq, sink, cosk)

            ctx.fused_all = True

            if return_coverage:
                return outputs, dropout_results
            else:
                return (outputs,)

        if mask is not None:
            # Self Attention Pad Mask
            mask = mask.to(torch.bool)

            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)  # for the head dimension
            else:
                mask = mask.unsqueeze(1).unsqueeze(2)  # for the head and query dimension

        # Input Linear GEMM Q
        # input1: (activations) [seql_q, bsz, embed_dim] -> [len_q * bsz, embed_dim]
        # input2: (weights)     [embed_dim, embed_dim]. transpose(0, 1)
        # output:               [len_q * bsz, embed_dim] -> [seql_q, bsz, embed_dim]
        # GEMM: ( (seql_q*seqs) x embed_dim ) x ( embed_dim x embed_dim ) = (seql_q*seqs x embed_dim)
        input_lin_q_results = torch.addmm(input_bias_q,
                                          inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                                          input_weights_q.transpose(0, 1),
                                          beta=1., alpha=1.)
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
            input_lin_kv_results = torch.addmm(input_bias_kv,
                                               inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)),
                                               input_weights_kv.transpose(0, 1),
                                               beta=1., alpha=1.)
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

        # TODO: rotary pos encoding
        if rotary_pos_enc:
            assert pos_emb_q is not None and pos_emb_k is not None
            cosq, sinq = pos_emb_q
            queries = apply_rotary_pos_emb(queries, cosq, sinq)
            cosk, sink = pos_emb_k
            keys_ = apply_rotary_pos_emb(keys, cosk, sink)
            keys.copy_(keys_)
        else:
            sinq, cosq = null_tensor, null_tensor
            sink, cosk = null_tensor, null_tensor

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
            batches, seql_q, seql_k = matmul1_results.size()
            bsz = int(batches / heads)
            matmul1_results = matmul1_results.view(bsz, heads, seql_q, seql_k)
            # after unsqueezing the mask should have size [bsz x 1 x 1 x seql_k]
            matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            matmul1_results = matmul1_results.view(bsz * heads, seql_q, seql_k)

        if matmul1_results.type() == 'torch.cuda.HalfTensor':
            softmax_results = F.softmax(matmul1_results, dim=-1, dtype=torch.float32).type_as(matmul1_results)
        else:
            softmax_results = F.softmax(matmul1_results, dim=-1)

        nan_mask = torch.isnan(softmax_results)
        if nan_mask.any():
            softmax_results.masked_fill_(nan_mask, 0)

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
                                          dtype=dropout_results.dtype, device=dropout_results.device)
            torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results.transpose(1, 0))
        else:
            matmul2_results = torch.matmul(dropout_results, values.transpose(0, 1)).transpose(0, 1)

        # view from [len_q, bsz*heads, head_dim] to [len_q, bsz, embed]
        matmul2_results = matmul2_results.contiguous().view(inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))

        # Output Linear GEMM
        # Input1: (activations) [seql_q, seqs, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ] transpose(0,1)
        # Output:               [ seql_q, seqs, embed_dim ]
        # GEMM: ( seql_q*seqs x embed_dim ) x ( embed_dim x embed_dim ) = ( seql_q*seqs x embed_dim )
        outputs = torch.addmm(output_bias,
                              matmul2_results.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                              output_weights.transpose(0, 1),
                              beta=1., alpha=1.)
        outputs = outputs.view(inputs_q.size(0), inputs_q.size(1), output_weights.size(0))

        if not ctx.recompute:
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
                                  input_bias_q,
                                  input_bias_kv,
                                  output_weights,
                                  dropout_mask, mask,
                                  dropout_prob_t,
                                  sinq, cosq, sink, cosk)
        else:
            ctx.save_for_backward(heads_t,
                                  scale_t,
                                  null_tensor, null_tensor, null_tensor, null_tensor, null_tensor,
                                  inputs_q,
                                  inputs_kv,
                                  input_weights_q,
                                  input_weights_kv,
                                  input_bias_q,
                                  input_bias_kv,
                                  output_weights,
                                  dropout_mask, mask,
                                  dropout_prob_t,
                                  sinq, cosq, sink, cosk)

            del input_lin_q_results, queries
            del input_lin_kv_results, keys, values
            del matmul1_results, matmul2_results
            del softmax_results, dropout_results
            dropout_results = null_tensor

        if return_coverage:
            return (outputs, dropout_results)
        else:
            return (outputs,)

    @staticmethod
    @custom_bwd
    def backward(ctx, *output_grads):

        incremental = ctx.incremental
        len_q = ctx.len_q
        len_key = ctx.len_k

        if ctx.return_coverage:
            output_grads, coverage_grads = output_grads
        else:
            output_grads = output_grads[0]

        output_grads = output_grads.contiguous()

        if ctx.fused_all:
            assert encdec_multihead_attn_bias_cuda is not None and len_key <= 2048

            heads_t, \
                scale_t, \
                matmul2_results, \
                dropout_results, \
                attn_scores,\
                input_lin_q_results, \
                input_lin_kv_results, \
                inputs_q, \
                inputs_kv, \
                input_weights_q, \
                input_weights_kv, \
                input_bias_q, \
                input_bias_kv, \
                output_weights, \
                dropout_mask, mask,\
                dropout_prob_t, \
                sinq, cosq, sink, cosk = ctx.saved_tensors

            if input_weights_q.requires_grad:

                cuda_module = encdec_multihead_attn_bias_blaslt if encdec_multihead_attn_bias_blaslt is not None \
                    else encdec_multihead_attn_bias_cuda

                if ctx.recompute:
                    input_q_grads, \
                    input_kv_grads, \
                    input_weight_q_grads, \
                    input_weight_kv_grads, \
                    output_weight_grads, \
                    input_bias_q_grads, input_bias_kv_grads, output_bias_grads \
                        = cuda_module.backward_recompute(heads_t[0], output_grads,
                                                         inputs_q, inputs_kv,
                                                         input_weights_q,
                                                         input_weights_kv,
                                                         input_bias_q,
                                                         input_bias_kv,
                                                         output_weights, dropout_mask, mask,
                                                         dropout_prob_t[0])
                else:
                    input_q_grads, \
                        input_kv_grads, \
                        input_weight_q_grads, \
                        input_weight_kv_grads, \
                        output_weight_grads, \
                        input_bias_q_grads, input_bias_kv_grads, output_bias_grads \
                        = cuda_module.backward(heads_t[0], output_grads, matmul2_results,
                                               dropout_results,
                                               attn_scores,
                                               input_lin_q_results,
                                               input_lin_kv_results,
                                               inputs_q, inputs_kv, input_weights_q,
                                               input_weights_kv,
                                               output_weights, dropout_mask,
                                               dropout_prob_t[0])

            else:
                input_q_grads, \
                input_kv_grads, \
                    = encdec_multihead_attn_bias_cuda.backward_input_only(heads_t[0], output_grads, matmul2_results,
                                                                          dropout_results,
                                                                          attn_scores,
                                                                          input_lin_q_results,
                                                                          input_lin_kv_results,
                                                                          inputs_q, inputs_kv, input_weights_q,
                                                                          input_weights_kv,
                                                                          output_weights, dropout_mask,
                                                                          dropout_prob_t[0])

                input_weight_q_grads, \
                    input_weight_kv_grads, \
                    output_weight_grads, \
                    input_bias_q_grads, input_bias_kv_grads, output_bias_grads \
                    = None, None, None, None, None, None

            del ctx.recompute
            del ctx.fused_all

            return None, None, None \
                , input_q_grads, input_kv_grads \
                , input_weight_q_grads, input_weight_kv_grads, output_weight_grads \
                , input_bias_q_grads, input_bias_kv_grads, output_bias_grads \
                , None, None, None, None, None, None, None, None, None


        heads_t, scale_t, matmul2_results, dropout_results, softmax_results, \
        input_lin_q_results, input_lin_kv_results, \
        inputs_q, inputs_kv, \
        input_weights_q, input_weights_kv, input_bias_q, input_bias_kv, output_weights,  \
        dropout_mask, pad_mask, dropout_prob_t, \
        sinq, cosq, sink, cosk, \
            = ctx.saved_tensors

        head_dim = inputs_q.size(2) // heads_t.item()
        bsz = inputs_q.size(1)

        if ctx.recompute:
            assert ctx.incremental is not True
            heads = heads_t[0]

            # Recomputing the tensors in the forward pass here
            input_lin_q_results = torch.addmm(input_bias_q,
                                              inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                                              input_weights_q.transpose(0, 1),
                                              beta=1., alpha=1.)
            input_lin_q_results = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1), input_weights_q.size(0))

            queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads, head_dim)

            input_lin_kv_results = torch.addmm(input_bias_kv,
                                               inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)),
                                               input_weights_kv.transpose(0, 1),
                                               beta=1., alpha=1.)
            input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1),
                                                             input_weights_kv.size(0))

            input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1) * heads, 2, head_dim)
            keys = input_lin_kv_results[:, :, 0, :]
            values = input_lin_kv_results[:, :, 1, :]

            matmul1_results = torch.empty((queries.size(1), queries.size(0), keys.size(0)), dtype=queries.dtype,
                                          device=queries.device)
            matmul1_results.baddbmm_(queries.transpose(0, 1),
                                     keys.transpose(0, 1).transpose(1, 2),
                                     beta=0.0, alpha=scale_t[0])

            if pad_mask is not None:
                batches, seql_q, seql_k = matmul1_results.size()
                bsz = int(batches / heads)
                matmul1_results = matmul1_results.view(bsz, heads, seql_q, seql_k)
                # after unsqueezing the mask should have size [bsz x 1 x 1 x seql_k]
                matmul1_results = matmul1_results.masked_fill_(pad_mask, float('-inf'))
                matmul1_results = matmul1_results.view(bsz * heads, seql_q, seql_k)

            if matmul1_results.type() == 'torch.cuda.HalfTensor':
                softmax_results = F.softmax(matmul1_results, dim=-1, dtype=torch.float32).type_as(matmul1_results)
            else:
                softmax_results = F.softmax(matmul1_results, dim=-1)

            nan_mask = torch.isnan(softmax_results)
            if nan_mask.any():
                softmax_results.masked_fill_(nan_mask, 0)

            if dropout_prob_t[0] > 0:
                pinv = 1.0 / (1.0 - dropout_prob_t[0])
                dropout_results = softmax_results * dropout_mask * pinv
            else:
                dropout_results = softmax_results

            matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)),
                                          dtype=dropout_results.dtype, device=dropout_results.device)

            torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results.transpose(1, 0))

            matmul2_results = matmul2_results.contiguous().view(inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))

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

        if input_weights_q.requires_grad:
            output_weight_grads = torch.mm(
                output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1),
                matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))

            output_bias_grads = torch.sum(
                output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_weight_grads, output_bias_grads = None, None

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
        try:
            softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results.dtype)
        except TypeError:
            softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)

        # Matmul1 - DGRAD1
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k]
        # Input2: (activations) [seql_k, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_q, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_q x seql_k ) x ( seql_k x head_dim ) = ( seql_q x head_dim )
        torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1),
                      out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        # Matmul1 - DGRAD2
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k] transpose(1,2)
        # Input2: (activations) [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_k, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_k x seql_q ) x ( seql_q x head_dim ) = ( seql_k x head_dim )
        torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1),
                      out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])

        # TODO:
        if ctx.rotary_pos_enc:
            queries_grads = queries_grads * cosq + rotate_backward(sinq * queries_grads)
            keys_grads_ = keys_grads * cosk + rotate_backward(sink * keys_grads)
            keys_grads.copy_(keys_grads_)

        # Input Q Linear GEMM - DGRAD
        # input1: (data grads) [seql_q, seqs, embed_dim(1024)]
        # input2: (weights)    [embed_dim (1024), embed_dim (1024)]
        # output:              [seql_q, seqs, embed_dim]
        # GEMM: ( (seql_q*seqs) x embed_dim ) x ( embed_dim x embed_dim ) = (seql_q*seqs x embed_dim)
        queries_grads = queries_grads.view(inputs_q.size(0) * inputs_q.size(1), heads_t[0] * head_dim)
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
        if input_weights_q.requires_grad:
            input_weight_q_grads = torch.mm(queries_grads.transpose(0, 1),
                                            inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)))

            input_bias_q_grads = torch.sum(queries_grads, 0)
        else:
            input_weight_q_grads, input_bias_q_grads = None, None
        # Input KV Linear GEMM - WGRAD
        # input1: (data grads)  [seql_k*seqs, 2*embed_dim(2048)]
        # input2: (activations) [seql_k*seqs, embed_dim(1024)]
        # output:               [2*embed_dim, embed_dim]
        # GEMM: ( 2*embed_dim x seql_k*seqs ) x ( seql_k*seqs x embed_dim ) = (2*embed_dim x embed_dim)

        if input_weights_q.requires_grad:
            input_weight_kv_grads = torch.mm(input_lin_kv_results_grads.transpose(0, 1),
                                             inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)))

            input_bias_kv_grads = torch.sum(input_lin_kv_results_grads, 0)
        else:
            input_weight_kv_grads, input_bias_kv_grads = None, None

        return None, None, None \
            , input_q_grads, input_kv_grads \
            , input_weight_q_grads, input_weight_kv_grads, output_weight_grads \
            , input_bias_q_grads, input_bias_kv_grads, output_bias_grads \
            , None, None, None, None, None, None, None, None, None


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


def encdec_attn_bias_func(*args):
    args = _cast_if_autocast_enabled(*args)
    with torch.cuda.amp.autocast(enabled=False):
        return EncdecAttnBiasFunc.apply(*args)


class EncdecAttnBiasCompactFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd()
    def forward(ctx, recompute, is_training, heads,
                input_lin_q_results, input_lin_kv_results,
                mask, dropout_prob,
                incremental, incremental_cache,
                rotary_pos_enc, pos_emb_q, pos_emb_k,
                low_precision, return_coverage):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([]).to(input_lin_q_results.device)
        head_dim = input_lin_q_results.size(2) // heads
        scale_t = torch.tensor([head_dim ** -0.5])
        use_mask = (mask is not None)

        bsz, len_q, len_k = input_lin_q_results.size(1), input_lin_q_results.size(0), input_lin_kv_results.size(0)
        ctx.incremental = incremental
        ctx.fused_softmax_dropout = False
        ctx.fused_all = False
        ctx.len_q = len_q
        ctx.len_k = len_k
        ctx.low_precision = low_precision
        ctx.return_coverage = return_coverage
        ctx.recompute = recompute
        ctx.rotary_pos_enc = rotary_pos_enc

        if encdec_multihead_attn_bias_cuda is not None and not incremental and len_k <= 2048 \
                and input_lin_q_results.type() == 'torch.cuda.HalfTensor' and not rotary_pos_enc and low_precision:

            mask_ = mask

            mask = mask.unsqueeze(1).unsqueeze(2).bool()

            cuda_module = encdec_multihead_attn_bias_blaslt
            ctx.recompute = False

            attn_scores, dropout_results, dropout_mask, \
                matmul2_results \
                = cuda_module.forward_compact(is_training, heads, input_lin_q_results, input_lin_kv_results,
                                      mask, dropout_prob)

            sinq, cosq, = null_tensor, null_tensor
            sink, cosk, = null_tensor, null_tensor

            ctx.save_for_backward(heads_t,
                                  scale_t,
                                  dropout_results,
                                  attn_scores,
                                  input_lin_q_results,
                                  input_lin_kv_results,
                                  dropout_mask, mask,
                                  dropout_prob_t,
                                  sinq, cosq, sink, cosk)


            ctx.fused_all = True

            if return_coverage:
                return matmul2_results, dropout_results
            else:
                return (matmul2_results,)

        if mask is not None:
            # Self Attention Pad Mask
            mask = mask.to(torch.bool)

            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)  # for the head dimension
            else:
                mask = mask.unsqueeze(1).unsqueeze(2)  # for the head and query dimension

        queries = input_lin_q_results.view(input_lin_q_results.size(0), input_lin_q_results.size(1) * heads, head_dim)

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

            input_lin_kv_results = input_lin_kv_results.view(input_lin_kv_results.size(0), input_lin_kv_results.size(1) * heads,
                                                             2, head_dim)
            keys = input_lin_kv_results[:, :, 0, :]
            values = input_lin_kv_results[:, :, 1, :]

            if incremental:
                keys = keys.contiguous().view(len_k, bsz, heads * head_dim)
                values = values.contiguous().view(len_k, bsz, heads * head_dim)

                incremental_cache['c_k'] = keys
                incremental_cache['c_v'] = values

                keys = keys.view(len_k, bsz * heads, head_dim)
                values = values.view(len_k, bsz * heads, head_dim)

        # TODO: rotary pos encoding
        if rotary_pos_enc:
            assert pos_emb_q is not None and pos_emb_k is not None
            cosq, sinq = pos_emb_q
            queries = apply_rotary_pos_emb(queries, cosq, sinq)
            cosk, sink = pos_emb_k
            keys_ = apply_rotary_pos_emb(keys, cosk, sink)
            keys.copy_(keys_)
        else:
            sinq, cosq = null_tensor, null_tensor
            sink, cosk = null_tensor, null_tensor

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
            batches, seql_q, seql_k = matmul1_results.size()
            bsz = int(batches / heads)
            matmul1_results = matmul1_results.view(bsz, heads, seql_q, seql_k)
            # after unsqueezing the mask should have size [bsz x 1 x 1 x seql_k]
            matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            matmul1_results = matmul1_results.view(bsz * heads, seql_q, seql_k)

        softmax_results = F.softmax(matmul1_results, dim=-1)

        nan_mask = torch.isnan(softmax_results)
        if nan_mask.any():
            softmax_results.masked_fill_(nan_mask, 0)

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
            matmul2_results = torch.empty((dropout_results.size(0), dropout_results.size(1), values.size(2)),
                                          dtype=dropout_results.dtype, device=dropout_results.device)
            torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        else:
            matmul2_results = torch.matmul(dropout_results, values.transpose(0, 1))

        matmul2_results = matmul2_results.transpose(0, 1).contiguous()

        # return [len_q, bsz*heads, head_dim]

        ctx.save_for_backward(heads_t,
                              scale_t,
                              matmul2_results,
                              dropout_results,
                              softmax_results,
                              input_lin_q_results,
                              input_lin_kv_results,
                              dropout_mask, mask,
                              dropout_prob_t,
                              sinq, cosq, sink, cosk)

        if return_coverage:
            return (matmul2_results, dropout_results)
        else:
            return (matmul2_results,)

    @staticmethod
    @custom_bwd
    def backward(ctx, *output_grads):

        incremental = ctx.incremental
        len_q = ctx.len_q
        len_key = ctx.len_k

        if ctx.return_coverage:
            output_lin_grads, coverage_grads = output_grads
        else:
            output_lin_grads = output_grads[0]

        output_lin_grads = output_lin_grads.contiguous()

        if ctx.fused_all:
            assert encdec_multihead_attn_bias_cuda is not None and len_key <= 2048

            heads_t, \
                scale_t, \
                dropout_results, \
                attn_scores,\
                input_lin_q_results, \
                input_lin_kv_results, \
                dropout_mask, mask,\
                dropout_prob_t, \
                sinq, cosq, sink, cosk = ctx.saved_tensors

            cuda_module = encdec_multihead_attn_bias_blaslt

            input_lin_q_results_grads, input_lin_kv_results_grads \
                = cuda_module.backward_compact(heads_t[0], output_lin_grads,
                                               dropout_results,
                                               attn_scores,
                                               input_lin_q_results,
                                               input_lin_kv_results,
                                               dropout_mask,
                                               dropout_prob_t[0])



            return None, None, None \
                , input_lin_q_results_grads, input_lin_kv_results_grads \
                , None, None, \
                   None, None, \
                   None, None, None,\
                   None, None


        heads_t, scale_t, dropout_results, softmax_results, \
        input_lin_q_results, input_lin_kv_results, \
        dropout_mask, pad_mask, dropout_prob_t, \
        sinq, cosq, sink, cosk, \
            = ctx.saved_tensors

        embed_dim = input_lin_q_results.size(2)
        head_dim = embed_dim.size(2) // heads_t.item()
        bsz = input_lin_q_results.size(1)

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

        # [seql_q, seqs*heads, head_dim] -> [seqs*heads, seql_q, head_dim]
        output_lin_grads = output_lin_grads.transpose(0, 1)

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
        try:
            softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results.dtype)
        except TypeError: # backward compatibility
            softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)

        # Matmul1 - DGRAD1
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k]
        # Input2: (activations) [seql_k, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_q, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_q x seql_k ) x ( seql_k x head_dim ) = ( seql_q x head_dim )
        torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1),
                      out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        # Matmul1 - DGRAD2
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k] transpose(1,2)
        # Input2: (activations) [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_k, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_k x seql_q ) x ( seql_q x head_dim ) = ( seql_k x head_dim )
        torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1),
                      out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])

        # TODO:
        if ctx.rotary_pos_enc:
            queries_grads = queries_grads * cosq + rotate_backward(sinq * queries_grads)
            keys_grads_ = keys_grads * cosk + rotate_backward(sink * keys_grads)
            keys_grads.copy_(keys_grads_)

        return None, None, None \
            , input_lin_q_results_grads, input_lin_kv_results_grads \
            , None, None, \
               None, None, \
               None, None, None, \
               None, None


def encdec_attn_bias_compact_func(*args):
    args = _cast_if_autocast_enabled(*args)
    with torch.cuda.amp.autocast(enabled=False):
        return EncdecAttnBiasCompactFunc.apply(*args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=32,
                        help='Size of embedding / transformer hidden')
    parser.add_argument('-gpu', default=0, type=int,
                        help="Seed for deterministic runs.")

    test_function = encdec_attn_bias_func

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

            self.in_proj_weight_q = torch.Tensor(model_size, model_size)
            self.in_proj_weight_kv = torch.Tensor(2 * model_size, model_size)
            self.out_proj_weight = torch.Tensor(model_size, model_size)

            self.in_proj_bias_q = torch.Tensor(model_size)
            self.in_proj_bias_kv = torch.Tensor(2 * model_size)
            self.out_proj_bias = torch.Tensor(model_size)

            self.reset_parameters()

        def reset_parameters(self):
            std_ = 0.02
            torch.nn.init.normal_(self.in_proj_weight_q, 0.0, std_)
            torch.nn.init.normal_(self.in_proj_weight_kv, 0.0, std_)
            torch.nn.init.normal_(self.out_proj_weight, 0.0, std_)

            torch.nn.init.constant_(self.in_proj_bias_q, 0.)
            torch.nn.init.constant_(self.in_proj_bias_kv, 0.)
            torch.nn.init.constant_(self.out_proj_bias, 0.)


    class TestAttention(torch.nn.Module):

        def __init__(self, test_function, model_size=16, heads=1):
            super().__init__()
            self.model_size = model_size
            self.heads = heads
            self.head_dim = model_size // heads

            self.function = test_function

        def forward(self, in_proj_weight_q, in_proj_bias_q, input, context, in_proj_weight_kv, in_proj_bias_kv,
                    out_proj_weight, out_proj_bias, mask,
                    recompute=False, use_rotary_enc=False, pos_emb_q=None, pos_emb_k=None):
            is_training = True
            dropout = 0.0
            low_precision = False
            return_coverage = False
            # .apply(time_masking, is_training,
            #        num_heads, query, key,
            #        in_proj_weight_q, in_proj_weight_kv,
            #        out_proj_weight, attn_mask, dropout,
            #        incremental, incremental_cache)

            return self.function(recompute, is_training, self.heads, input, context,
                                 in_proj_weight_q, in_proj_weight_kv, out_proj_weight,
                                 in_proj_bias_q, in_proj_bias_kv, out_proj_bias,
                                 mask, dropout,
                                 False, None,  # For the incremental stuff
                                 use_rotary_enc, pos_emb_q, pos_emb_k,
                                 low_precision, return_coverage)  # double precision set to true


    bsz = 4
    len_q = 5
    len_r = 15

    input_states = torch.randn(*(len_q, bsz, opt.model_size)).double().cuda()
    input_states.requires_grad = True
    net = TestAttention(test_function, model_size=opt.model_size, heads=opt.n_heads)

    parameters = Parameters(opt.model_size, opt.n_heads)

    in_proj_weight_q = parameters.in_proj_weight_q.double().cuda()
    in_proj_weight_kv = parameters.in_proj_weight_kv.double().cuda()
    out_proj_weight = parameters.out_proj_weight.double().cuda()

    in_proj_bias_q = parameters.in_proj_bias_q.double().cuda()
    in_proj_bias_kv = parameters.in_proj_bias_kv.double().cuda()
    out_proj_bias = parameters.out_proj_bias.double().cuda()

    in_proj_weight_q.requires_grad = True
    out_proj_weight.requires_grad = True
    in_proj_weight_kv.requires_grad = True
    in_proj_bias_q.requires_grad = True
    in_proj_bias_kv.requires_grad = True
    out_proj_bias.requires_grad = True

    mask = input_states.new(*(bsz, len_r)).bernoulli_(p=0.25).bool()

    print("gradchecking start.")
    #
    context = torch.randn(*(len_r, bsz, opt.model_size)).double().cuda()
    context.requires_grad = True
    #
    recompute = False

    try:
        torch.autograd.gradcheck(net, (in_proj_weight_q, in_proj_bias_q, input_states, context, in_proj_weight_kv,
                                       in_proj_bias_kv,
                                       out_proj_weight, out_proj_bias,
                                       mask, recompute), atol=1e-04, rtol=0.001)
    except RuntimeError as e:
        print(e)

    print("gradchecking completed.")


