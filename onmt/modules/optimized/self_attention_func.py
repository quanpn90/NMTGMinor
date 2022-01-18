"""
Self-attention with multi-head attention.
Code is taken from apex self-attention implementation
https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/multihead_attn
"""

import torch
import torch.nn.functional as F

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .compat import custom_fwd, custom_bwd

try:
    import self_multihead_attn_cuda
except (ModuleNotFoundError, ImportError) as e:
    self_multihead_attn_cuda = None

try:
    import self_multihead_attn_blaslt
except (ModuleNotFoundError, ImportError) as e:
    self_multihead_attn_blaslt = None


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rotate_backward(dx):
    dx2, dx1 = dx[..., :dx.shape[-1] // 2], dx[..., dx.shape[-1] // 2:]
    return torch.cat((dx1, -dx2), dim=dx1.ndim - 1)


class SelfAttnFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd()
    def forward(ctx, use_time_mask, is_training, heads, inputs,
                input_weights, output_weights,
                input_biases, output_biases,
                mask, dropout_prob,
                rotary_pos_enc, pos_emb,
                incremental, incremental_cache,
                low_precision, return_coverage):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs.size(2) // heads
        scale_t = torch.tensor([head_dim ** -0.5])

        ctx.rotary_pos_enc = rotary_pos_enc
        ctx.return_coverage = return_coverage
        ctx.low_precision = low_precision
        ctx.use_time_mask = use_time_mask

        bsz, len_q = inputs.size(1), inputs.size(0)

        # print(low_precision, incremental, inputs.type())
        if low_precision and self_multihead_attn_cuda is not None and not incremental and len_q <= 2048 \
                and inputs.type() == 'torch.cuda.HalfTensor' \
                and not rotary_pos_enc:
            ctx.fused = True

            if mask is not None:
                if use_time_mask:
                    mask = mask.bool()
                else:  # [b x len_k] -> [b x 1 x 1 x len_k]
                    mask = mask.unsqueeze(1).unsqueeze(2).bool()
            else:
                if use_time_mask:
                    mask = inputs.new(len_q, len_q).zero_().bool()
                else:
                    mask = inputs.new(bsz, 1, 1, len_q).zero_().bool()  # works

            if self_multihead_attn_blaslt is not None:
                # print("Using ATTN BLASLT")
                input_lin_results, \
                attn_scores, \
                dropout_results, \
                dropout_mask, \
                matmul2_results, \
                outputs = self_multihead_attn_blaslt.forward(use_time_mask, is_training, heads,
                                                           inputs.contiguous(), input_weights, output_weights,
                                                           input_biases, output_biases,
                                                           mask, dropout_prob)
            else:
                input_lin_results, \
                attn_scores, \
                dropout_results, \
                dropout_mask, \
                matmul2_results, \
                outputs = self_multihead_attn_cuda.forward(use_time_mask, is_training, heads,
                                                           inputs.contiguous(), input_weights, output_weights,
                                                           input_biases, output_biases,
                                                           mask, dropout_prob)

            ctx.save_for_backward(heads_t,
                                  scale_t,
                                  matmul2_results,
                                  dropout_results,
                                  attn_scores,
                                  input_lin_results,
                                  inputs,
                                  input_weights,
                                  output_weights,
                                  dropout_mask,
                                  dropout_prob_t,
                                  mask)

            return outputs, dropout_results

        ctx.fused = False

        # Input Linear GEMM
        # input1: (activations) [seql_q, seqs, embed_dim(1024)]
        # input2: (weights)     [embed_dim*3 (3072), embed_dim (1024)] (transpose [0,1])
        # output:               [seql_q, seqs, embed_dim*3]
        # GEMM: ( (seql_q*seqs) x embed_dim ) x ( embed_dim x embed_dim*3 ) = (seql_q*seqs x embed_dim*3)
        input_lin_results = torch.addmm(input_biases,
                                        inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)),
                                        input_weights.transpose(0, 1),
                                        beta=1., alpha=1.)

        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # input_lin_results: [seql_q, seqs, heads(16), 3, head_dim(64)]
        # input_lin_results: [seql_q, batches=seqs*heads, 3, head_dim]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]

        if incremental:
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

        len_k = keys.size(0)

        # apply rotary position encodings
        if rotary_pos_enc:
            assert pos_emb is not None and pos_emb is not None
            cos, sin = pos_emb
            queries_, keys_ = apply_rotary_pos_emb(queries, keys, cos, sin)
            queries.copy_(queries_)
            keys.copy_(keys_)
        else:
            sin, cos = null_tensor, null_tensor

        # Matmul1 Batched GEMMs
        # The output tensor is specified prior to the Batch GEMM because baddbmm requires its specification
        # baddbmm is used to apply the scale parameter via the Batched GEMM's alpha parameter instead of
        # a separate elementwise operation.
        # Input1: (Queries) [seql_q, seqs*heads, head_dim] tranpose(0,1)
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
            if use_time_mask:
                assert (len(mask.size()) == 2), "Timing mask is not 2D!"
                mask = mask.to(torch.bool)
                matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            # Key Padding Mask
            else:
                batches, seql_q, seql_k = matmul1_results.size()
                seqs = int(batches / heads)
                matmul1_results = matmul1_results.view(seqs, heads, seql_q, seql_k)
                mask = mask.to(torch.bool)
                matmul1_results = matmul1_results.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                matmul1_results = matmul1_results.view(seqs * heads, seql_q, seql_k)

        # Softmax and Dropout attention
        softmax_results = F.softmax(matmul1_results, dim=-1)

        # Dropout - is not executed for inference
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=(1. - dropout_prob_t[0]))
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor

        nan_mask = torch.isnan(dropout_results)
        if nan_mask.any():
            dropout_results.masked_fill_(nan_mask, 0)

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
                                          dtype=dropout_results.dtype, device=queries.device).transpose(1, 0)
            matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        else:
            matmul2_results = torch.matmul(dropout_results, values.transpose(0, 1))
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs.size(0), inputs.size(1),
                                                                            inputs.size(2))

        # Output Linear GEMM
        # Input1: (activations) [seql_q, seqs, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ] transpose(0,1)
        # Output:               [ seql_q, seqs, embed_dim ]
        # GEMM: ( seql_q*seqs x embed_dim ) x ( embed_dim x embed_dim ) = ( seql_q*seqs x embed_dim )
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
                              inputs,
                              input_weights,
                              output_weights,
                              dropout_mask,
                              dropout_prob_t,
                              sin, cos)

        if return_coverage:
            return (outputs, dropout_results)
        else:
            return (outputs,)

    @staticmethod
    @custom_bwd
    def backward(ctx, *output_grads):

        if ctx.return_coverage:
            output_grads, coverage_grads = output_grads
        else:
            output_grads = output_grads[0]

        if ctx.fused:
            heads_t, \
            scale_t, \
            matmul2_results, \
            dropout_results, \
            attn_scores, \
            input_lin_results, \
            inputs, \
            input_weights, \
            output_weights, \
            dropout_mask, \
            dropout_prob_t, pad_mask = ctx.saved_tensors

            if input_weights.requires_grad:

                if self_multihead_attn_blaslt is not None:
                    input_grads, \
                    input_weight_grads, \
                    output_weight_grads, \
                    input_bias_grads, \
                    output_bias_grads = \
                        self_multihead_attn_blaslt.backward(ctx.use_time_mask, heads_t[0],
                                                          output_grads.contiguous(), matmul2_results,
                                                          dropout_results, attn_scores,
                                                          input_lin_results, inputs, input_weights,
                                                          output_weights, dropout_mask, dropout_prob_t[0])
                else:
                    input_grads, \
                        input_weight_grads, \
                        output_weight_grads, \
                        input_bias_grads, \
                        output_bias_grads = \
                        self_multihead_attn_cuda.backward(ctx.use_time_mask, heads_t[0],
                                                          output_grads.contiguous(), matmul2_results,
                                                          dropout_results, attn_scores,
                                                          input_lin_results, inputs, input_weights,
                                                          output_weights, dropout_mask, dropout_prob_t[0])

            else:
                input_grads = self_multihead_attn_cuda.backward_input_only(ctx.use_time_mask, heads_t[0],
                                                                           output_grads.contiguous(), matmul2_results,
                                                                           dropout_results, attn_scores,
                                                                           input_lin_results, inputs, input_weights,
                                                                           output_weights, dropout_mask,
                                                                           dropout_prob_t[0])
                input_weight_grads = None
                input_bias_grads = None
                output_weight_grads = None
                output_bias_grads = None

            return None, None, None, \
                   input_grads, \
                   input_weight_grads, output_weight_grads, \
                   input_bias_grads, output_bias_grads, \
                   None, None, None, None, None, None, None, None

        heads_t, \
        scale_t, \
        matmul2_results, \
        dropout_results, \
        softmax_results, \
        input_lin_results, \
        inputs, \
        input_weights, \
        output_weights, \
        dropout_mask, \
        dropout_prob_t, \
        sin, cos = ctx.saved_tensors

        head_dim = inputs.size(2) // heads_t.item()

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # input_lin_results: [seql_q, seqs, heads(16), 3, head_dim(64)]
        # input_lin_results: [seql_q, batches=seqs*heads, 3, head_dim]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads_t[0], 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]

        len_key = keys.size(0)

        # Slice out q,k,v from one big set of gradients entering the input linear's bprop
        # (should only impact meta data, no copies!)
        # The gradients are identical in size to the Input Linear outputs.
        # The tensor is declared before hand to properly slice out query, key, and value grads.
        input_lin_results_grads = torch.empty_like(input_lin_results)
        queries_grads = input_lin_results_grads[:, :, 0, :]
        keys_grads = input_lin_results_grads[:, :, 1, :]
        values_grads = input_lin_results_grads[:, :, 2, :]

        # Output Linear GEMM - DGRAD
        # Input1: (data grads)  [seql_q, seqs, embed_dim=heads*head_dim]
        # Input2: (weights)     [ embed_dim, embed_dim ]
        # Output:               [ seql_q, seqs, embed_dim ]
        # GEMM: ( seql_q*seqs x embed_dim ) x ( embed_dim x embed_dim ) = ( seql_q*seqs x embed_dim )
        output_grads = output_grads.contiguous()

        output_lin_grads = torch.mm(
            output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        # Output Linear GEMM - WGRAD
        # Input1: (data grads)  [seql_q*seqs, embed_dim=heads*head_dim] transpose(0,1)
        # Input2: (activations) [seql_q*seqs, embed_dim ]
        # Output:               [ seql_q, seqs, embed_dim ]
        # GEMM: ( embed_dim x seql_q*seqs ) x ( seql_q*seqs x embed_dim ) = ( embed_dim x embed_dim )

        if output_weights.requires_grad:
            output_weight_grads = torch.mm(
                output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1),
                matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
            output_bias_grads = torch.sum(
                output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_weight_grads = None
            output_bias_grads = None
        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.size(1) * heads_t[0], head_dim).transpose(0, 1)

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
        torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1),
                      out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        # Matmul1 - DGRAD2
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k] transpose(1,2)
        # Input2: (activations) [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_k, head_dim] transpose(0,1)
        # GEMM: Per batch: ( seql_k x seql_q ) x ( seql_q x head_dim ) = ( seql_k x head_dim )
        torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1),
                      out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])

        if ctx.rotary_pos_enc:
            queries_grads_ = queries_grads * cos + rotate_backward(sin * queries_grads)
            keys_grads_ = keys_grads * cos + rotate_backward(sin * keys_grads)
            queries_grads.copy_(queries_grads_)
            keys_grads.copy_(keys_grads_)

        # Input Linear GEMM - DGRAD
        # input1: (data grads) [seql_q, seqs, 3*embed_dim(3072)]
        # input2: (weights)    [embed_dim*3 (3072), embed_dim (1024)]
        # output:              [seql_q, seqs, embed_dim]
        # GEMM: ( (seql_q*seqs) x 3*embed_dim ) x ( 3*embed_dim x embed_dim ) = (seql_q*seqs x embed_dim)
        input_lin_results_grads = input_lin_results_grads.view(inputs.size(0) * inputs.size(1),
                                                               heads_t[0] * 3 * head_dim)
        input_grads = torch.mm(input_lin_results_grads, input_weights)
        input_grads = input_grads.view(inputs.size(0), inputs.size(1), inputs.size(2))
        # Input Linear GEMM - WGRAD
        # input1: (data grads)  [seql_q*seqs, 3*embed_dim(3072)]
        # input2: (activations) [seql_q*seqs, embed_dim(1024)]
        # output:               [3*embed_dim, embed_dim]
        # GEMM: ( 3*embed_dim x seql_q*seqs ) x ( seql_q*seqs x embed_dim ) = (3*embed_dim x embed_dim)

        if input_weights.requires_grad:
            input_weight_grads = torch.mm(input_lin_results_grads.transpose(0, 1),
                                          inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)))

            input_bias_grads = torch.sum(input_lin_results_grads, 0)
        else:
            input_weight_grads = None
            input_bias_grads = None

        return None, None, None, \
               input_grads, \
               input_weight_grads, output_weight_grads, \
               input_bias_grads, output_bias_grads, \
               None, None, None, None, None, None, None, None


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        try:
            return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())
        except AttributeError:
            return torch.cuda.amp.autocast_mode._cast(args, torch.half)


def self_attn_func(*args):
    args = _cast_if_autocast_enabled(*args)
    with torch.cuda.amp.autocast(enabled=False):
        return SelfAttnFunc.apply(*args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=32,
                        help='Size of embedding / transformer hidden')
    parser.add_argument('-gpu', default=0, type=int,
                        help="Seed for deterministic runs.")

    test_function = self_attn_func

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

            self.in_proj_weight = torch.Tensor(3 * model_size, model_size)
            self.out_proj_weight = torch.Tensor(model_size, model_size)

            self.in_proj_bias = torch.Tensor(3 * model_size)
            self.out_proj_bias = torch.Tensor(model_size)

            self.reset_parameters()

        def reset_parameters(self):
            std_ = 0.02
            torch.nn.init.normal_(self.in_proj_weight, 0.0, std_)
            torch.nn.init.normal_(self.out_proj_weight, 0.0, std_)

            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj_bias, 0.)


    class TestAttention(torch.nn.Module):

        def __init__(self, test_function, model_size=16, heads=1):
            super().__init__()
            self.model_size = model_size
            self.heads = heads
            self.head_dim = model_size // heads

            self.function = test_function

        def forward(self, input_weights, output_weights, input, input_biases, output_biases, mask,
                    use_time_mask=False):
            is_training = True
            dropout = 0.0
            double_precision = True
            return_coverage = False

            # use_time_mask, is_training, heads, inputs,
            # input_weights, output_weights,
            # input_biases, output_biases,
            # mask, dropout_prob,
            # rotary_pos_enc, pos_emb,
            # incremental, incremental_cache,
            # return_coverage

            return self.function(use_time_mask, is_training, self.heads, input,
                                 input_weights, output_weights,
                                 input_biases, output_biases,
                                 mask, dropout,
                                 False, None,  # For the incremental stuff
                                 False, None,
                                 return_coverage)  # double precision set to true


    bsz = 4
    len_q = 15
    len_r = len_q

    input_states = torch.randn(*(len_q, bsz, opt.model_size)).double().cuda()
    input_states.requires_grad = True
    net = TestAttention(test_function, model_size=opt.model_size, heads=opt.n_heads)

    parameters = Parameters(opt.model_size, opt.n_heads)

    in_proj_weight = parameters.in_proj_weight.double().cuda()
    out_proj_weight = parameters.out_proj_weight.double().cuda()

    in_proj_bias = parameters.in_proj_bias.double().cuda()
    out_proj_bias = parameters.out_proj_bias.double().cuda()

    in_proj_weight.requires_grad = True
    out_proj_weight.requires_grad = True
    in_proj_bias.requires_grad = True
    out_proj_bias.requires_grad = True

    mask = input_states.new(*(bsz, len_r)).bernoulli_(p=0.25).bool()

    print("gradchecking start.")

    use_time_mask = False

    torch.autograd.gradcheck(net, (in_proj_weight, out_proj_weight, input_states,
                                   in_proj_bias, out_proj_bias,
                                   mask, use_time_mask), atol=1e-04, rtol=0.001)

    mask = input_states.new(*(len_q, len_r)).bernoulli_(p=0.25).bool()

    print("gradchecking with time mask start.")

    use_time_mask = True

    torch.autograd.gradcheck(net, (in_proj_weight, out_proj_weight, input_states,
                                   in_proj_bias, out_proj_bias,
                                   mask, use_time_mask), atol=1e-04, rtol=0.001)

    print("gradchecking completed.")
