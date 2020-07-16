import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.linear import FeedForward as position_wise_feed_forward
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from torch.autograd.function import Function
import sys
from torch.utils.checkpoint import get_device_states, set_device_states
from onmt.modules.dropout import variational_dropout


def deterministic_dropout(input, p=0.5, training=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return nn.functional.dropout(input, p=p, training=training)


class RelativeSelfAttention(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # self.layer_norm = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.layer_norm = nn.LayerNorm((opt.model_size,), elementwise_affine=True)
        # self.attn = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=1)
        self.attn = RelPartialLearnableMultiHeadAttn(opt.n_heads, opt.model_size, opt.model_size // opt.n_heads,
                                                     dropatt=opt.attn_dropout)
        self.dropout = opt.attn_dropout
        self.variational = opt.variational_dropout

    def forward(self, input, pos, attn_mask=None, incremental=False, incremental_cache=None, cleaning=False):
        q = self.layer_norm(input)
        attn, coverage, incremental_cache = self.attn(q, pos, attn_mask,
                                                      incremental=incremental, incremental_cache=incremental_cache)

        if not self.variational:
            o = F.dropout(attn, p=self.dropout, training=self.training, inplace=False)
        else:
            o = variational_dropout(attn, p=self.dropout, inplace=False, training=self.training)

        if cleaning:
            del q, attn
        return o, coverage, incremental_cache


class FeedForward(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.layer_norm = nn.LayerNorm((opt.model_size,), elementwise_affine=True)
        # self.layer_norm = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.feedforward = position_wise_feed_forward(opt.model_size, opt.inner_size, opt.dropout,
                                                      variational=opt.variational_dropout)
        self.dropout = opt.dropout
        self.variational = opt.variational_dropout

    def forward(self, input, cleaning=False):

        x_norm = self.layer_norm(input)
        x_ff = self.feedforward(x_norm)

        if not self.variational:
            o = F.dropout(x_ff, p=self.dropout, training=self.training, inplace=False)
        else:
            o = variational_dropout(x_ff, p=self.dropout, inplace=False, training=self.training)

        if cleaning:
            del x_norm, x_ff

        return o


class SourceAttention(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.layer_norm = nn.LayerNorm((opt.model_size,), elementwise_affine=True)
        self.attn = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=2)
        self.dropout = opt.attn_dropout
        self.variational = opt.variational_dropout

    def forward(self, input, context, attn_mask=None, incremental=False, incremental_cache=None, cleaning=False):
        q = self.layer_norm(input)
        attn, coverage, incremental_cache = self.attn(q, context, context, attn_mask, incremental, incremental_cache)

        if not self.variational:
            o = F.dropout(attn, p=self.dropout, training=self.training, inplace=False)
        else:
            o = variational_dropout(attn, p=self.dropout, inplace=False, training=self.training)

        if cleaning:
            del q, attn
        return o, coverage, incremental_cache


class ReversibleEncoderFunction(Function):

    @staticmethod
    def forward(ctx, hidden_states, pos, layers, attn_mask):

        attn_output, hidden_states = torch.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            # forward pass in the layer
            attn_output, hidden_states = layer(
                attn_output, hidden_states, pos, attn_mask
            )

        # attach params to ctx for backward

        # why should we detach here? because Y1 Y2 were built within torch.no_grad()
        # so cutting the backward from these variables seems unnecessary

        # save_for_backward will release memory more efficiently
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach(), pos)
        # ctx.save_for_backward(attn_output, hidden_states)
        ctx.layers = layers
        ctx.attn_mask = attn_mask

        with torch.no_grad():
            output = attn_output + hidden_states

        return output

        # concatenate 2 revnet outputs:
        # return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):

        # print(grad_hidden_states.sum())
        # grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)
        grad_attn_output = grad_hidden_states

        # retrieve params from ctx
        attn_output, hidden_states, pos = ctx.saved_tensors
        layers = ctx.layers
        attn_mask = ctx.attn_mask

        for idx, layer in enumerate(layers[::-1]):
            # backprop
            attn_output, hidden_states, grad_attn_output, grad_hidden_states = layer.backward_pass(
                attn_output, hidden_states, grad_attn_output, grad_hidden_states, pos, attn_mask
            )

        grad_hidden_states = torch.cat([grad_attn_output, grad_hidden_states], dim=-1)

        # the position encodings don't need embeddings
        return grad_hidden_states, None, None, None


class ReversibleTransformerEncoderLayer(nn.Module):

    def __init__(self, opt, death_rate=0.0):
        super().__init__()
        self.self_attn = RelativeSelfAttention(opt)
        self.feedforward = FeedForward(opt)
        self.death_rate = death_rate
        self.forward_coin = True

    def _init_attention_seed(self, *args):
        """
            This function sets a new seed for the
            attention layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """

        self.attn_cpu_state = torch.get_rng_state()
        self.attn_gpu_devices, self.attn_gpu_states = get_device_states(*args)

    def _init_feedforward_seed(self, *args):
        """
                    This function sets a new seed for the
                    feed forward layer to make dropout deterministic
                    for both forward calls: 1 normal forward
                    call and 1 forward call in backward
                    to recalculate activations.
                """

        self.ffn_cpu_state = torch.get_rng_state()
        self.ffn_gpu_devices, self.ffn_gpu_states = get_device_states(*args)

    def forward(self, x1, x2, pos, attn_mask=None):
        """
        :param pos: position embeddings
        :param x2:
        :param x1:
        :param attn_mask:
        :return:
        """
        with torch.no_grad():
            # every forward pass we sample a different seed
            # for dropout and save for forward fn in backward pass
            # to have correct dropout

            self._init_attention_seed(x2)
            z1, _, _ = self.self_attn(x2, pos, attn_mask, cleaning=True)

            y1 = z1 + x1

            self._init_feedforward_seed(y1)
            z2 = self.feedforward(y1, cleaning=True)

            y2 = z2 + x2

            del x1, x2, z1, z2

        """return Y1 and Y2"""
        return y1, y2

    def backward_pass(self, y1, y2, dy1, dy2, pos, attn_mask=None):
        """
        :param pos:
        :param y1:
        :param y2:
        :param dy1:
        :param dy2:
        :param attn_mask:
        :return:
        """
        """Implementation of the backward pass for reversible transformer encoder"""

        with torch.enable_grad():
            y1.requires_grad = True
            with torch.random.fork_rng(devices=self.ffn_gpu_devices, enabled=True):
                torch.set_rng_state(self.ffn_cpu_state)
                set_device_states(self.ffn_gpu_devices, self.ffn_gpu_states)

                z2 = self.feedforward(y1)

            # res_hidden_states.backward(grad_hidden_states, retain_graph=True)
            torch.autograd.backward(z2, dy2)

        with torch.no_grad():
            # restore X2 = Y2 - G(Y1)
            x2 = y2 - z2
            del z2, y2

            # DX1 = DY1 + Y1.grad
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True

            with torch.random.fork_rng(devices=self.attn_gpu_devices, enabled=True):
                torch.set_rng_state(self.attn_cpu_state)
                set_device_states(self.attn_gpu_devices, self.attn_gpu_states)

                z1, _, _ = self.self_attn(x2, pos, attn_mask)

            z1.backward(dx1)

        with torch.no_grad():
            # restore X1 = Y1 - F(X2)
            x1 = y1 - z1
            del y1, z1

            dx2 = dy2 + x2.grad
            x2.grad = None
            del dy2
            x2 = x2.detach()

        return x1, x2, dx1, dx2


class ReversibleDecoderFunction(Function):

    @staticmethod
    def forward(ctx, hidden_states, pos, context, layers, tgt_mask, src_mask,
                incremental=False, incremental_cache=None):

        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        B = bsz * seq_len
        idx = 0
        attn_output, hidden_states = torch.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            idx = idx + 1
            # forward pass in the layer
            attn_output, hidden_states, coverage, incremental_cache = layer(
                attn_output, hidden_states, pos, context, tgt_mask, src_mask,
                incremental=incremental, incremental_cache=incremental_cache
            )

        # attach params to ctx for backward

        # why should we detach here? because Y1 Y2 were built within torch.no_grad()
        # so cutting the backward from these variables seems unnecessary

        # save_for_backward will release memory more efficiently
        # detach() seems to be required especially for context ...
        ctx.save_for_backward(attn_output, hidden_states, context, pos)
        ctx.layers = layers
        ctx.src_mask = src_mask
        ctx.tgt_mask = tgt_mask

        with torch.no_grad():
            output = attn_output + hidden_states

        # concatenate 2 revnet outputs:
        return output

    @staticmethod
    def backward(ctx, grad_hidden_states):
        # We need three arguments because the forward pass returned 3 arguments
        # grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)
        grad_attn_output = grad_hidden_states

        # retrieve params from ctx
        attn_output, hidden_states, context, pos = ctx.saved_tensors
        layers = ctx.layers
        src_mask = ctx.src_mask
        tgt_mask = ctx.tgt_mask
        grad_context = None  # we need to sum up the gradients of the context manually

        for idx, layer in enumerate(layers[::-1]):
            # backprop
            """Note: Here for each layer we detach the context once because we need to consider it
            as a separate variable and then later accumulate the gradients"""
            attn_output, hidden_states, grad_attn_output, grad_hidden_states, grad_context_ = layer.backward_pass(
                attn_output, hidden_states, grad_attn_output, grad_hidden_states,
                pos, context.detach(), tgt_mask, src_mask
            )

            # with torch.no_grad():
            if grad_context is None:
                grad_context = grad_context_
            elif grad_context_ is not None:  # prevent ignoring layer making this None
                grad_context.add_(grad_context_)
                del grad_context_

        grad_hidden_states = torch.cat([grad_attn_output, grad_hidden_states], dim=-1)
        return grad_hidden_states, None, grad_context, None, None, None, None, None


class ReversibleTransformerDecoderLayer(nn.Module):

    # def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, ignore_source=False,
    #              variational=False, death_rate=0.0):
    def __init__(self, opt, death_rate=0.0):
        super(ReversibleTransformerDecoderLayer, self).__init__()

        self.ignore_source = opt.ignore_source
        assert not self.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.dropout = opt.dropout

        self.self_attention = RelativeSelfAttention(opt)
        self.feed_forward = FeedForward(opt)
        if not self.ignore_source:
            self.src_attention = SourceAttention(opt)

    def _init_src_attention_seed(self, *args):
        """
            This function sets a new seed for the
            attention layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """

        self.src_attn_cpu_state = torch.get_rng_state()
        self.src_attn_gpu_devices, self.src_attn_gpu_states = get_device_states(*args)

    def _init_attention_seed(self, *args):
        """
            This function sets a new seed for the
            attention layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """

        # randomize seeds
        self.attn_cpu_state = torch.get_rng_state()
        self.attn_gpu_devices, self.attn_gpu_states = get_device_states(*args)

    def _init_feedforward_seed(self, *args):
        """
                    This function sets a new seed for the
                    feed forward layer to make dropout deterministic
                    for both forward calls: 1 normal forward
                    call and 1 forward call in backward
                    to recalculate activations.
                """

        # randomize seeds
        self.ffn_cpu_state = torch.get_rng_state()
        self.ffn_gpu_devices, self.ffn_gpu_states = get_device_states(*args)

    def forward(self, x1, x2, pos, context, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True):
        """
        :param pos:
        :param x1: X1
        :param x2: X2
        :param context:
        :param mask_tgt:
        :param mask_src:
        :param incremental:
        :param incremental_cache:
        :param reuse_source:
        :return:
        """
        # if self.training:
        #     coin = (torch.rand(1)[0].item() >= self.death_rate)
        #
        # self.forward_coin = coin

        with torch.no_grad():
            # prepare the state for the first function (att > src->att)
            self._init_attention_seed(x2)
            f_x2, coverage, incremental_cache = self.self_attention(x2, pos, mask_tgt,
                                                                    incremental=incremental,
                                                                    incremental_cache=incremental_cache,
                                                                    cleaning=True)

            z = f_x2
            # print("self_attention", z.sum() / (z.size(0) * z.size(1)))
            # if not self.ignore_source:
            f_x2, coverage, incremental_cache = self.src_attention(f_x2, context, mask_src,
                                                                   incremental=incremental,
                                                                   incremental_cache=incremental_cache,
                                                                   cleaning=True)

            # print("src_attention", f_x2.sum() / (f_x2.size(0) * f_x2.size(1)))
            f_x2 = f_x2 + z
            del z

            # if self.training and self.death_rate > 0:
            #     f_x2 = f_x2 / (1 - self.death_rate)

            y1 = x1 + f_x2
            # del f_x2, x1

            # prepare the state for the second function
            self._init_feedforward_seed(y1)
            # print("y1", y1.sum() / (y1.size(0) * y1.size(1)))
            g_y1 = self.feed_forward(y1, cleaning=True)

            # if self.training and self.death_rate > 0:
            #     g_y1 = g_y1 / (1 - self.death_rate)

            y2 = x2 + g_y1

            # print("y2", y2.sum() / (y2.size(0) * y2.size(1)))

            del g_y1, x2

        """return Y1 and Y2"""
        return y1, y2, coverage, incremental_cache

    def backward_pass(self, y1, y2, dy1, dy2, pos, context,
                      mask_tgt, mask_src,
                      incremental=False, incremental_cache=None, reuse_source=False):
        """
        :param pos:
        :param y1
        :param y2
        :param dy1: dL/dX2
        :param dy2: dL/dY2
        :param context:
        :param mask_tgt:
        :param mask_src:
        :param incremental:
        :param incremental_cache:
        :param reuse_source:
        :return:
        """

        # if not self.forward_coin:  # this layer was skipped, just return
        #     return y1, y2, dy1, dy2, None

        # first block: recompute the ffn transition function
        with torch.enable_grad():
            y1.requires_grad = True

            with torch.random.fork_rng(devices=self.ffn_gpu_devices, enabled=True):
                torch.set_rng_state(self.ffn_cpu_state)
                set_device_states(self.ffn_gpu_devices, self.ffn_gpu_states)

                g_y1 = self.feed_forward(y1)

            torch.autograd.backward(g_y1, dy2)

        with torch.no_grad():
            # restore X2 = Y2 - G(Y1)
            x2 = y2 - g_y1

            # DX1 = DY1 + Y1.grad
            dx1 = dy1 + y1.grad
            del y2, g_y1, dy1
            y1.grad = None

        # second block
        with torch.enable_grad():
            x2.requires_grad = True
            context.requires_grad = True

            with torch.random.fork_rng(devices=self.attn_gpu_devices, enabled=True):
                torch.set_rng_state(self.attn_cpu_state)
                set_device_states(self.attn_gpu_devices, self.attn_gpu_states)

                f_x2, coverage, incremental_cache = self.self_attention(x2, pos, mask_tgt,
                                                                        incremental=incremental,
                                                                        incremental_cache=incremental_cache)

                z = f_x2

                # if not self.ignore_source:
                f_x2, _, _ = self.src_attention(f_x2, context, mask_src,
                                                incremental=incremental,
                                                incremental_cache=incremental_cache)

                f_x2 = f_x2 + z

            torch.autograd.backward(f_x2, dx1)

        with torch.no_grad():
            # restore X1 = Y1 - F(X2)
            x1 = y1 - f_x2
            del y1, f_x2

            dx2 = dy2 + x2.grad
            x2.grad = None
            del dy2
            x2 = x2.detach()
            grad_context = context.grad
            del context.grad

        # # third block
        # with torch.enable_grad():
        #     x2.requires_grad = True
        #
        #     with torch.random.fork_rng(devices=self.attn_gpu_devices, enabled=True):
        #         torch.set_rng_state(self.attn_cpu_state)
        #         set_device_states(self.attn_gpu_devices, self.attn_gpu_states)
        #
        #         f_x2, _, _ = self.self_attention(x2, mask_tgt)
        #
        #         if self.training and self.death_rate > 0:
        #             f_x2 = f_x2 / (1 - self.death_rate)
        #
        #     torch.autograd.backward(f_x2, dz1)
        #
        # with torch.no_grad():
        #     # restore X1 = Y1 - F(X2)
        #     x1 = z1 - f_x2
        #
        #     dx1 = dz1
        #     dx2 = dy2 + x2.grad
        #     del z1, f_x2
        #
        #     x2.grad = None
        #     x2 = x2.detach()

        return x1, x2, dx1, dx2, grad_context
