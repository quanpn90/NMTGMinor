import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.linear import FeedForward
from onmt.modules.attention import MultiHeadAttention
from torch.autograd.function import Function
import sys
from torch.utils.checkpoint import get_device_states, set_device_states


def deterministic_dropout(input, p=0.5, training=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return nn.functional.dropout(input, p=p, training=training)


class ReversibleEncoderFunction(Function):

    @staticmethod
    def forward(ctx, hidden_states, layers, attn_mask):

        attn_output, hidden_states = torch.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            # forward pass in the layer
            attn_output, hidden_states = layer(
                attn_output, hidden_states, attn_mask
            )

        # attach params to ctx for backward

        # why should we detach here? because Y1 Y2 were built within torch.no_grad()
        # so cutting the backward from these variables seems unnecessary

        # save_for_backward will release memory more efficiently
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
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
        attn_output, hidden_states = ctx.saved_tensors
        layers = ctx.layers
        attn_mask = ctx.attn_mask

        for idx, layer in enumerate(layers[::-1]):
            # backprop
            attn_output, hidden_states, grad_attn_output, grad_hidden_states = layer.backward_pass(
                attn_output, hidden_states, grad_attn_output, grad_hidden_states, attn_mask
            )

        grad_hidden_states = torch.cat([grad_attn_output, grad_hidden_states], dim=-1)

        return grad_hidden_states, None, None


class ReversibleTransformerEncoderLayer(nn.Module):

    def __init__(self, opt, death_rate=0.0):

        self.variational = opt.variational_dropout
        d_model = opt.model_size
        p = opt.dropout
        self.death_rate = death_rate
        self.dropout = p
        h = opt.n_heads
        attn_p = opt.attn_dropout

        super().__init__()
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, share=2)

        ff_p = opt.dropout
        self.feedforward = FeedForward(opt.model_size, opt.inner_size, ff_p, variational=self.variational)

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

    def forward(self, prev_attn_output, hidden_states, attn_mask=None):
        """
        :param prev_attn_output: X1
        :param hidden_states:    X2
        :param attn_mask:
        :return:
        """
        with torch.no_grad():
            # every forward pass we sample a different seed
            # for dropout and save for forward fn in backward pass
            # to have correct dropout

            coin = True
            if self.training:
                if self.training:
                    coin = (torch.rand(1)[0].item() >= self.death_rate)

            self.forward_coin = coin

            if coin:
                self._init_attention_seed(hidden_states)
                query = self.preprocess_attn(hidden_states)
                attn_output, _, _ = self.multihead(query, query, query, attn_mask)

                if self.training and self.death_rate > 0:
                    attn_output = attn_output / (1 - self.death_rate)

                # before dropout: add a seed
                attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

                # Y_1 = X_1 + F(X_2)
                attn_output = attn_output + prev_attn_output
                # free memory:
                del prev_attn_output, query

                self._init_feedforward_seed(attn_output)
                ffn_input = self.preprocess_ffn(attn_output)
                ffn_output = self.feedforward(ffn_input)
                if self.training and self.death_rate > 0:
                    ffn_output = ffn_output / (1 - self.death_rate)

                # dropout
                ffn_output = F.dropout(ffn_output, p=self.dropout, training=self.training)

                # Y_2 = X_2 + F(Y_1)
                hidden_states = hidden_states + ffn_output

                del ffn_input, ffn_output

            else:
                hidden_states = hidden_states
                attn_output = prev_attn_output

        """return Y1 and Y2"""
        return attn_output, hidden_states

    def backward_pass(self, attn_output, hidden_states, grad_attn_output, grad_hidden_states, attn_mask=None):
        """
        :param attn_output: Y1
        :param hidden_states: Y2
        :param grad_attn_output: dL/dY1
        :param grad_hidden_states: dL/dY2
        :param attn_mask:
        :return:
        """
        """Implementation of the backward pass for reversible transformer encoder"""

        if not self.forward_coin:  # this layer was skipped, just return
            return attn_output, hidden_states, grad_attn_output, grad_hidden_states

        with torch.enable_grad():
            attn_output.requires_grad = True
            with torch.random.fork_rng(devices=self.ffn_gpu_devices, enabled=True):
                torch.set_rng_state(self.ffn_cpu_state)
                set_device_states(self.ffn_gpu_devices, self.ffn_gpu_states)
                res_hidden_states = self.feedforward(self.preprocess_ffn(attn_output))
                if self.training and self.death_rate > 0:
                    res_hidden_states = res_hidden_states / (1 - self.death_rate)
                res_hidden_states = F.dropout(res_hidden_states, p=self.dropout,
                                                                training=self.training)
            # res_hidden_states.backward(grad_hidden_states, retain_graph=True)
            torch.autograd.backward(res_hidden_states, grad_hidden_states)

        with torch.no_grad():
            # restore X2 = Y2 - G(Y1)
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states

            # DX1 = DY1 + Y1.grad
            grad_attn_output = grad_attn_output + attn_output.grad
            attn_output.grad = None

        with torch.enable_grad():

            hidden_states.requires_grad = True

            with torch.random.fork_rng(devices=self.attn_gpu_devices, enabled=True):
                torch.set_rng_state(self.attn_cpu_state)
                set_device_states(self.attn_gpu_devices, self.attn_gpu_states)
                res_attn_output = self.preprocess_attn(hidden_states)
                # torch.manual_seed(self.attention_seed)
                # I forgot there is attention dropout in attention layer too ....
                res_attn_output, _, _ = self.multihead(res_attn_output, res_attn_output, res_attn_output, attn_mask)
                if self.training and self.death_rate > 0:
                    res_attn_output = res_attn_output / (1 - self.death_rate)
                res_attn_output = F.dropout(res_attn_output, p=self.dropout,
                                                              training=self.training)

            res_attn_output.backward(grad_attn_output)
            # torch.autograd.backward(res_attn_output, grad_attn_output)

        with torch.no_grad():
            # restore X1 = Y1 - F(X2)
            attn_output = attn_output - res_attn_output
            del res_attn_output

            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None
            hidden_states = hidden_states.detach()

        return attn_output, hidden_states, grad_attn_output, grad_hidden_states


class ReversibleDecoderFunction(Function):

    @staticmethod
    def forward(ctx, hidden_states, context, layers, tgt_mask, src_mask,
                incremental=False, incremental_cache=None):

        attn_output, hidden_states = torch.chunk(hidden_states, 2, dim=-1)

        for layer in layers:
            # forward pass in the layer
            attn_output, hidden_states, coverage, incremental_cache = layer(
                attn_output, hidden_states, context, tgt_mask, src_mask,
                incremental=incremental, incremental_cache=incremental_cache
            )

        # attach params to ctx for backward

        # why should we detach here? because Y1 Y2 were built within torch.no_grad()
        # so cutting the backward from these variables seems unnecessary

        # save_for_backward will release memory more efficiently
        # detach() seems to be required especially for context ...
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach(), context)
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
        attn_output, hidden_states, context = ctx.saved_tensors
        layers = ctx.layers
        src_mask = ctx.src_mask
        tgt_mask = ctx.tgt_mask
        grad_context = None  # we need to sum up the gradients of the context manually

        for idx, layer in enumerate(layers[::-1]):
            # backprop
            # print(idx, grad_attn_output.shape, grad_hidden_states.shape)
            attn_output, hidden_states, grad_attn_output, grad_hidden_states, grad_context_ = layer.backward_pass(
                attn_output, hidden_states, grad_attn_output, grad_hidden_states,
                context.detach(), tgt_mask, src_mask
            )

            # with torch.no_grad():
            if grad_context is None:
                grad_context = grad_context_
            elif grad_context_ is not None:  # prevent ignoring layer making this None
                grad_context += grad_context_
                del grad_context_

        grad_hidden_states = torch.cat([grad_attn_output, grad_hidden_states], dim=-1)
        return grad_hidden_states, grad_context, None, None, None, None, None


class ReversibleTransformerDecoderLayer(nn.Module):

    # def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, ignore_source=False,
    #              variational=False, death_rate=0.0):
    def __init__(self, opt, death_rate=0.0):
        super(ReversibleTransformerDecoderLayer, self).__init__()
        d_model = opt.model_size

        self.ignore_source = opt.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.dropout = opt.dropout

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                          variational=self.variational)
            self.multihead_src = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=2)

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)

        self.multihead_tgt = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=1)

        # if onmt.constants.activation_layer == 'linear_relu_linear':
        ff_p = opt.dropout
        feedforward = FeedForward(opt.model_size, opt.inner_size, ff_p, variational=self.variational)
        # elif onmt.constants.activation_layer == 'maxout':
        #     k = int(math.ceil(d_ff / opt.model))
        #     feedforward = MaxOut(d_model, d_model, k)
        # elif onmt.constants.activation_layer == 'linear_swish_linear':
        #     ff_p = opt.dropout
        #     feedforward = FeedForwardSwish(d_model, opt.inner_size, ff_p)
        # else:
        #     raise NotImplementedError

        self.feedforward = feedforward

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

    def forward(self, prev_attn_output, hidden_states, context, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True):
        """
        :param prev_attn_output: X1
        :param hidden_states: X2
        :param context:
        :param mask_tgt:
        :param mask_src:
        :param incremental:
        :param incremental_cache:
        :param reuse_source:
        :return:
        """
        coverage = None
        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        self.forward_coin = coin

        if coin:
            with torch.no_grad():

                # prepare the state for the first function (att > src->att)
                self._init_attention_seed(hidden_states)
                attn_output = self.preprocess_attn(hidden_states)
                attn_output, _, incremental_cache = self.multihead_tgt(attn_output, attn_output, attn_output, mask_tgt,
                                                                       incremental=incremental,
                                                                       incremental_cache=incremental_cache)

                # attn_output = F.relu(attn_output, inplace=True)
                attn_output = F.dropout(attn_output, p=self.dropout)
                self_attn = attn_output

                if not self.ignore_source:
                    attn_output = self.preprocess_src_attn(attn_output)
                    attn_output, coverage, incremental_cache = self.multihead_src(attn_output, context, context, mask_src,
                                                                                  incremental=incremental,
                                                                                  incremental_cache=incremental_cache)

                    # attn_output = F.relu(attn_output, inplace=True)
                    attn_output = nn.functional.dropout(attn_output, p=self.dropout)
                    # this is not just residual
                    # attn_output only contains information about the source
                    attn_output = attn_output + self_attn

                del self_attn

                if self.training and self.death_rate > 0:
                    attn_output = attn_output / (1 - self.death_rate)

                # Y1 = F(X2) + X1
                attn_output = attn_output + prev_attn_output

                # free memory:
                del prev_attn_output

                # prepare the state for the second function
                self._init_feedforward_seed(attn_output)
                ffn_input = self.preprocess_ffn(attn_output)
                ffn_output = self.feedforward(ffn_input)

                # dropout
                ffn_output = torch.nn.functional.dropout(ffn_output, p=self.dropout, training=self.training)

                # scale before
                if self.training and self.death_rate > 0:
                    ffn_output = ffn_output / (1 - self.death_rate)

                hidden_states = hidden_states + ffn_output

                del ffn_input, ffn_output

        else:
            hidden_states = hidden_states
            attn_output = prev_attn_output

        """return Y1 and Y2"""
        return attn_output, hidden_states, coverage, incremental_cache

    def backward_pass(self, attn_output, hidden_states, grad_attn_output, grad_hidden_states, context,
                      mask_tgt, mask_src,
                      incremental=False, incremental_cache=None, reuse_source=False):
        """
        :param attn_output: X2
        :param hidden_states: Y2
        :param grad_attn_output: dL/dX2
        :param grad_hidden_states: dL/dY2
        :param context:
        :param mask_tgt:
        :param mask_src:
        :param incremental:
        :param incremental_cache:
        :param reuse_source:
        :return:
        """

        if not self.forward_coin:  # this layer was skipped, just return
            return attn_output, hidden_states, grad_attn_output, grad_hidden_states, None

        # first block: recompute the ffn transition function
        with torch.enable_grad():
            attn_output.requires_grad = True

            with torch.random.fork_rng(devices=self.ffn_gpu_devices, enabled=True):
                torch.set_rng_state(self.ffn_cpu_state)
                set_device_states(self.ffn_gpu_devices, self.ffn_gpu_states)

                res_hidden_states = self.feedforward(self.preprocess_ffn(attn_output))
                if self.training and self.death_rate > 0:
                    res_hidden_states = res_hidden_states / (1 - self.death_rate)
                res_hidden_states = torch.nn.functional.dropout(res_hidden_states, p=self.dropout,
                                                                training=self.training)
            # res_hidden_states.backward(grad_hidden_states, retain_graph=True)
            torch.autograd.backward(res_hidden_states, grad_hidden_states)

        with torch.no_grad():
            # restore X2 = Y2 - G(Y1)
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states

            grad_attn_output = grad_attn_output + attn_output.grad
            attn_output.grad = None

        # second block
        with torch.enable_grad():

            hidden_states.requires_grad = True
            context.requires_grad = True

            with torch.random.fork_rng(devices=self.attn_gpu_devices, enabled=True):
                torch.set_rng_state(self.attn_cpu_state)
                set_device_states(self.attn_gpu_devices, self.attn_gpu_states)

                res_attn_output = self.preprocess_attn(hidden_states)
                res_attn_output, _, incremental_cache = self.multihead_tgt(res_attn_output, res_attn_output,
                                                                           res_attn_output, mask_tgt,
                                                                           incremental=incremental,
                                                                           incremental_cache=incremental_cache)

                # ignore the residual connection here
                # res_attn_output = F.relu(res_attn_output, inplace=True)
                res_attn_output = nn.functional.dropout(res_attn_output, p=self.dropout)
                self_attn = res_attn_output

                if not self.ignore_source:
                    assert incremental is False, "Incremental is not allowed in the backward pass"
                    res_attn_output = self.preprocess_src_attn(res_attn_output)
                    res_attn_output, _, _ = self.multihead_src(res_attn_output, context,
                                                               context, mask_src,
                                                               incremental=incremental,
                                                               incremental_cache=incremental_cache)

                    # res_attn_output = F.relu(res_attn_output, inplace=True)
                    res_attn_output = nn.functional.dropout(res_attn_output, p=self.dropout)
                    res_attn_output = res_attn_output + self_attn

                if self.training and self.death_rate > 0:
                    res_attn_output = res_attn_output / (1 - self.death_rate)

            torch.autograd.backward(res_attn_output, grad_attn_output)

        with torch.no_grad():
            # restore X1 = Y1 - F(X2)
            attn_output = attn_output - res_attn_output
            del res_attn_output
            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None
            hidden_states = hidden_states.detach()
            grad_context = context.grad
            del context

        return attn_output, hidden_states, grad_attn_output, grad_hidden_states, grad_context


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=32,
                        help='Size of embedding / transformer hidden')

    opt = parser.parse_args()

    opt.layers = 1
    opt.variational_dropout = False
    opt.dropout = 0.0
    opt.attn_dropout = 0.0
    opt.n_heads = 1

    layers = nn.ModuleList([ReversibleTransformerEncoderLayer(opt) for _ in range(opt.layers)])

    print(layers)

