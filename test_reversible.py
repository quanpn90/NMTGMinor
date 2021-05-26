#!/usr/bin/env python3
import onmt
import torch
import torch.nn as nn


if __name__ == "__main__":

    from onmt.models.multilingual_translator.reversible_transformers import reversible_encoder, \
        ReversibleTransformerEncoderLayer

    from onmt.models.multilingual_translator.reversible_transformers import reversible_decoder, \
        ReversibleTransformerDecoderLayer

    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=32,
                        help='Size of embedding / transformer hidden')
    parser.add_argument('-gpu', default=0, type=int,
                        help="Seed for deterministic runs.")

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

    opt.layers = 4
    opt.variational_dropout = False
    opt.dropout = 0.0
    opt.attn_dropout = 0.0
    opt.residual_dropout = 0.0
    opt.ffn_dropout = 0.0
    opt.n_heads = 4
    opt.inner_size = 4 * opt.model_size
    opt.ffn_glu = False
    opt.ffn_activation = 'relu'
    opt.head_dim = opt.model_size // opt.n_heads
    opt.learnable_position_encoding = False
    opt.ignore_source = False

    layers = torch.nn.ModuleList()

    for l in range(opt.layers):
        layer = ReversibleTransformerEncoderLayer(opt)
        layers.append(layer)

    class TestEncoder(torch.nn.Module):

        def __init__(self, layers):
            super().__init__()
            self.function = reversible_encoder
            self.layers = layers

        def forward(self, input, pos):

            return self.function(self.layers, input, pos, None)


    bsz = 4
    len_q = 7
    len_r = 7
    len_k = 12

    device = torch.device('cuda:0')
    input_states = torch.randn(*(len_q, bsz, opt.model_size), dtype=torch.float64, requires_grad=True, device=device)
    pos = torch.randn(*(len_q, 1, opt.model_size), dtype=torch.float64, requires_grad=False, device=device)

    net = TestEncoder(layers)
    net = net.double().cuda()

    print(net)

    # print("gradchecking ENCODER start.")
    #
    # torch.autograd.gradcheck(net, (input_states, pos), eps=1e-6, atol=1e-5, rtol=1e-3)
    #
    # print("gradchecking ENCODER completed.")

    class TestDecoder(torch.nn.Module):

        def __init__(self, layers):
            super().__init__()
            self.function = reversible_decoder
            self.layers = layers

        def forward(self, input, pos, context):

            return self.function(self.layers, input, pos, context, None, None, None, None)


    device = torch.device('cuda:0')
    input_states = torch.randn(*(len_q, bsz, opt.model_size), dtype=torch.float64, requires_grad=True, device=device)
    pos = torch.randn(*(len_q, 1, opt.model_size), dtype=torch.float64, requires_grad=False, device=device)

    context = torch.randn(*(len_k, bsz, opt.model_size), dtype=torch.float64, requires_grad=True, device=device)

    layers = torch.nn.ModuleList()

    for l in range(opt.layers):
        layer = ReversibleTransformerDecoderLayer(opt)
        layers.append(layer)

    net = TestDecoder(layers)
    net = net.double().cuda()

    print("gradchecking DECODER start.")
    torch.autograd.gradcheck(net, (input_states, pos, context), eps=1e-6, atol=1e-5, rtol=1e-3)

    print("Completed.")
    # net(input_states, pos, context)