import torch.nn as nn
import onmt
import torch

# from onmt.reversible_models.transformers import ReversibleTransformerEncoderLayer, ReversibleEncoderFunction, \
#     ReversibleTransformerDecoderLayer, ReversibleDecoderFunction
from onmt.reversible_models.relative_transformers import ReversibleTransformerEncoderLayer, ReversibleEncoderFunction, \
    ReversibleTransformerDecoderLayer, ReversibleDecoderFunction


class TestEncoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, input, pos):

        return ReversibleEncoderFunction.apply(input, pos, self.layers, None)


class TestDecoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, input, context, pos):

        return ReversibleDecoderFunction.apply(input, pos, context, self.layers,
                                               None, None, False, None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=16,
                        help='Size of embedding / transformer hidden')
    parser.add_argument('-gpu', default=0, type=int,
                        help="Seed for deterministic runs.")
    parser.add_argument('-test_decoder', action='store_true',
                        help='Test decoder')

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

    onmt.constants.weight_norm = False
    onmt.constants.checkpointing = False
    onmt.constants.max_position_length = 4096
    onmt.constants.double_precision = True

    opt.layers = 2
    opt.variational_dropout = False
    opt.dropout = 0.0
    opt.attn_dropout = 0.0
    opt.n_heads = 1
    opt.inner_size = 16

    bsz = 4
    seq_len = 16

    input_states = torch.randn(*(seq_len, bsz, opt.model_size*2)).double().cuda()
    pos = torch.randn(*(seq_len, 1, opt.model_size)).double().cuda()

    pos.requires_grad=False

    if not opt.test_decoder:
        layers = nn.ModuleList([ReversibleTransformerEncoderLayer(opt) for _ in range(opt.layers)])
        # layers.cuda()

        net = TestEncoder(layers)
        net = net.double().cuda()
        print(net)

        print("start gradchecking  ...")
        input_states.requires_grad = True
        torch.autograd.gradcheck(net, (input_states, pos))

        print("gradchecking completed.")
    else:
        print("Testing decoder ...")
        opt.ignore_source = False

        layers = nn.ModuleList([ReversibleTransformerDecoderLayer(opt) for x in range(opt.layers)])

        net = TestDecoder(layers)
        net = net.double().cuda()

        src_seq_len = 8
        context = torch.randn(*(src_seq_len, bsz, opt.model_size)).double().cuda()

        print("start gradchecking for input and context...")
        input_states.requires_grad = True
        context.requires_grad = True
        torch.autograd.gradcheck(net, (input_states, context, pos))
        print("gradchecking completed.")

        # context.requires_grad = True
        # input.requires
        # print("start gradchecking  for context...")
        # input_states.requires_grad = True
        # torch.autograd.gradcheck(net, (input_states, context))
        # print("gradchecking completed.")
