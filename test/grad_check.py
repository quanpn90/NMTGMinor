import torch.nn as nn
import onmt
import torch

from onmt.modules.optimized.relative_self_attention_func import RelativeShiftFunction


class TestDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.function = RelativeShiftFunction.apply

    def forward(self, input):

        return self.function(input)


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

    bsz = 3
    seq_len = 5
    len_q = 2

    x = torch.arange(seq_len - 1, -1, -1).unsqueeze(0).unsqueeze(0)
    x = x.expand(bsz, len_q, seq_len)

    print(x)

    input_states = torch.randn(*(bsz, len_q, seq_len)).double().cuda()

    net = TestDecoder()

    net = net.double().cuda()
    print(net)
    x = x.double().cuda()
    print(net(x))

    print("start gradchecking  ...")
    input_states.requires_grad = True
    torch.autograd.gradcheck(net, (input_states))

    print("gradchecking completed.")


        # context.requires_grad = True
        # input.requires
        # print("start gradchecking  for context...")
        # input_states.requires_grad = True
        # torch.autograd.gradcheck(net, (input_states, context))
        # print("gradchecking completed.")
