import torch.nn as nn
import onmt
import torch
from torch.nn import Parameter


from onmt.modules.optimized.relative_self_attention_func import RelativeShiftFunction


class Parameters(nn.Module):

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
        nn.init.normal_(self.in_proj_weight, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)
        nn.init.normal_(self.pos_proj_weight, 0.0, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.constant_(self.pos_proj_bias, 0.)

        nn.init.normal_(self.r_w_bias, 0.0, std_)
        nn.init.normal_(self.r_r_bias, 0.0, std_)


class TestFeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.function = feed_forward_relu

    def forward(self, input, input_weights, input_biases, output_weights, output_biases):

        training = self.training
        dropout = 0.1
        variational = True

        return self.function(input, input_weights, input_biases, output_weights, output_biases,
                             dropout, training, variational)


class TestAttention(nn.Module):

    def __init__(self, model_size=16, heads=1):
        super().__init__()
        self.model_size = model_size
        self.heads = heads
        self.head_dim = model_size // heads
        # self.function = RelativeShiftFunction.apply

        self.in_proj_weight = Parameter(torch.Tensor(3 * model_size, model_size))
        self.out_proj_weight = Parameter(torch.Tensor(model_size, model_size))
        self.pos_proj_weight = Parameter(torch.Tensor(model_size, model_size))

        self.in_proj_bias = Parameter(torch.Tensor(3 * model_size))
        self.out_proj_bias = Parameter(torch.Tensor(model_size))
        self.pos_proj_bias = Parameter(torch.Tensor(model_size))

        from onmt.modules.optimized.relative_self_attention_func import relative_self_attn_func
        self.function = relative_self_attn_func

        self.r_w_bias = nn.Parameter(torch.Tensor(self.heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.heads, self.head_dim))
        self.reset_parameters()

    def forward(self, input, pos, in_proj_weight, out_proj_weight, pos_proj_weight,
                in_proj_bias, out_proj_bias, pos_proj_bias, r_w_bias, r_r_bias):

        use_time_mask = False
        is_training = True
        mask = None
        dropout = 0.0

        return self.function(input, pos, use_time_mask, is_training, self.heads,
                             in_proj_weight, out_proj_weight, pos_proj_weight,
                             in_proj_bias, out_proj_bias, pos_proj_bias,
                             r_w_bias, r_r_bias,
                             mask, dropout, False, None, True)   # double precision set to true


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=32,
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
    opt.n_heads = 8
    opt.inner_size = 16

    bsz = 1
    seq_len = 2
    len_r = 15

    # x = torch.arange(seq_len - 1, -1, -1).unsqueeze(0).unsqueeze(0)
    # x = x.expand(bsz, seq_len, seq_len)
    #
    # print(x)
    #
    input_states = torch.randn(*(seq_len, bsz, opt.model_size)).double().cuda()
    onmt.constants.double_precision = True
    pos = torch.randn(*(len_r, 1, opt.model_size)).double().cuda()
    net = TestAttention(model_size=opt.model_size, heads=opt.n_heads)

    net = net.double().cuda()

    print("start gradchecking  ...")
    input_states.requires_grad = True

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

    torch.autograd.gradcheck(net, (input_states, pos, in_proj_weight, out_proj_weight, pos_proj_weight,
                                   in_proj_bias, out_proj_bias, pos_proj_bias, r_w_bias, r_r_bias))
    #
    print("gradchecking completed.")