import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from onmt.modules.optimized.swish import FastSwish
from apex import amp

try:
    import dynamic_conv_cuda
except Exception as e:
    dynamic_conv_cuda = None


class DynamicConvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        if dynamic_conv_cuda is None:
            print("Dynamic Convolution is not compiled for CUDA yet.")
            raise NotImplementedError
        outputs = dynamic_conv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = dynamic_conv_cuda.backward(
                grad_output.contiguous(),
                ctx.padding_l,
                *ctx.saved_tensors)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


@amp.half_function
def dynamic_convolution(input, weights, padding_l):

    return DynamicConvFunction.apply(input, weights, padding_l)


class DynamicConvolution(torch.nn.Module):

    def __init__(self, input_size,  kernel_size=1,
            weight_softmax=False,
            num_heads=1,
            weight_dropout=0.,
            bias=True,
            renorm_padding=False,
            query_size=None,

    ):
        super(DynamicConvolution, self).__init__()

        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        # self.padding_l = padding_l
        self.padding_l = kernel_size // 2 if kernel_size % 2 == 1 \
            else ((kernel_size - 1) // 2, kernel_size // 2)

        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout
        self.renorm_padding = renorm_padding
        self.bias = bias

        self.input_linear = nn.Linear(input_size, input_size * 2, bias)
        self.output_linear = nn.Linear(input_size, input_size, bias)

        self.weight_linear = nn.Linear(input_size, num_heads * kernel_size, bias)
        if bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    def reset_parameters(self, init='normal'):

        if init == 'normal':
            nn.init.xavier_normal_(self.input_linear.weight)
            nn.init.xavier_normal_(self.weight_linear.weight)
            nn.init.xavier_normal_(self.output_linear.weight)
        else:
            nn.init.xavier_uniform_(self.input_linear.weight)
            nn.init.xavier_uniform_(self.weight_linear.weight)
            nn.init.xavier_uniform_(self.output_linear.weight)

        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)
            nn.init.constant_(self.weight_linear.bias, 0.)
            nn.init.constant_(self.input_linear.bias, 0.)
            nn.init.constant_(self.output_linear.bias, 0.)

    def forward(self, x, pad_mask=None):

        x = F.glu(self.input_linear(x))

        T, B, C = x.size()

        K, H = self.kernel_size, self.num_heads

        weight = self.weight_linear(x).view(T, B, H, K)

        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = F.dropout(weight, p=self.weight_dropout, training=self.training)

        # [seq_len x batch_size x heads x kernel_size] -> [batch_size x heads x kernel_size x seq_len]
        weight = weight.permute(1, 2, 3, 0).contiguous()

        if pad_mask is not None:
            x = x.masked_fill(pad_mask, 0)

        x = x.permute(1, 2, 0).contiguous()
        x = dynamic_convolution(x, weight, self.padding_l).permute(2, 0, 1)

        if self.conv_bias is not None:
            x = x + self.conv_bias.view(1, 1, -1)

        x = self.output_linear(x)

        return x


class Conv2dSubsampling(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.0):
        """
        :param input_dim: the log mel feature (normally 40)
        :param output_dim: network size (512)
        :param dropout: dropout rate
        """

        super(Conv2dSubsampling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # first conv is nn.Conv2d(1, output_dim, 3, 2)
        # secnd conv is nn.Conv2d(output_dim, odin, 3, 2)
        self.in_conv_weight = nn.Parameter(torch.Tensor(output_dim, 1, 3, 3))
        self.in_conv_bias = nn.Parameter(torch.Tensor(output_dim))
        self.in_stride = 2

        self.out_conv_weight = nn.Parameter(torch.Tensor(output_dim, output_dim, 3, 3))
        self.out_conv_bias = nn.Parameter(torch.Tensor(output_dim))
        self.out_stride = 2

        cnn_feature_size = output_dim * (((input_dim - 1) // 2 - 1) // 2)
        self.out_weight = nn.Parameter(torch.Tensor(output_dim, cnn_feature_size))
        self.out_bias = nn.Parameter(torch.Tensor(output_dim))

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.in_conv_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_conv_weight, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.in_conv_weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.in_conv_bias, -bound, bound)

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.out_conv_weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.out_conv_bias, -bound, bound)

        std_ = math.sqrt(2.0 / (self.output_dim + self.output_dim))
        nn.init.normal_(self.out_weight, 0.0, std_)
        nn.init.constant_(self.out_bias, 0.)

        return

    def forward(self, input, input_mask):
        """
        :param input: [bsz x seq_len x input_size]
        :param input_mask: [bsz x seq_len]
        :return:
        """

        input = input.unsqueeze(1)  # [bsz x 1 x seq_len x input_size]

        # padding = 0, dilation = 1, groups = 1
        input = F.conv2d(input, self.in_conv_weight, self.in_conv_bias, self.in_stride, 0, 1, 1)
        input = F.relu(input)

        input = F.conv2d(input, self.out_conv_weight, self.out_conv_bias, self.out_stride, 0, 1, 1)
        input = F.relu(input)

        b, c, t, f = input.size()
        input = input.transpose(1, 2).contiguous().view(b, t, c * f)

        input = F.linear(input, self.out_weight, self.out_bias)
        # input = F.dropout(input, p=self.dropout, training=self.training)

        mask = input_mask[:, :-2:2][:, :-2:2]

        return input, mask


class ConformerConvBlock(nn.Module):

    def __init__(self, channels, kernel_size, activation="relu", bias=True):
        super(ConformerConvBlock, self).__init__()

        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(channels, 2*channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=channels, bias=bias)

        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = FastSwish()

        # self.in_pointwise_weight = nn.Conv1d(channels, 2*channels, kernel_size=1, stride=1, padding=0, bias=False)
        # self.in_pointwise_bias = nn.Parameter(torch.Tensor(2 * channels))
        #
        # self.depthwise_weight = nn.Parameter(torch.Tensor(channels, channels // channels, kernel_size))
        # self.depthwise_bias = nn.Parameter(torch.Tensor(channels))
        # self.padding = (kernel_size - 1) // 2
        # self.groups = channels
        #

        # self.out_pointwise_weight = nn.Parameter(torch.Tensor(channels, channels, 1))
        # self.out_pointwise_bias = nn.Parameter(torch.Tensor(channels))
        #
        # self.activation = activation
        self.reset_parameters()

    def reset_parameters(self, init='normal'):

        if init == 'normal':
            nn.init.kaiming_normal_(self.pointwise_conv1.weight)
            nn.init.kaiming_normal_(self.depthwise_conv.weight)
            nn.init.kaiming_normal_(self.pointwise_conv2.weight)
        else:
            nn.init.kaiming_uniform_(self.pointwise_conv1.weight)
            nn.init.kaiming_uniform_(self.depthwise_conv.weight)
            nn.init.kaiming_uniform_(self.pointwise_conv2.weight)

        nn.init.constant_(self.pointwise_conv1.bias, 0)
        nn.init.constant_(self.pointwise_conv2.bias, 0)
        nn.init.constant_(self.depthwise_conv.bias, 0)
    #     nn.init.kaiming_uniform_(self.in_pointwise_weight, a=math.sqrt(5))
    #     nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
    #     nn.init.kaiming_uniform_(self.out_pointwise_weight, a=math.sqrt(5))
    #
    #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.in_pointwise_weight)
    #     bound = 1 / math.sqrt(fan_in)
    #     init.uniform_(self.in_pointwise_bias, -bound, bound)
    #
    #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.depthwise_weight)
    #     bound = 1 / math.sqrt(fan_in)
    #     init.uniform_(self.depthwise_bias, -bound, bound)
    #
    #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.out_pointwise_weight)
    #     bound = 1 / math.sqrt(fan_in)
    #     init.uniform_(self.out_pointwise_bias, -bound, bound)

    def forward(self, x):
        """
        :param x: [seq_len x bsz x hidden_size]
        :return:
        """

        x = x.transpose(0, 1).transpose(1, 2)  # to [bsz x hidden_size x seq_len]

        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        # x = F.conv1d(x, self.in_pointwise_weight, self.in_pointwise_bias, 1, 0, 1, 1)
        # x = F.glu(x, dim=1)
        #
        # x = F.conv1d(x, self.depthwise_weight, self.depthwise_bias, 1, self.padding, 1, self.groups)
        # x = self.activation(x)
        #
        # x = F.conv1d(x, self.out_pointwise_weight, self.out_pointwise_bias, 1, 0, 1, 1)

        x = x.transpose(1, 2).transpose(0, 1)  # back to [seq_len x bsz x hidden_size]

        return x


if __name__ == "__main__":

    bsz = 160
    seq_len = 1000
    input_size = 48
    output_size = 128
    kernel = 31

    subsampler = Conv2dSubsampling(input_size, output_size)
    subsampler = subsampler.cuda()

    conv = ConformerConvBlock(output_size, kernel)
    conv = conv.cuda()

    input = torch.randn(seq_len, bsz, input_size)
    mask = torch.randn(bsz, seq_len)

    input = input.cuda()
    mask = mask.cuda()

    input, mask = subsampler(input.transpose(0, 1), mask)
    print(input.size())
    print(mask.size())

    output = conv(input.transpose(0, 1))
    print(output.size())



