import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


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

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super(ConformerConvBlock, self).__init__()

        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(channels, 2*channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=channels, bias=bias)

        # self.batch_norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

        # self.in_pointwise_weight = nn.Conv1d(channels, 2*channels, kernel_size=1, stride=1, padding=0, bias=False)
        # self.in_pointwise_bias = nn.Parameter(torch.Tensor(2 * channels))
        #
        # self.depthwise_weight = nn.Parameter(torch.Tensor(channels, channels // channels, kernel_size))
        # self.depthwise_bias = nn.Parameter(torch.Tensor(channels))
        # self.padding = (kernel_size - 1) // 2
        # self.groups = channels
        #
        # self.norm = nn.BatchNorm1d(channels)
        # self.out_pointwise_weight = nn.Parameter(torch.Tensor(channels, channels, 1))
        # self.out_pointwise_bias = nn.Parameter(torch.Tensor(channels))
        #
        # self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.kaiming_normal_(self.pointwise_conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.depthwise_conv.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise_conv2.weight, nonlinearity='relu')

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

    def forward(self, x, pad_mask=None):
        """
        :param pad_mask: [seq_len x bsz] indicating which element is correct
        (this should be the same with the attention mask (pad=1, unpad=0)
        :param x: [seq_len x bsz x hidden_size]
        :return:
        """

        x = x.transpose(0, 1).transpose(1, 2)  # to [bsz x hidden_size x seq_len]

        # pointwise conv does not need to mask because its elementwise projection
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        if pad_mask is not None:
            pad_mask = pad_mask.transpose(0, 1).transpose(1, 2)
            # print(x.size(), pad_mask.size())
            x = x.masked_fill_(pad_mask, 0)
        x = self.depthwise_conv(x)
        x = self.activation(x)

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



