import torch.nn as nn
import torch.nn.functional as F
import torch, math
import onmt



class CopyGenerator(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(CopyGenerator, self).__init__()

        self.linear = nn.Linear(hidden_size, output_size)

        stdv = 1. / math.sqrt(self.linear.weight.size(1))

        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)

        self.linear.bias.data.zero_()

        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, net_output, log_softmax=True):

        input = net_output['hiddens'] # (len_tgt x B x H) or (LxB x H)
        src = net_output['src'] # T x B
        attn = net_output['coverage'] # len_tgt x B x len_src or (LxB x len_src)

        # 3 dimensional for T x B x H format
        # 2 dimensional for ( T x B ) x H format or the tensor with pads cleaned
        n_input_dim = input.dim()

        # note:
        # during training, we use masking to reduce computation
        # so we have to clean mask from attn and src as well
        # during testing, the input has 3 dimensions
        if n_input_dim == 3:
            if src.dim() == 2:
                attn = attn.transpose(0, 1)
                src = src.t().unsqueeze(0).expand_as(attn)

        # added float to the end
        # print(input.size())
        logits = self.linear(input).float() # len_tgt x B x H

        gate = torch.sigmoid(self.gate(input).float()) # len_tgt x B x 1

        p_g = F.softmax(logits, dim=-1)

        # p_g = p_g.mul(1 - gate)
        p_g = torch.mul(p_g, 1 - gate) # len_tgt x B x H

        # p_g.scatter_add_(1, src.t().repeat(tlen, 1), p_c)
        # print(attn.size(), gate.size())
        p_c = torch.mul(attn, gate) # len_tgt x B x len_src


        # add up the probabilities into p_g
        p_g.scatter_add_(n_input_dim-1, src, p_c)

        output = p_g.clamp(min=1e-8)

        if log_softmax:
            output = torch.log(output)

        return output