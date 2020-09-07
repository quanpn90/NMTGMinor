import torch
import torch.nn.functional as F
from onmt.modules.dropout import variational_dropout



class BatchEnsembleMM(object):

    @staticmethod
    def forward(x, weight, bias, ensemble_r, ensemble_s):
        """
        :param x: [T x B x H]
        :param weight: [H_out x H]
        :param bias: [H_out]
        :param ensemble_r: [B x H]
        :param ensemble_s: [B x H_out]
        :return:
        """
        bsz, len_x, hin = x.size(1), x.size(0), x.size(2)
        hout = weight.size(0)

        assert bsz == ensemble_s.size(0)
        # assert ensemble * bsz_per_ensemble == bsz, "Mini-batch must divide evenly to the ensembles"

        # element-wise [T x B x H] \times [B x H]
        x_r = torch.mul(x, ensemble_r)

        # GEMM No Bias. Otherwise use addmm
        x_mm = torch.mm(x_r.view(-1, hin), weight.transpose(0, 1))
        x_mm = x_mm.view(len_x, bsz, hout)

        # element-wise [T x B x Hout] \times [B x Hout]
        x_s = torch.mul(x_mm, ensemble_s)

        # add bias
        x_s = torch.add(x_s, bias)

        # we need to store the intermediate results for the backward pass
        return x_s, x_mm, x_r

    # maybe we need some allocated memory as well
    @staticmethod
    def backward(grad_y, x, x_r, x_mm, weight, ensemble_r, ensemble_s):
        bsz, len_x, hin = x.size(1), x.size(0), x.size(2)
        hout = x_mm.size(-1)

        grad_bias = grad_y
        grad_s = grad_y

        # backprop through the last element-wise multiplication
        grad_ensemble_s = torch.mul(grad_s, x_mm)
        grad_ensemble_s = torch.sum(grad_ensemble_s, dim=0)

        # backprop through the MM
        grad_mm = torch.mul(grad_s, ensemble_s)
        grad_mm = grad_mm.view(-1, hout)
        grad_r = torch.mm(grad_mm, weight).view(len_x, bsz, hin)
        # GEMM: [hout x bsz] \times [bsz x hin]
        grad_weight = torch.mm(grad_mm.transpose(0, 1), x_r.view(-1, hin))

        # back prop through the first element-wise multiplication
        # element-wise [len_x, bsz, hin] \cdot [bsz, hin]
        grad_x = torch.mul(grad_r, ensemble_r)
        # grad ensemble r
        grad_ensemble_r = torch.mul(grad_r, x)
        grad_ensemble_r = torch.sum(grad_ensemble_r, dim=0)

        return grad_x, grad_weight, grad_bias, grad_ensemble_r, grad_ensemble_s


class BatchEnsembleLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, ensemble_r, ensemble_s):

        x_s, x_mm, x_r = BatchEnsembleMM.forward(x, weight, bias, ensemble_r, ensemble_s)

        output = x_s
        ctx.save_for_backward(x, weight, bias, ensemble_r, ensemble_s, x_mm, x_r)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        x, weight, bias, ensemble_r, ensemble_s, x_mm, x_r = ctx.saved_tensors

        grad_x, grad_weight, grad_bias, grad_ensemble_r, grad_ensemble_s = \
            BatchEnsembleMM.backward(grad_output, x, x_r, x_mm, weight, ensemble_r, ensemble_s)

        return grad_x, grad_weight, grad_bias, grad_ensemble_r, grad_ensemble_s


class BatchEnsembleLinear(torch.nn.Module):

    # TODO: write gradcheck testing
    def __init__(self, input_size, output_size, ensemble):

        super().__init__()

        self.weight = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        self.r = torch.nn.Parameter(torch.Tensor(ensemble, input_size))
        self.s = torch.nn.Parameter(torch.Tensor(ensemble, output_size))

        self.reset_parameters()

    def reset_parameters(self, init='normal'):
        if init == 'normal':
            torch.nn.init.xavier_normal_(self.weight)
        else:
            torch.nn.init.xavier_uniform_(self.weight)



        # for batch ensemble we init r_i and s_i with random sign vectors
        # with torch.no_grad():
        #     self.r.bernoulli_(0.5).mul_(-2).add_(1)
        #     self.s.bernoulli_(0.5).mul_(-2).add_(1)
        torch.nn.init.normal_(self.r, 0.0, 0.02)
        torch.nn.init.normal_(self.s, 0.0, 0.02)

    def forward(self, input, indices=None):
        """
        :param input: T x B x H
        :param indices: T x B or B
        :return:
        """
        ensemble = self.r.size(0)
        bsz = input.size(1) if len(input.shape) == 3 else input.size(0)

        if indices is None:  # if indices are not None, then w
            with torch.no_grad():
                indices = torch.arange(0, bsz, device=input.device, dtype=torch.long)
                indices = torch.remainder(indices, ensemble)

        # during training, we randomly select the ensemble_id into batch size
        if self.training:
            r = torch.index_select(self.r, 0, indices)
            s = torch.index_select(self.s, 0, indices)

            if len(input.shape) == 3:
                return BatchEnsembleLinearFunction.apply(input, self.weight, self.bias, r, s)
            if len(input.shape) == 2:
                return torch.mul(F.linear(torch.mul(input, r), weight, bias), s)

        # during eval we have to repeat the dimensions ensemble times
        else:
            if len(input.shape) == 3:
                if indices is not None:
                    len_x, bsz, hin = input.size(0), input.size(1), input.size(2)
                    input = input.repeat(1, ensemble, 1)
                    # we need the transpose step to ensure that both should have ensemble x batch
                    # but should it be ensemble x batch or batch x ensemble ? ...
                    # TODO: test at decoding time. batch_size=beam_size=1 should yield the same result
                    # r = self.r.repeat(bsz, 1).view(bsz, ensemble, self.r.size(-1)).\
                    #     transpose(0, 1).contiguous().view(-1, self.r.size(-1))
                    # s = self.s.repeat(bsz, 1).view(bsz, ensemble, self.s.size(-1)).\
                    #     transpose(0, 1).contiguous().view(-1, self.s.size(-1))
                    input = input.view(len_x, ensemble, bsz, hin)
                    r = self.r.unsqueeze(1)  # ensemble x 1 x hin
                    s = self.s.unsqueeze(1)  # ensemble x 1 x hout
                    output = torch.mul(F.linear(torch.mul(input, r), self.weight, self.bias), s)
                    output = output.view(len_x, ensemble, bsz, output.size(-1))
                    # output = BatchEnsembleLinearFunction.apply(input, self.weight, self.bias, r, s)
                    # output = output.view(len_x, ensemble, bsz, -1)
                    output = torch.mean(output, dim=1)
                    return output
                else:
                    r = torch.index_select(self.r, 0, indices)
                    s = torch.index_select(self.s, 0, indices)
                    if len(input.shape) == 3:
                        return BatchEnsembleLinearFunction.apply(input, self.weight, self.bias, r, s)
                    if len(input.shape) == 2:
                        return torch.mul(F.linear(torch.mul(input, r), weight, bias), s)

            else:
                bsz, hin = input.size(0), input.size(1)
                input = input.repeat(ensemble, 1)
                r = self.r.repeat(bsz, 1).view(bsz, ensemble, self.r.size(-1)).\
                    transpose(0, 1).view(-1, self.r.size(-1))
                s = self.s.repeat(bsz, 1).view(bsz, ensemble, self.s.size(-1)).\
                    transpose(0, 1).view(-1, self.s.size(-1))
                output = torch.mul(F.linear(torch.mul(input, r), weight, bias), s)
                output = output.view(ensemble, bsz, -1)
                output = torch.mean(output, dim=0)
                return output


class BEPositionWiseFeedForward(torch.nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, model_size, inner_size, dropout=0., variational=False, activation='relu', ensemble=1):
        super().__init__()
        # self.input_linear = BatchEnsembleLinear(model_size, inner_size, ensemble)
        # self.output_linear = BatchEnsembleLinear(inner_size, model_size, ensemble)
        self.variational = variational
        self.dropout = dropout
        self.activation = activation
        self.ensemble = ensemble

        self.in_proj_weight = torch.nn.Parameter(torch.Tensor(inner_size, model_size))
        self.out_proj_weight = torch.nn.Parameter(torch.Tensor(model_size, inner_size))

        self.in_proj_bias = torch.nn.Parameter(torch.Tensor(inner_size))
        self.out_proj_bias = torch.nn.Parameter(torch.Tensor(model_size))

        self.r_in = torch.nn.Parameter(torch.Tensor(ensemble, model_size))
        self.s_in = torch.nn.Parameter(torch.Tensor(ensemble, inner_size))

        self.r_out = torch.nn.Parameter(torch.Tensor(ensemble, inner_size))
        self.s_out = torch.nn.Parameter(torch.Tensor(ensemble, model_size))

    def forward(self, input, indices=None):

        len_x, bsz = input.size(0), input.size(1)
        ensemble = self.r_in.size(0)

        if self.training:
            with torch.no_grad():
                indices = torch.arange(0, bsz, device=input.device, dtype=torch.long)
                indices = torch.remainder(indices, ensemble)

            r_in = torch.index_select(self.r_in, 0, indices)
            s_in = torch.index_select(self.s_in, 0, indices)
            r_out = torch.index_select(self.r_out, 0, indices)
            s_out = torch.index_select(self.s_out, 0, indices)

            input = torch.mul(input, r_in)
            input = F.linear(input, self.in_proj_weight, self.in_proj_bias)
            input = torch.mul(input, s_in)

            input = F.relu(input)
            if self.variational:
                input = variational_dropout(input, p=self.dropout, training=self.training)
            else:
                input = F.dropout(input, p=self.dropout, training=self.training)

            input = torch.mul(input, r_out)
            input = F.linear(input, self.out_proj_weight, self.out_proj_bias)
            input = torch.mul(input, s_out)

            return input
        else:
            input = input.repeat(1, ensemble, 1).view(len_x, ensemble, bsz, input.size(-1))
            input = torch.mul(input, self.r_in.unsqueeze(1))
            input = F.linear(input, self.in_proj_weight, self.in_proj_bias)
            input = torch.mul(input, self.s_in.unsqueeze(1))

            input = F.relu(input)

            input = torch.mul(input, self.r_out.unsqueeze(1))
            input = F.linear(input, self.out_proj_weight, self.out_proj_bias)
            input = torch.mul(input, self.s_out.unsqueeze(1))

            input = torch.mean(input, dim=1)

            return input
        # hidden = self.input_linear(input, indices)
        # hidden = F.relu(hidden)
        # if self.variational:
        #     hidden = variational_dropout(hidden, p=self.dropout, training=self.training)
        # else:
        #     hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        # hidden = self.output_linear(hidden, indices)

        return hidden

    def reset_parameters(self, init='normal'):

        torch.nn.init.xavier_normal_(self.in_proj_weight)
        torch.nn.init.xavier_normal_(self.out_proj_weight)

        torch.nn.init.constant_(self.in_proj_bias, 0.0)
        torch.nn.init.constant_(self.out_proj_bias, 0.0)

        torch.nn.init.normal_(self.r_in, 0.0, 0.02)
        torch.nn.init.normal_(self.s_in, 0.0, 0.02)

        torch.nn.init.normal_(self.r_out, 0.0, 0.02)
        torch.nn.init.normal_(self.s_out, 0.0, 0.02)

        # self.input_linear.reset_parameters(init)
        # self.output_linear.reset_parameters(init)


if __name__ == "__main__":

    bsz = 16
    seq_len = 6
    input_size = 16
    output_size = 32
    ensemble = 72

    model = BatchEnsembleLinear(input_size, output_size, ensemble)

    input = torch.randn((seq_len, bsz, input_size), requires_grad=True)
    print(input)

    model = model.double().cuda()

    input = input.double().cuda()

    print("Gradchecking ...")
    torch.autograd.gradcheck(model, input)