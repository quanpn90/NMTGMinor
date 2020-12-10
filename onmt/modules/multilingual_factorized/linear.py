import torch
import torch.nn.functional as F


class BatchEnsembleMM(object):

    @staticmethod
    def forward(x, weight, bias, ensemble_r, ensemble_s):
        """
        :param x: [T x B x H]
        :param weight: [H_out x H]
        :param bias: [H_out]
        :param ensemble_r: [B x R x H]
        :param ensemble_s: [B x R x H_out]
        :return:
        """
        bsz, len_x, hin = x.size(1), x.size(0), x.size(2)
        hout = weight.size(0)
        rank = ensemble_s.size(1)

        assert bsz == ensemble_s.size(0)
        # assert ensemble * bsz_per_ensemble == bsz, "Mini-batch must divide evenly to the ensembles"

        # element-wise [T x B x 1 x H] \times [B x R x H]
        x_r = torch.mul(x.unsqueeze(2), ensemble_r)

        # GEMM No Bias.
        x_mm = torch.mm(x_r.view(-1, hin), weight.transpose(0, 1))
        x_mm = x_mm.view(len_x, bsz, rank, hout)

        # element-wise [T x B x R x Hout] \times [B x R x Hout]
        x_s = torch.mul(x_mm, ensemble_s)

        # sum the outputs over rank
        x_s = torch.sum(x_s, dim=2)  #

        # add bias
        y = torch.add(x_s, bias)

        # we need to store the intermediate results for the backward pass
        return y, x_mm, x_r

    # maybe we need some allocated memory as well
    @staticmethod
    def backward(grad_y, x, x_r, x_mm, weight, ensemble_r, ensemble_s):
        """
        :param grad_y: # T x B x H
        :param x:
        :param x_r:
        :param x_mm:
        :param weight:
        :param ensemble_r:
        :param ensemble_s:
        :return:
        """
        bsz, len_x, hin = x.size(1), x.size(0), x.size(2)
        hout = x_mm.size(-1)
        rank = ensemble_s.size(1)

        grad_bias = torch.sum(grad_y, dim=[0, 1])
        grad_s = grad_y.unsqueeze(2)  # [T x B x H] > [T x B x 1 x H]

        # backprop through the last element-wise multiplication
        grad_ensemble_s = torch.mul(grad_s, x_mm)
        grad_ensemble_s = torch.sum(grad_ensemble_s, dim=0)  # sum over the T dimension

        # backprop through the MM
        # [T x B x 1 x H] * [B x R x H] > [T x B x R x H]
        grad_mm = torch.mul(grad_s, ensemble_s)
        grad_mm = grad_mm.view(-1, hout)
        grad_r = torch.mm(grad_mm, weight).view(len_x, bsz, rank, hin)
        # GEMM: [hout x bsz] \times [bsz x hin]
        grad_weight = torch.mm(grad_mm.transpose(0, 1), x_r.view(-1, hin))

        # back prop through the first element-wise multiplication
        # [T x B x R x H] * [B x R x H]
        grad_x = torch.sum(torch.mul(grad_r, ensemble_r), dim=2)
        # grad ensemble r
        # [T x B x R x H] * [T x B x 1 x H]
        grad_ensemble_r = torch.mul(grad_r, x.unsqueeze(2))
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


class MultilingualLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, n_factors=1, rank=1,
                 use_multiplicative=False, weight_drop=0.0):

        super().__init__()

        self.use_multiplicative = use_multiplicative
        self.weight_drop = weight_drop

        self.weight = torch.nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        self.r = torch.nn.Parameter(torch.Tensor(n_factors, rank, input_size))
        self.s = torch.nn.Parameter(torch.Tensor(n_factors, rank, output_size))

        if use_multiplicative:
            self.rm = torch.nn.Parameter(torch.Tensor(n_factors, rank, input_size))
            self.sm = torch.nn.Parameter(torch.Tensor(n_factors, rank, output_size))

        self.reset_parameters()

    def reset_parameters(self, init='normal'):
        if init == 'normal':
            torch.nn.init.xavier_normal_(self.weight)
        else:
            torch.nn.init.xavier_uniform_(self.weight)

        # for batch ensemble we init r_i and s_i with random sign vectors
        if self.use_multiplicative:
            with torch.no_grad():
                self.rm.bernoulli_(0.5).mul_(-2).add_(1)  # -1 1 -1 1
                self.sm.bernoulli_(0.5).mul_(-2).add_(1)
        torch.nn.init.normal_(self.r, 0.0, 0.1)
        torch.nn.init.normal_(self.s, 0.0, 0.1)

    def forward(self, input, indices=None):
        """
        :param input: T x B x H
        :param indices: T x B or B
        :return:
        """
        n_factors = self.r.size(0)
        bsz = input.size(1)
        seq_len = input.size(0)

        weight_ = F.dropout(self.weight, p=self.weight_drop, training=self.training)

        if indices.size(0) == 1 and len(indices.shape) == 1:
            r = torch.index_select(self.r, 0, indices).squeeze(0)
            s = torch.index_select(self.s, 0, indices).squeeze(0)

            # weight_mask = torch.sum(torch.einsum('bi,bj->bij', (s, r)), dim=0)
            # weight_mask = torch.bmm(s.unsqueeze(-1), r.unsqueeze(1))
            if self.use_multiplicative:
                rm = torch.index_select(self.rm, 0, indices).squeeze(0)
                sm = torch.index_select(self.sm, 0, indices).squeeze(0)
                weight_ = weight_ * torch.sum(torch.bmm(rm.unsqueeze(-1), sm.unsqueeze(1)), dim=0)

            weight_mask = torch.bmm(r.unsqueeze(-1), s.unsqueeze(1))
            weight_mask = torch.sum(weight_mask, dim=0)
            weight_ = weight_ + weight_mask

            input = F.linear(input, weight_.t(), self.bias)
            # input = torch.addmm(self.bias, input.view(-1, input.size(-1)), weight_)
            # input = input.view(seq_len, bsz, input.size(-1))
            return input
        else:
            print(indices.size(), input.size())
            raise NotImplementedError

        # if len(indices.shape) == 1:
        #     r = torch.index_select(self.r, 0, indices)
        #     s = torch.index_select(self.s, 0, indices)
        # else:
        #     print("T x B language factors not implemented atm.")
        #     raise NotImplementedError
        #
        # input = torch.mul(input.unsqueeze(2), r)
        # input = F.linear(input, self.weight)
        # input = torch.mul(input, s)
        # input = torch.sum(input, dim=2).add(self.bias)
        # return input
        # return BatchEnsembleLinearFunction.apply(input, self.weight, self.bias, r, s)


# Multilingual Factorized Weight
class MFWPositionWiseFeedForward(torch.nn.Module):
    """
    Position Wise Feedforward model with factorized weights
    """

    def __init__(self, model_size, inner_size, dropout=0., variational=False, activation='relu',
                 n_languages=1, rank=1, use_multiplicative=False, weight_drop=0.0):
        super().__init__()
        self.input_linear = MultilingualLinear(model_size, inner_size, n_languages,
                                               rank, use_multiplicative, weight_drop)
        self.output_linear = MultilingualLinear(inner_size, model_size, n_languages,
                                                rank, use_multiplicative, weight_drop)
        self.variational = variational
        self.dropout = dropout
        self.activation = activation
        self.n_languages = n_languages
        self.weight_drop = weight_drop

        if self.variational:
            from onmt.modules.dropout import variational_dropout
            self.dropout_function = variational_dropout
        else:
            self.dropout_function = F.dropout

    def forward(self, hidden, indices=None):

        # expand if necessary
        # if indices.size(0) == 1 and len(indices.shape) == 1:
        #     indices = indices.expand(hidden.size(1))

        hidden = self.input_linear(hidden, indices)
        hidden = F.relu(hidden, inplace=True)
        hidden = self.dropout_function(hidden, p=self.dropout, training=self.training)
        hidden = self.output_linear(hidden, indices)
        return hidden

    def reset_parameters(self, init='normal'):

        self.input_linear.reset_parameters(init)
        self.output_linear.reset_parameters(init)


if __name__ == "__main__":

    bsz = 16
    seq_len = 6
    input_size = 16
    output_size = 32
    ensemble = 72
    rank = 2

    input = torch.randn((seq_len, bsz, input_size), requires_grad=True)
    weight = torch.randn((output_size, input_size), requires_grad=True)
    bias = torch.randn((output_size,), requires_grad=True)
    r = torch.randn((bsz, rank, input_size), requires_grad=True)
    s = torch.randn((bsz, rank, output_size), requires_grad=True)

    function = BatchEnsembleLinearFunction.apply

    input = input.double().cuda()
    weight = weight.double().cuda()
    bias = bias.double().cuda()
    r = r.double().cuda()
    s = s.double().cuda()

    print("Gradchecking ...")
    torch.autograd.gradcheck(function, (input, weight, bias, r, s))