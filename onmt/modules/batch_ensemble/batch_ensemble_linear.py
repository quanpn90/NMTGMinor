import torch
import torch.nn.functional as F


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
        ctx.save_for_backward(x, weight, bias, ensemble_r, ensemble_s, x_mm, x_rr)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        x, weight, bias, ensemble_r, ensemble_s, x_mm, x_rr = ctx.saved_tensors

        grad_x, grad_weight, grad_bias, grad_ensemble_r, grad_ensemble_s = \
            BatchEnsembleMM.backward(grad_output, x, weight, bias, ensemble_r, ensemble_s, x_mm, x_rr)

        return grad_x, grad_weight, grad_bias, grad_ensemble_r, grad_ensemble_s


class BatchEnsembleLinear(torch.nn.Module):

    # TODO: write gradcheck testing
    def __init__(self, input_size, output_size, ensemble):

        super().__init__()

        self.weight = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        self.r = torch.nn.Parameter(torch.Tensor(ensemble, input_size))
        self.s = torch.nn.Parameter(torch.Tensor(ensemble, output_size))

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.bias)

        # for batch ensemble we init r_i and s_i with random sign vectors
        self.r.bernoulli_(0.5).mul_(-2).add_(1)
        self.s.bernoulli_(0.5).mul_(-2).add_(1)

    def forward(self, input):

        ensemble = self.r.size(0)

        # during training, we randomly select the ensemble_id into batch size
        if self.training:
            if len(input.shape) == 3:
                len_x, bsz, hin = input.size(0), input.size(1), input.size(2)
                rand_indices = torch.randint(0, ensemble, (bsz,), device=input.device, dtype=torch.long)
                r = torch.index_select(self.r, rand_indices, dim=0)
                s = torch.index_select(self.s, rand_indices, dim=0)
                return BatchEnsembleLinear.apply(input, self.weight, self.bias, r, s)
            if len(input.shape) == 2:
                bsz, hin = input.size(0), input.size(1)
                rand_indices = torch.randint(0, ensemble, (bsz,), device=input.device, dtype=torch.long)
                r = torch.index_select(self.r, rand_indices, dim=0)
                s = torch.index_select(self.s, rand_indices, dim=0)
                return torch.mul(F.linear(torch.mul(input, r), weight, bias), s)

        # during eval we have to repeat the dimensions ensemble times
        else:
            if len(input.shape) == 3:
                len_x, bsz, hin = input.size(0), input.size(1), input.size(2)
                input = input.repeat(1, ensemble, 1)

                # we need the transpose step to ensure that both should have ensemble x batch
                # but should it be ensemble x batch or batch x ensemble ? ...
                # TODO: test at decoding time. batch_size=beam_size=1 should yield the same result
                r = self.r.repeat(bsz, 1).view(bsz, ensemble, self.r.size(-1)).\
                    transpose(0, 1).view(-1, self.r.size(-1))
                s = self.s.repeat(bsz, 1).view(bsz, ensemble, self.s.size(-1)).\
                    transpose(0, 1).view(-1, self.s.size(-1))
                output = BatchEnsembleLinear.apply(input, self.weight, self.bias, r, s)
                output = output.view(len_x, ensemble, bsz, -1)
                output = torch.mean(output, dim=1)
                return output
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
