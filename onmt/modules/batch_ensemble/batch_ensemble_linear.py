

class BatchEnsembleMMNoBias(object):

    @staticmethod
    def forward(x, weight, ensemble_r, ensemble_s,
                bsz_per_ensemble):

        bsz, len_x, hsize = x.size(0), x.size(1), x.size(2)

        n_ensemble = ensemble_s.size(0)
        assert n_ensemble * bsz_per_ensemble == bsz, "Mini-batch must divide evenly to the ensembles"

        # ensemble_r and s should have size [n_ensemble x hsize]
        x_r = torch.mul(x.view(bsz_per_ensemble, n_ensemble, len_x, hsize), ensemble_r.unsqueeze(1))

        # GEMM No Bias. Otherwise use addmm
        x_mm = torch.mm(x_r.view(bsz, len_x, hsize).view(-1, hsize), weight.transpose(0, 1))
        x_s = torch.mul(x_mm.view(bsz, len_x, hsize).view(bsz_per_ensemble, n_ensemble, len_x, hsize),
                        ensemble_s.unsqueeze(1))

        x_s = x_s.view(bsz, len_x, hsize)

        # we need them for backward pass
        return x_s, x_mm, x_r

    @staticmethod
    def backward(grad_y, x, x_r, x_mm, weight, ensemble_r, ensemble_s,
                 bsz_per_ensemble):
        bsz, len_x, hsize = x.size(0), x.size(1), x.size(2)
        n_ensemble = ensemble_s.size(0)

        # backprop through the last element-wise multiplication
        grad_s = grad_y.view(bsz_per_ensemble, n_ensemble, len_x, hsize)
        grad_ensemble_s = torch.mul(grad_s,
                                    x_mm.view(bsz, len_x, hsize).view(bsz_per_ensemble, n_ensemble, len_x, hsize))
        grad_ensemble_s = torch.sum(grad_ensemble_s, dim=[0, 2])

        # backprop through the MM
        grad_mm = torch.mul(grad_s, ensemble_s.unsqueeze(1))
        grad_mm = grad_mm.view(bsz, len_x, h_size).view(-1, h_size)
        grad_r = torch.mm(grad_mm, weight)
        grad_weight = torch.mm(grad_mm.transpose(0, 1), x_r.view(bsz, len_x, hsize).view(-1, hsize))

        # back prop through the first element-wise multiplication
        grad_r = grad_r.view(bsz, len_x, h_size).view(bsz_per_ensemble, n_ensemble, len_x, h_size)
        grad_x = torch.mul(grad_r,
                           ensemble_r.size(0))
        grad_ensemble_r = torch.mul(grad_r, x.view(bsz_per_ensemble, n_ensemble, len_x, h_size))
        grad_ensemble_r = torch.sum(grad_ensemble_r, dim=[0, 2])
        grad_x = grad_x.view(bsz, len_x, h_size)

        return grad_x, grad_weight, grad_ensemble_r, grad_ensemble_s

