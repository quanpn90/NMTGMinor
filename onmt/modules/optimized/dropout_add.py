import torch
import unittest
import numpy as np
from time import time

try:
    import apex.amp as amp
    from apex.amp import half_function
except (ModuleNotFoundError, ImportError) as e:
    amp = None
    from .compat import half_function

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .compat import custom_fwd, custom_bwd

try:
    import fused_dropout_add_cuda
except (ModuleNotFoundError, ImportError) as e:
    fused_dropout_add_cuda = None


class FusedDropoutAdd(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, residual, dropout_prob, is_training):

        null_tensor = torch.tensor([])
        dropout_prob_t = torch.tensor([dropout_prob])

        if fused_dropout_add_cuda is not None and input.is_cuda and input.type() == 'torch.cuda.HalfTensor':
            # print("Fused dropout add")
            dropout_mask, output = fused_dropout_add_cuda.forward(is_training, input, residual, dropout_prob)
        else:
            if is_training:
                dropout_results, dropout_mask = torch._fused_dropout(input, p=(1. - dropout_prob))
            else:
                dropout_mask = null_tensor
                dropout_results = input
            output = dropout_results + residual

        ctx.save_for_backward(dropout_mask, dropout_prob_t)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grads):

        dropout_mask, dropout_prob_t = ctx.saved_tensors

        if fused_dropout_add_cuda is not None and output_grads.is_cuda and output_grads.dtype == torch.float16:
            grad_input = fused_dropout_add_cuda.backward(output_grads, dropout_mask, dropout_prob_t[0])
        else:
            grad_input = torch._masked_scale(output_grads, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))

        return grad_input, output_grads, None, None


@half_function
def fused_dropout_add(input, residual, dropout, is_training):

    return FusedDropoutAdd.apply(input, residual, dropout, is_training)


if __name__ == '__main__':

    batch_size = 512
    seq_len = 64
    hidden_size = 1024
    num_iters = 100
    dropout = 0.0

    class TestMLP(unittest.TestCase):
#
        def test_creation(self):
            test_input = torch.empty(seq_len, batch_size, hidden_size,
                                     device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()
            test_add_input = torch.empty(seq_len, batch_size, hidden_size,
                                         device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()

            output = fused_dropout_add(test_input, test_add_input, dropout, True)

        def test_numeric(self):

            test_input = torch.empty(seq_len, batch_size, hidden_size,
                                     device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()
            test_add_input = torch.empty(seq_len, batch_size, hidden_size,
                                         device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()

            output = fused_dropout_add(test_input, test_add_input, dropout, True)

            ref_input = test_input.clone().detach().requires_grad_()
            ref_add_input = test_add_input.clone().detach().requires_grad_()

            ref_output = ref_input + ref_add_input

            np.testing.assert_allclose(
                ref_output.detach().cpu().numpy(),
                output.detach().cpu().numpy(),
                atol=1e-5, rtol=1e-4)

            output.mean().mul(10.).backward()
            ref_output.mean().mul(10.).backward()

            np.testing.assert_allclose(
                test_input.grad.detach().cpu().numpy(),
                ref_input.grad.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

            np.testing.assert_allclose(
                test_add_input.grad.detach().cpu().numpy(),
                ref_add_input.grad.detach().cpu().numpy(),
                atol=1e-7, rtol=1e-5)

        def test_performance_half(self):

            test_input = torch.empty(seq_len, batch_size, hidden_size,
                                     device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()
            test_add_input = torch.empty(seq_len, batch_size, hidden_size,
                                         device="cuda", dtype=torch.half).uniform_(-1., 1.).requires_grad_()

            ref_input = test_input.clone().detach().requires_grad_()
            ref_add_input = test_add_input.clone().detach().requires_grad_()

            # Warm up GPU
            for _ in range(100):
                ref_out = ref_input + ref_add_input
                ref_loss = ref_out.mean()
                ref_loss.backward()
                output = fused_dropout_add(test_input, test_add_input, dropout, False)
                test_loss = output.mean()
                test_loss.backward()

            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                ref_out = ref_input + ref_add_input
                ref_loss = ref_out.mean()
                ref_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"\nPytorch DropoutAdd time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                output = fused_dropout_add(test_input, test_add_input, dropout, False)
                test_loss = output.mean()
                test_loss.backward()
            torch.cuda.synchronize()
            stop_time = time()
            print(F"C++ DropoutAdd time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
            torch.cuda.profiler.stop()

    unittest.main()
