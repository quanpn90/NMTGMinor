import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from copy import deepcopy
from time import time
import unittest
import numpy as np

from self_attention_func import self_attn_func


class Parameters(torch.nn.Module):

    def __init__(self, model_size=16, heads=1):
        self.model_size = model_size
        self.heads = heads
        self.head_dim = model_size // heads
        super(Parameters, self).__init__()

        self.in_proj_weight = Parameter(torch.Tensor(3 * model_size, model_size))
        self.out_proj_weight = Parameter(torch.Tensor(model_size, model_size))

        self.in_proj_bias = Parameter(torch.Tensor(3 * model_size))
        self.out_proj_bias = Parameter(torch.Tensor(model_size))

        self.reset_parameters()

    def reset_parameters(self):
        std_ = 0.02
        torch.nn.init.normal_(self.in_proj_weight, 0.0, std_)
        torch.nn.init.normal_(self.out_proj_weight, 0.0, std_)

        torch.nn.init.constant_(self.in_proj_bias, 0.)
        torch.nn.init.constant_(self.out_proj_bias, 0.)


class SelfMultiheadAttnTest(unittest.TestCase):

    def setUp(self, seed=23272123):
        torch.cuda.set_device(0)

        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        self.seq_length = 32
        self.sequences = 512
        self.hidden_dim = 1024
        self.heads = 16
        self.dropout_prob = 0.0

        embed_dim = self.hidden_dim

        self.ref_parameters = Parameters(model_size=self.hidden_dim, heads=self.heads)
        self.ref_parameters = self.ref_parameters.cuda().half()

        self.tst_parameters = deepcopy(self.ref_parameters)

        self.ref_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim,
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

        # Reset seed so parameters are identical
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        self.tst_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim,
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        self.tst_inputs.data.copy_(self.ref_inputs.data)

    def test_input(self):
        print("Checking if all inputs are the same ...")
        self.assertTrue(torch.allclose(self.ref_inputs, self.tst_inputs, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.ref_parameters.in_proj_weight, self.tst_parameters.in_proj_weight,
                                       atol=1e-5, rtol=1e-5))

        print("Done.")

    def test_output(self):

        print("Testing self-attention with random mask ....")
        training = True

        self.ref_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim,
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

        with torch.no_grad():
            self.tst_inputs.copy_(self.ref_inputs)

        mask = ((torch.randn(self.sequences, self.seq_length) > 0)).bool().cuda()

        ref_output, ref_coverage = self_attn_func(False, training, self.heads, self.ref_inputs,
                                                  self.ref_parameters.in_proj_weight,
                                                  self.ref_parameters.out_proj_weight,
                                                  self.ref_parameters.in_proj_bias,
                                                  self.ref_parameters.out_proj_bias,
                                                  mask, self.dropout_prob,
                                                  False, None, False, None,
                                                  False, True)

        tst_output, tst_coverage = self_attn_func(False, training, self.heads, self.tst_inputs,
                                                  self.tst_parameters.in_proj_weight,
                                                  self.tst_parameters.out_proj_weight,
                                                  self.tst_parameters.in_proj_bias,
                                                  self.tst_parameters.out_proj_bias,
                                                  mask, self.dropout_prob,
                                                  False, None, False, None,
                                                  True, True)

        self.assertTrue(torch.allclose(ref_output, tst_output, atol=1e-2, rtol=1e-2))

        grad_outputs_ref = torch.randn_like(tst_output)

        grad_outputs_tst = torch.randn_like(tst_output).copy_(grad_outputs_ref)

        tst_output.data.copy_(ref_output.data)
        ref_output.backward(grad_outputs_ref)
        tst_output.backward(grad_outputs_tst)

        self.assertTrue(torch.allclose(self.ref_parameters.out_proj_weight.grad,
                                       self.tst_parameters.out_proj_weight.grad,
                                       atol=1e-1, rtol=1e-1))

        print("GRAD TEST", self.tst_parameters.in_proj_weight.grad)
        print("GRAD TEST", self.ref_parameters.in_proj_weight.grad)
        print("GRAD TEST", self.ref_parameters.in_proj_weight.grad - self.tst_parameters.in_proj_weight.grad)

        # self.assertTrue(torch.allclose(self.ref_parameters.in_proj_weight.grad,
        #                                self.tst_parameters.in_proj_weight.grad,
        #                                atol=1e-2, rtol=1e-2))
        np.testing.assert_allclose(
            self.ref_parameters.in_proj_weight.grad.detach().cpu().numpy(),
            self.tst_parameters.in_proj_weight.grad.detach().cpu().numpy(),
            atol=1e-2, rtol=1e-2)

        # self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad,
        #                                atol=1e-3, rtol=1e-3))

        np.testing.assert_allclose(
            self.ref_inputs.grad.detach().cpu().numpy(),
            self.tst_inputs.grad.detach().cpu().numpy(),
            atol=1e-3, rtol=1e-3)

    def test_output_autoregressive(self):

        print("Testing self-attention with time mask ....")
        training = True

        self.ref_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim,
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

        with torch.no_grad():
            self.tst_inputs.copy_(self.ref_inputs)

        # mask = ((torch.randn(self.sequences, self.seq_length) > 0)).bool().cuda()
        mask = torch.triu(
                self.ref_inputs.new_ones(self.seq_length, self.seq_length), diagonal=1).bool()

        ref_output, ref_coverage = self_attn_func(True, training, self.heads, self.ref_inputs,
                                                  self.ref_parameters.in_proj_weight,
                                                  self.ref_parameters.out_proj_weight,
                                                  self.ref_parameters.in_proj_bias,
                                                  self.ref_parameters.out_proj_bias,
                                                  mask, self.dropout_prob,
                                                  False, None, False, None,
                                                  False, True)

        tst_output, tst_coverage = self_attn_func(True, training, self.heads, self.tst_inputs,
                                                  self.tst_parameters.in_proj_weight,
                                                  self.tst_parameters.out_proj_weight,
                                                  self.tst_parameters.in_proj_bias,
                                                  self.tst_parameters.out_proj_bias,
                                                  mask, self.dropout_prob,
                                                  False, None, False, None,
                                                  True, True)

        self.assertTrue(torch.allclose(ref_output, tst_output, atol=1e-2, rtol=1e-2))
        grad_outputs_ref = torch.randn_like(tst_output)

        grad_outputs_tst = torch.randn_like(tst_output).copy_(grad_outputs_ref)

        tst_output.data.copy_(ref_output.data)
        ref_output.backward(grad_outputs_ref)
        tst_output.backward(grad_outputs_tst)

        self.assertTrue(torch.allclose(self.ref_parameters.out_proj_weight.grad,
                                       self.tst_parameters.out_proj_weight.grad,
                                       atol=1e-1, rtol=1e-1))

        print("GRAD TEST", self.tst_parameters.in_proj_weight.grad)
        print("GRAD TEST", self.ref_parameters.in_proj_weight.grad)
        print("GRAD TEST", self.ref_parameters.in_proj_weight.grad - self.tst_parameters.in_proj_weight.grad)

        # self.assertTrue(torch.allclose(self.ref_parameters.in_proj_weight.grad,
        #                                self.tst_parameters.in_proj_weight.grad,
        #                                atol=1e-2, rtol=1e-2))
        #
        np.testing.assert_allclose(
            self.ref_parameters.in_proj_weight.grad.data.cpu().numpy(),
            self.tst_parameters.in_proj_weight.grad.data.cpu().numpy(),
            atol=1e-3, rtol=1e-3)

        # self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad,
        #                                atol=1e-3, rtol=1e-3))
        #
        np.testing.assert_allclose(
            self.ref_inputs.detach().cpu().numpy(),
            self.tst_inputs.detach().cpu().numpy(),
            atol=1e-3, rtol=1e-3)

    def test_performance(self):
        training = True
        for dropout in [0.0, 0.5]:

            mask = ((torch.randn(self.sequences, self.seq_length) > 0)).bool().cuda()

            num_iters = 32
            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                ref_output, ref_coverage = self_attn_func(False, training, self.heads, self.ref_inputs,
                                                          self.ref_parameters.in_proj_weight,
                                                          self.ref_parameters.out_proj_weight,
                                                          self.ref_parameters.in_proj_bias,
                                                          self.ref_parameters.out_proj_bias,
                                                          mask, dropout,
                                                          False, None, False, None,
                                                          False, True)

                grad_outputs_ref = torch.randn_like(ref_output)
                ref_output.backward(grad_outputs_ref)
                self.ref_parameters.zero_grad()

                tst_output, tst_coverage = self_attn_func(False, training, self.heads, self.tst_inputs,
                                                          self.tst_parameters.in_proj_weight,
                                                          self.tst_parameters.out_proj_weight,
                                                          self.tst_parameters.in_proj_bias,
                                                          self.tst_parameters.out_proj_bias,
                                                          mask, dropout,
                                                          False, None, False, None,
                                                          True, True)

                grad_outputs_tst = torch.randn_like(tst_output)
                tst_output.backward(grad_outputs_tst)
                self.tst_parameters.zero_grad()

            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                ref_output, ref_coverage = self_attn_func(False, training, self.heads, self.ref_inputs,
                                                          self.ref_parameters.in_proj_weight,
                                                          self.ref_parameters.out_proj_weight,
                                                          self.ref_parameters.in_proj_bias,
                                                          self.ref_parameters.out_proj_bias,
                                                          mask, dropout,
                                                          False, None, False, None,
                                                          False, True)

                grad_outputs_ref = torch.randn_like(ref_output)
                ref_output.backward(grad_outputs_ref)
                self.ref_parameters.zero_grad()

            torch.cuda.synchronize()
            stop_time = time()
            print(F"\nPytorch Self-Attn time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

            torch.cuda.profiler.start()
            torch.cuda.synchronize()
            start_time = time()
            for _ in range(num_iters):
                tst_output, tst_coverage = self_attn_func(False, training, self.heads, self.tst_inputs,
                                                          self.tst_parameters.in_proj_weight,
                                                          self.tst_parameters.out_proj_weight,
                                                          self.tst_parameters.in_proj_bias,
                                                          self.tst_parameters.out_proj_bias,
                                                          mask, dropout,
                                                          False, None, False, None,
                                                          True, True)

                grad_outputs_tst = torch.randn_like(tst_output)
                tst_output.backward(grad_outputs_tst)
                self.tst_parameters.zero_grad()

            torch.cuda.synchronize()
            stop_time = time()
            print(F"\nCUDA Self-Attn time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")


if __name__ == '__main__':
    unittest.main()
