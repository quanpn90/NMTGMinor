import torch

import unittest

from modules.self_multihead_attn import SelfMultiheadAttn
from time import time


class SelfMultiheadAttnTest(unittest.TestCase):
    def setUp(self, seed=1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.seq_length = 512
        self.sequences = 8
        self.hidden_dim = 1024
        self.heads = 16
        self.dropout_prob = 0.0

        self.ref_layer = SelfMultiheadAttn(self.hidden_dim,
                                           self.heads,
                                           dropout=self.dropout_prob,
                                           bias=True,
                                           mask_additive=True,
                                           impl='default')
        self.ref_layer.cuda().half()
        self.ref_layer.reset_parameters()
        self.ref_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim,
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        # Reset seed so parameters are identical
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.tst_layer = SelfMultiheadAttn(self.hidden_dim,
                                           self.heads,
                                           dropout=self.dropout_prob,
                                           bias=True,
                                           mask_additive=True,
                                           impl='fast')
        self.tst_layer.cuda().half()
        self.tst_layer.reset_parameters()

        self.tst_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim,
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

    def test_self_multihead_attn_additive_mask(self):
        grads = torch.randn_like(self.tst_inputs)
        mask = ((torch.randn(self.sequences, self.seq_length) > 0) * -10000.0).half().cuda()
        # print(mask)

        for i in range(20):
            grads = torch.randn_like(self.tst_inputs)
            mask = ((torch.randn(self.sequences, self.seq_length) > 0) * -10000.0).half().cuda()

            ref_outputs, _ = self.ref_layer.forward(self.ref_inputs,
                                                    self.ref_inputs,
                                                    self.ref_inputs,
                                                    key_padding_mask=mask,
                                                    need_weights=False,
                                                    attn_mask=None,
                                                    is_training=True)

            tst_outputs, _ = self.tst_layer.forward(self.tst_inputs,
                                                    self.tst_inputs,
                                                    self.tst_inputs,
                                                    key_padding_mask=mask,
                                                    need_weights=False,
                                                    attn_mask=None,
                                                    is_training=True)

            self.ref_inputs.backward(grads)
            self.tst_inputs.backward(grads)

            self.assertTrue(torch.allclose(self.ref_inputs, self.tst_inputs, atol=1e-3, rtol=1e-3))
            self.assertTrue(not torch.any(torch.isnan(self.tst_inputs.grad)))
            self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
            self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))

    def test_speed(self):
        grads = torch.randn_like(self.tst_inputs)
        mask = ((torch.randn(self.sequences, self.seq_length) > 0) * -10000.0).half().cuda()
        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()

        num_iters = 100
        for i in range(num_iters):
            ref_outputs, _ = self.ref_layer.forward(self.ref_inputs,
                                                    self.ref_inputs,
                                                    self.ref_inputs,
                                                    key_padding_mask=mask,
                                                    need_weights=False,
                                                    attn_mask=None,
                                                    is_training=True)

            self.ref_inputs.backward(grads)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"\nPytorch Self ATTN time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()

        num_iters = 100
        for i in range(num_iters):
            tst_outputs, _ = self.tst_layer.forward(self.tst_inputs,
                                                    self.tst_inputs,
                                                    self.tst_inputs,
                                                    key_padding_mask=mask,
                                                    need_weights=False,
                                                    attn_mask=None,
                                                    is_training=True)

            self.tst_inputs.backward(grads)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"\nC++ Self ATTN time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")


if __name__ == '__main__':
    unittest.main()
