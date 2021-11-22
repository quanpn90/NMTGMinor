from __future__ import division

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import os, sys
from onmt.model_factory import build_model, build_language_model, build_classifier, optimize_model
from copy import deepcopy
from onmt.utils import checkpoint_paths, normalize_gradients
import glob
import unittest
from time import time

from torch.cuda.amp import autocast


def custom_build_model(opt, dict, lm=False, type='seq2seq'):
    if type == 'seq2seq':
        if not lm:
            model = build_model(opt, dict)
        else:
            model = build_language_model(opt, dict)
    elif type == 'classifier':
        model = build_classifier(opt, dict)

    # optimize_model(model)

    return model


class TestWav2vec(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestWav2vec, self).__init__(*args, **kwargs)
        model = "saves17/model_test.pt"

        # checkpoint for main model
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage)

        if 'optim' in checkpoint:
            del checkpoint['optim']

        main_checkpoint = checkpoint

        best_checkpoint = main_checkpoint

        model_opt = checkpoint['opt']

        dicts = checkpoint['dicts']

        # only create the object
        model_opt.enc_state_dict = None
        model_opt.dec_state_dict = None

        main_model = custom_build_model(model_opt, checkpoint['dicts'], lm=False, type="seq2seq")
        fast_model = custom_build_model(model_opt, checkpoint['dicts'], lm=False, type="seq2seq")

        print("Loading main model from %s ..." % model)

        from copy import deepcopy

        # if opt.cuda:
        #     main_model = main_model.cuda()
        #
        print("Loading model complete ...")
        self.main_model = main_model.encoder
        self.fast_model = fast_model.encoder

        self.fast_model.load_state_dict(self.main_model.state_dict())

        self.main_model = self.main_model.cuda()

        # make fast_model fast
        optimize_model(self.fast_model)
        self.fast_model = self.fast_model.cuda()

        seq_len = 512 * 256
        bsz = 16

        self.ref_input = torch.randn(bsz, seq_len, 1, dtype=torch.float32, device=torch.device("cuda"))
        self.input = torch.randn(bsz, seq_len // 256, 512, dtype=torch.float32, device=torch.device("cuda"))
        self.tst_input = self.ref_input.new(*self.ref_input.size()).copy_(self.ref_input)
        self.mask = ((torch.randn(bsz, seq_len) > 0)).bool().cuda().zero_()  # no mask
        self.short_mask = ((torch.randn(bsz, seq_len // 256) > 0)).bool().cuda().zero_()  # no mask

    def setUp(self):

        pass

    def test_input(self):
        print("Checking if all inputs are the same ...")
        self.assertTrue(torch.allclose(self.ref_input, self.tst_input, atol=1e-5, rtol=1e-5))

        ref_parameters = list(self.main_model.parameters())
        tst_parameters = list(self.fast_model.parameters())

        # print("Checking if all parameters are the same ...")
        # for (ref_param, tst_param) in zip(ref_parameters, tst_parameters):
        #     self.assertTrue(torch.allclose(ref_param, tst_param, atol=1e-5, rtol=1e-5))

        print("Done.")

    def test_output(self):
        # They seem to have dropout going on?
        self.main_model.eval()
        self.fast_model.eval()
        with autocast():
            ref_output = self.main_model.test_run(self.ref_input, self.mask)

            tst_output = self.fast_model.test_run(self.tst_input, self.mask)

        print(ref_output - tst_output)
        self.assertTrue(torch.allclose(ref_output, tst_output, atol=1e-2, rtol=1e-2))

        del ref_output
        del tst_output

    def test_performance(self):
        training = True

        num_iters = 10

        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            with autocast():
                ref_output = self.main_model.test_run(self.ref_input, self.mask)

            grad_outputs_ref = torch.randn_like(ref_output)
            ref_output.backward(grad_outputs_ref)

            with autocast():
                tst_output = self.fast_model.test_run(self.tst_input, self.mask)

            grad_outputs_tst = torch.randn_like(tst_output)
            tst_output.backward(grad_outputs_tst)

        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            with autocast():
                ref_output = self.main_model.test_run(self.ref_input, self.mask)

            grad_outputs_ref = torch.randn_like(ref_output)
            ref_output.backward(grad_outputs_ref)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"\nPytorch Wav2vec time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            with autocast():
                tst_output = self.fast_model.test_run(self.tst_input, self.mask)

            grad_outputs_tst = torch.randn_like(tst_output)
            tst_output.backward(grad_outputs_tst)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"\nCUDA Wav2vec time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            with autocast():
                ref_output = self.main_model.test_run(self.input, self.short_mask)

            grad_outputs_ref = torch.randn_like(ref_output)
            ref_output.backward(grad_outputs_ref)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"\nPytorch Wav2vec-No-CNN time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        start_time = time()
        for _ in range(num_iters):
            with autocast():
                tst_output = self.fast_model.test_run(self.input, self.short_mask)

            grad_outputs_tst = torch.randn_like(tst_output)
            tst_output.backward(grad_outputs_tst)

        torch.cuda.synchronize()
        stop_time = time()
        print(F"\nCUDA Wav2vec-No-CNN time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")


if __name__ == '__main__':
    unittest.main()
