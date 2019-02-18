""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import datetime 


class Logger(object):

    def __init__(self, optim, meters, scaler=None):

        self.optim = optim
        self.meters = meters
        self.start_time = time.time()
        self.scaler = scaler

    def reset(self):

        for key in self.meters:
            self.meters[key].reset()
        self.start_time = time.time()

    def reset_meter(self, key):
        self.meters[key].reset()

    def reset_time(self):
        self.start_time = time.time()

    def log(self, epoch, iteration, data_size):

        ppl = math.exp(self.meters["report_loss"].sum / self.meters["report_tgt_words"].sum)
        grad_norm = self.meters["gnorm"].avg
        oom_count = self.meters["oom"].sum

        baseline = self.meters['baseline'].avg
        kl = self.meters['kl'].avg # normalized by 6 distributions and the batch_size
        R = self.meters['R'].avg # 
        ce = self.meters['ce'].avg
        q_ent = self.meters['q_entropy'].avg
        q_mean = self.meters['q_mean'].avg
        q_var = self.meters['q_var'].avg
        kl_prior = self.meters['kl_prior'].avg

        log_string = (("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " + "%5.0f tgt tok/s; lscale %0.2f; gnorm %.3f; oom %d") %
                          (epoch, iteration+1, data_size,
                           ppl,
                           self.optim.getLearningRate(),
                           self.optim._step,
                           self.meters["report_tgt_words"].sum/(time.time()-self.start_time),
                           self.scaler.loss_scale,
                           grad_norm if grad_norm else 0, 
                           oom_count))

        if ce is not None:
            log_string += "; ce %.3f" % ce

        if baseline is not None:
            log_string += "; bl %.3f" % baseline

        if kl is not None:
            log_string += "; kl %.3f" % kl

        if kl_prior is not None:
            log_string += "; kl_prior %.3f" % kl_prior

        if R is not None:
            log_string += "; R %.3f" % R

        if q_ent is not None:
            log_string += "; q_ent %.3f" % q_ent

        if q_mean is not None:
            log_string += "; q_mean %.3f" % q_mean

        if q_var is not None:
            log_string += "; q_var %.3f" % q_var

        # Don't forget to print this ...
        print(log_string)