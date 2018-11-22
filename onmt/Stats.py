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
        R = self.meters['R'].avg

        log_string = (("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " + "%5.0f tgt tok/s; lscale %0.2f; gnorm %.3f; oom %d") %
                          (epoch, iteration+1, data_size,
                           ppl,
                           self.optim.getLearningRate(),
                           self.optim._step,
                           self.meters["report_tgt_words"].sum/(time.time()-self.start_time),
                           self.scaler.loss_scale,
                           grad_norm if grad_norm else 0, 
                           oom_count))


        

        if baseline is not None:
            log_string += "; bl %.3f" % baseline

        if kl is not None:
            log_string += "; kl %.3f" % kl

        if R is not None:
            log_string += "; R %.3f" % R


        log_string += "; %s elapsed" % str(datetime.timedelta(seconds=int(time.time() - self.start_time)))

        # Don't forget to print this ...
        print(log_string)