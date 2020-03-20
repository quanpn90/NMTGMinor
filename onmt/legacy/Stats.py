""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import datetime
from onmt.train_utils.Meters import AverageMeter, TimeMeter


class Logger(object):

    def __init__(self, optim, scaler=None):

        self.optim = optim
        self.meters = dict()
        self.start_time = time.time()
        self.scaler = scaler

        # initializing the meters
        self.meters["total_loss"] = AverageMeter()
        self.meters["total_words"] = AverageMeter()
        self.meters["report_loss"] = AverageMeter()
        self.meters["report_tgt_words"] = AverageMeter()
        self.meters["report_src_words"] = AverageMeter()
        self.meters["kl"] = AverageMeter()
        self.meters["kl_prior"] = AverageMeter()
        self.meters["gnorm"] = AverageMeter()
        self.meters["oom"] = AverageMeter()
        self.meters["total_sloss"] = AverageMeter()
        self.meters["baseline"] = AverageMeter()
        self.meters["R"] = AverageMeter()
        self.meters["ce"] = AverageMeter()
        self.meters["q_entropy"] = AverageMeter()
        self.meters["q_mean"] = AverageMeter()
        self.meters["q_var"] = AverageMeter()

        self.meters["l2"] = AverageMeter()
        self.meters["l2_target"] = AverageMeter()

        self.meters["total_lang_correct"] = AverageMeter()
        self.meters["total_sents"] = AverageMeter()

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
        l2 = self.meters['l2'].avg if 'l2' in self.meters else None
        l2_target = self.meters['l2_target'].avg if 'l2_target' in self.meters else None

        log_string = (("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d "
                       + "%5.0f tgt tok/s; gnorm %.3f; oom %d") %
                          (epoch, iteration+1, data_size,
                           ppl,
                           self.optim.getLearningRate(),
                           self.optim._step,
                           self.meters["report_tgt_words"].sum/(time.time()-self.start_time),
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

        if self.meters['total_lang_correct'].avg is not None:
            total_lang_correct = self.meters['total_lang_correct'].sum
            acc = total_lang_correct / self.meters['total_sents'].sum * 100.0
            log_string += "; acc %.3f " % acc

        if l2 is not None:
            log_string += "; l2 %.3f" % l2

        if l2_target is not None:
            log_string += "; l2 target %.3f" % l2_target

        # Don't forget to print this ...
        print(log_string)