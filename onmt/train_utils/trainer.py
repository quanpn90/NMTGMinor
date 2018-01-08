from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import random 
import numpy as np

class BaseTrainer(object):
    
    def __init__(self, model, loss_function, trainSets, validSets, dataset, optim, evaluator, opt):
        
        self.model = model
        self.trainSet = trainSets
        self.validSet = validSets
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.optim = optim 
        self.evaluator = evaluator
        self.opt = opt
        
        self.loss_function = loss_function
        
        
       

class XETrainer(object):

    def __init__(self, model, loss_function, trainSets, validSets, dataset, optim, evaluator, opt):
        
        super(BaseTrainer, self).__init__(model, loss_function, trainSets, validSets, dataset, optim, evaluator, opt)