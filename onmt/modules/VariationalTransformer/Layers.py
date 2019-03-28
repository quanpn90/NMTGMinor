	import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 

from onmt.modules.Bottle import Bottle
from onmt.modules.StaticDropout import StaticDropout
from onmt.modules.Linear import XavierLinear, group_linear, FeedForward
from onmt.modules.MaxOut import MaxOut
from onmt.modules.GlobalAttention import MultiHeadAttention
from onmt.modules.PrePostProcessing import PrePostProcessing

