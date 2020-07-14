import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.linear import FeedForward
from onmt.modules.attention import MultiHeadAttention
from torch.autograd.function import Function
import sys
from torch.utils.checkpoint import get_device_states, set_device_states

