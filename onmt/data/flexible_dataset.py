from __future__ import division

import math
import torch
import torch.utils.data
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter
from onmt.modules.dropout import switchout
import numpy as np


# 1. create a long tensor up to size N  (N is strictly larger than the largest)
# 2. fill in the tensor with samples

# input: batch (list) of list of tensors

#