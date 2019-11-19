from __future__ import division

import math
import torch
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter
from onmt.modules.WordDrop import switchout

