# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import logging
import os
import torch

from .optimizer import Optimizer
from .fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer


logger = logging.getLogger(__name__)


OPTIMIZER_REGISTRY = {}


def register_optimizer(name):
    """Decorator to register a new Optimizer."""

    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError('Cannot register duplicate Optimizer ({})'.format(name))
        if not issubclass(cls, Optimizer):
            raise ValueError('Optimizer ({}: {}) must extend Optimizer'.format(name, cls.__name__))
        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return register_optimizer_cls


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('nmtg.optim.' + module)


def get_optimizer_type(name):
    return OPTIMIZER_REGISTRY[name]


def get_optimizer_names():
    return list(OPTIMIZER_REGISTRY.keys())


def build_optimizer(name, args, params):
    if args.fp16:
        if args.cuda and torch.cuda.get_device_capability(0)[0] < 7:
            logger.warning('Your device does NOT support faster training with --fp16, '
                           'please switch to FP32 which is likely to be faster')
        if args.memory_efficient_fp16:
            optimizer = MemoryEfficientFP16Optimizer.build_optimizer(args, params)
        else:
            optimizer = FP16Optimizer.build_optimizer(args, params)
    else:
        if args.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
            logger.info('Your device may support faster training with --fp16')
        optimizer = get_optimizer_type(name).build_optimizer(args, params)
    return optimizer
