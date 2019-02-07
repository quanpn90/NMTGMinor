# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .optimizer import Optimizer
from .fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer


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
