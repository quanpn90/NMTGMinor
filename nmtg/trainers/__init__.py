# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .trainer import Trainer


TRAINER_REGISTRY = {}


def register_trainer(name):
    """Decorator to register a new Trainer."""

    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError('Cannot register duplicate Trainer ({})'.format(name))
        if not issubclass(cls, Trainer):
            raise ValueError('Trainer ({}: {}) must extend Trainer'.format(name, cls.__name__))
        TRAINER_REGISTRY[name] = cls
        return cls

    return register_trainer_cls


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('nmtg.trainers.' + module)


def get_trainer_type(name):
    return TRAINER_REGISTRY[name]


def get_trainer_names():
    return list(TRAINER_REGISTRY.keys())
