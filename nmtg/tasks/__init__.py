# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .task import Task


TASK_REGISTRY = {}


def register_task(name):
    """Decorator to register a new Task."""

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate Task ({})'.format(name))
        if not issubclass(cls, Task):
            raise ValueError('Task ({}: {}) must extend Task'.format(name, cls.__name__))
        TASK_REGISTRY[name] = cls
        return cls

    return register_task_cls


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('nmtg.tasks.' + module)


def get_task_type(name):
    return TASK_REGISTRY[name]


def get_task_names():
    return list(TASK_REGISTRY.keys())
