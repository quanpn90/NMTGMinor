# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import pkgutil
from .preprocessor import Preprocessor

PREPROCESSOR_REGISTRY = {}


def register_preprocessor(name):
    """Decorator to register a new Preprocessor."""

    def register_preprocessor_cls(cls):
        if name in PREPROCESSOR_REGISTRY:
            raise ValueError('Cannot register duplicate Preprocessor ({})'.format(name))
        if not issubclass(cls, Preprocessor):
            raise ValueError('Preprocessor ({}: {}) must extend Preprocessor'.format(name, cls.__name__))
        PREPROCESSOR_REGISTRY[name] = cls
        return cls

    return register_preprocessor_cls


# automatically import any Python files in the preprocessors/ directory
for module in pkgutil.walk_packages(__path__, 'nmtg.preprocessors.'):
    importlib.import_module(module.name)


def get_preprocessor_type(name):
    return PREPROCESSOR_REGISTRY[name]


def get_preprocessor_names():
    return list(PREPROCESSOR_REGISTRY.keys())
