# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Torch utilities for the Trainer class.
"""

import copy
import datetime
import io
import json
import math
import os
import sys
import warnings

from transformers.utils import logging


def set_rng_state_for_device(device_name, device_module, checkpoint_rng_state, is_distributed):
    """Helper to set RNG state for a specific device type (CUDA, NPU, MLU, MUSA)"""
    device_state_key = device_name.lower()
    err_template = "Didn't manage to set back the RNG states of the {backend} because of the following error:\n {exception}\nThis won't yield the same results as if the training had not been interrupted."
    try:
        if is_distributed:
            device_module.random.set_rng_state_all(checkpoint_rng_state[device_state_key])
        else:
            device_module.random.set_rng_state(checkpoint_rng_state[device_state_key])
    except Exception as e:
        # Log error if setting RNG state fails
        logger.error(err_template.format(backend=device_name, exception=e))
