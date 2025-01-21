# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Whisper model configuration"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from transformers.models.whisper.configuration_whisper import WhisperConfig


class BatchEnsembleWhisperConfig(WhisperConfig):

    def __init__(self, n_ensembles=4, ensemble_init="ones", **kwargs):
        super().__init__(**kwargs)

        self.n_ensembles = n_ensembles
        self.ensemble_init = ensemble_init

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict["n_ensembles"] = self.n_ensembles
        base_dict["ensemble_init"] = self.ensemble_init

        return base_dict

    @classmethod
    def from_whisper_config(cls, whisper_config: WhisperConfig,
                            n_ensembles=1, ensemble_init="random_sign", **kwargs):
        # Convert WhisperConfig to BatchEnsembleWhisperConfig
        config_dict = whisper_config.to_dict()
        config_dict["n_ensembles"] = n_ensembles  # Default or inferred value
        config_dict["ensemble_init"] = ensemble_init  # Default or inferred value
        return cls(**config_dict)


def convert_to_whisper_config(config: BatchEnsembleWhisperConfig):
    # Get the dictionary representation of the config
    config_dict = config.to_dict()

    # Remove additional BatchEnsemble-specific fields
    whisper_config_dict = {
        k: v for k, v in config_dict.items() if k not in {"n_ensembles", "ensemble_init"}
    }

    # Create a WhisperConfig from the filtered dictionary
    whisper_config = WhisperConfig.from_dict(whisper_config_dict)
    return whisper_config
