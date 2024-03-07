# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, final

import torch
from torch import Tensor
from torch.nn import Module

from onmt.data.vocabulary_info import VocabularyInfo
from onmt.typing import override

class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    max_target_seq_len: int
    target_vocab_info: VocabularyInfo

    def __init__(
        self, max_target_seq_len: int, target_vocab_info: VocabularyInfo
    ) -> None:
        """
        :param max_target_seq_len:
            The maximum length of sequences produced by the model.
        :param target_vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__()

        self.max_target_seq_len = max_target_seq_len
        self.target_vocab_info = target_vocab_info

    @abstractmethod
    def forward(self, batch, *args, **kwargs):
        """
        :param batch:
            The batch of sequences to process.
        """


