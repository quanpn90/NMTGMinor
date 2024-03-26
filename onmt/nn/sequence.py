# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, final

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
# from torcheval.metrics import Mean, Sum, Throughput

from onmt.data.vocabulary_info import VocabularyInfo
from onmt.nn.model import Model


class SequenceModel(Model, ABC):
    """Represents a sequence model."""

    max_seq_len: int
    vocab_info: VocabularyInfo

    def __init__(self, max_seq_len: int, vocab_info: VocabularyInfo) -> None:
        """
        :param max_seq_len:
            The maximum length of sequences produced by the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.vocab_info = vocab_info

    @abstractmethod
    def forward(self, batch):
        """
        :param batch:
            The batch of sequences to process.
        """

