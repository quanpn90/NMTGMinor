# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Optional, Tuple, final

from torch import Tensor

from onmt.data.vocabulary_info import VocabularyInfo
from onmt.nn.sequence import SequenceModel
from onmt.nn.transformer.frontend import TransformerFrontend
# from fairseq2.nn.incremental_state import IncrementalStateBag
# from fairseq2.nn.padding import PaddingMask
# from fairseq2.nn.projection import Projection
from onmt.nn.projection import Projection
from onmt.nn.transformer.decoder import TransformerDecoder
# from fairseq2.nn.transformer import TransformerDecoder
from onmt.typing import override


class DecoderModel(SequenceModel):
    """Represents a decoder model."""

    model_dim: int

    def __init__(
        self, model_dim: int, max_seq_len: int, vocab_info: VocabularyInfo
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param max_seq_len:
            The maximum length of sequences produced by the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(max_seq_len, vocab_info)

        self.model_dim = model_dim

    @override
    def forward(self, batch):
        decoder_output, decoder_padding_mask = self.decode(
            batch.seqs, batch.padding_mask
        )

        return self.project(decoder_output, decoder_padding_mask)

    @abstractmethod
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        *args,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Decode the specified sequences.

        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is
              the batch size, :math:`S` is the target sequence length, and
              :math:`M` is the dimensionality of the model.
            - The padding mask of the decoder output. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the target
              sequence length.
        """

    @abstractmethod
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ):
        """Produce logits for next-step prediction.

        :param decoder_output:
            The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is the
            batch size, :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the model.
        :param decoder_padding_mask:
            The padding mask of the decoder output. *Shape:* :math:`(N,S)`,
            where :math:`N` is the batch size and :math:`S` is the sequence
            length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class TransformerDecoderModel(DecoderModel):
    """Represents a Transformer-based decoder model."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        max_seq_len: int,
        vocab_info: VocabularyInfo,
    ) -> None:
        """
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs.
        :param max_seq_len:
            The maximum length of sequences produced by the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(decoder.model_dim, max_seq_len, vocab_info)

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

    @override
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        *args,
        **kwargs
        # state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Tensor]:
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        decoder_output, decoder_padding_mask = self.decoder(
            seqs, padding_mask, state_bag=state_bag
        )

        return decoder_output, decoder_padding_mask

    @override
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ):
        logits = self.final_proj(decoder_output)

        return logits, self.vocab_info.pad_idx
        # return SequenceModelOutput(logits, self.vocab_info.pad_idx)

