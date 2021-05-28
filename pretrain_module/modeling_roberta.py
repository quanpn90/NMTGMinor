# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch RoBERTa model. """



import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_roberta import RobertaConfig
from .modeling_bert import BertModel


_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "distilroberta-base",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]

# consistant with Fairseq
class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.max_position_id = config.max_position_embeddings
        self.bert_word_dropout = config.bert_word_dropout
        self.emb_dropout = nn.Dropout(config.bert_emb_dropout)
        self.emb_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, no_emb_offset=False):

        # ther is no offset for Zh pretrained model
        seq_length = input_ids.size(1)
        if no_emb_offset:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            if seq_length > self.max_position_id:
                position_ids = torch.clamp(position_ids, 0, self.max_position_id - 1)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        else:
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            if seq_length > self.max_position_id:
                position_ids = torch.clamp(position_ids, 0, self.max_position_id - 1)
        position_embeddings = self.position_embeddings(position_ids)

        if inputs_embeds is None:
            embed = self.word_embeddings
            if self.bert_word_dropout and self.training:
                mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
                    1 - self.bert_word_dropout). \
                           expand_as(embed.weight) / (1 - self.bert_word_dropout)

                masked_embed_weight = mask * embed.weight
            else:
                masked_embed_weight = embed.weight

        padding_idx = embed.padding_idx
        words_embeddings = F.embedding(
            input_ids, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.emb_layernorm(embeddings)
        embeddings = self.emb_dropout(embeddings)

        return embeddings

    def emb_step(self, tgt_len, input_ids, token_type_ids=None):
        position_ids = torch.tensor(tgt_len-1, dtype=torch.long, device=input_ids.device)
        if tgt_len > self.max_position_id:
            position_ids = torch.tensor(self.max_position_id-1, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embed = self.word_embeddings
        masked_embed_weight = embed.weight
        padding_idx = embed.padding_idx

        words_embeddings = F.embedding(
            input_ids, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

        # words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.emb_layernorm(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config,
                 bert_word_dropout=None,
                 bert_emb_dropout=None,
                 bert_atten_dropout=None,
                 bert_hidden_dropout=None,
                 bert_hidden_size=None,
                 is_decoder=False,
                 before_plm_output_ln=False,
                 gradient_checkpointing=False,
                 **kwargs,
                 ):

        super().__init__(config, bert_word_dropout,
                         bert_emb_dropout,
                         bert_atten_dropout,
                         bert_hidden_dropout,
                         bert_hidden_size,
                         is_decoder,
                         before_plm_output_ln,
                         gradient_checkpointing,
                         **kwargs
                         )

        # replace the original bert embedding with roberta embedding
        roberta_embeddings = RobertaEmbeddings(config)
        self.add_module('embeddings', roberta_embeddings)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx

