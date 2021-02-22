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
"""PyTorch BERT model. """


import math

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
import numpy as np

from .activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig

from .modeling_outputs import (
    BaseModelOutput,
)
from .modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices
import onmt.constants
from collections import defaultdict

#
# _CONFIG_FOR_DOC = "BertConfig"
# _TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
    print("FusedLayerNorm is available")

except ImportError:
    print("FusedLayerNorm is not available, we use torch.nn.LayerNorm")
    import torch.nn.LayerNorm as BertLayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.max_position_id = config.max_position_embeddings
        self.bert_word_dropout = config.bert_word_dropout
        # print("config.bert_word_dropout", config.bert_word_dropout)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.bert_emb_dropout)
        # print("config.bert_emb_dropout", config.bert_emb_dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, no_emb_offset=False):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        if seq_length > self.max_position_id:
            position_ids = torch.clamp(position_ids, 0, self.max_position_id-1)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # by me
        embed = self.word_embeddings

        # print(embed.weight[onmt.constants.BERT_MASK, :])

        if self.bert_word_dropout and self.training:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - self.bert_word_dropout).\
                       expand_as(embed.weight) / (1 - self.bert_word_dropout)

            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        padding_idx = embed.padding_idx

        words_embeddings = F.embedding(
            input_ids, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def emb_step(self, tgt_len, input_ids, token_type_ids=None):
        position_ids = torch.tensor(tgt_len-1, dtype=torch.long, device=input_ids.device)
        if tgt_len > self.max_position_id:
            position_ids = torch.tensor(self.max_position_id-1, dtype=torch.long, device=input_ids.device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embed = self.word_embeddings
        masked_embed_weight = embed.weight
        padding_idx = embed.padding_idx

        words_embeddings = F.embedding(
            input_ids, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.bert_atten_dropout)
        # print("config.bert_atten_dropout", config.bert_atten_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # hidden_states = hidden_states.transpose(0,1)
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def selfattn_step(self,
                      hidden_states,
                      attention_mask,
                      head_mask,
                      encoder_hidden_states=None,
                      encoder_attention_mask=None,
                      output_attentions=False,
                      buffer=None
                      ):
        # hidden_size -> all_head_size: 767 -> 768
        proj_query = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        # enc_dec_attention
        if encoder_hidden_states is not None:  # use src mask， otherwise use tgt mask
            attention_mask = encoder_attention_mask

            if buffer is not None and 'src_k' in buffer and 'src_v' in buffer:  # no repeated computation needed
                proj_key = buffer['src_k']
                proj_value = buffer['src_v']
            else:
                if buffer is None:
                    buffer = dict()
                proj_key = self.key(encoder_hidden_states)
                proj_value = self.value(encoder_hidden_states)
                buffer['src_k'] = proj_key
                buffer['src_v'] = proj_value

        # decoder self-attention
        else:
            proj_key = self.key(hidden_states)
            proj_value = self.value(hidden_states)
            if buffer is not None and 'k' in buffer and 'v' in buffer:
                proj_key = torch.cat([buffer['k'], proj_key], dim=1)  # time second 和之前time_step的进行拼接
                buffer['k'] = proj_key
                proj_value = torch.cat([buffer['v'], proj_value], dim=1)  # time second
                buffer['v'] = proj_value
            else:
                if buffer is None:
                    buffer = dict()
                buffer['k'] = proj_key   # step为0
                buffer['v'] = proj_value

        query_layer = self.transpose_for_scores(proj_query)
        key_layer = self.transpose_for_scores(proj_key)
        value_layer = self.transpose_for_scores(proj_value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 感觉如果纯粹的encoder 只会用到tuple的第一个元素，如果连接decoder, 则还需要attention_probs
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs, buffer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.bert_hidden_dropout)
        # print("config.bert_hidden_dropout", config.bert_hidden_dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def attn_step(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            buffer=None
            ):
        # self_outputs: (context_layer, attention_probs) or （context_layer, ）

        self_outputs, buffer = self.self.selfattn_step(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            buffer
        )

        # output: BertSelfOutput dropout--> add--> LN
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, buffer


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.bert_hidden_dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def bertlayer_step(
            self,
            hidden_states,
            attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            buffer=None
        ):

        self_attention_outputs, buffer = self.attention.attn_step(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            buffer=buffer
        )
        attention_output = self_attention_outputs[0] # context_layer
        outputs = self_attention_outputs[1:]  # (attention_probs,)add self attentions if we output attention weights

        cross_attention_outputs, buffer = self.crossattention.attn_step(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            buffer=buffer
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        # 1.dropout(intermediate_output) 2. add(attention_output) 3.LN
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs  # 单纯的encoder的时候， outputs是空tuple(), outputs 表示attention
        return outputs, buffer


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 默认返回值， 如果config 没有这个属性，默认返回值
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config,
                 bert_word_dropout=None,
                 bert_emb_dropout=None,
                 bert_atten_dropout=None,
                 bert_hidden_dropout=None,
                 bert_hidden_size=None,
                 is_decoder=False,
                 before_plm_output_ln=False,
                 gradient_checkpointing=False,
                 ):

        super().__init__(config)
        self.config = config
        if bert_word_dropout is not None:
            self.config.bert_word_dropout = bert_word_dropout
        if bert_emb_dropout is not None:
            self.config.bert_emb_dropout = bert_emb_dropout
        if bert_atten_dropout is not None:
            self.config.bert_atten_dropout = bert_atten_dropout
        if bert_hidden_dropout is not None:
            self.config.bert_hidden_dropout = bert_hidden_dropout
        if bert_hidden_size is not None:
            self.config.bert_hidden_size = bert_hidden_size

        self.config.is_decoder = is_decoder
        self.config.before_plm_output_ln = before_plm_output_ln
        self.config.gradient_checkpointing = gradient_checkpointing

        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        
        if self.config.before_plm_output_ln:
            self.before_plm_output_ln = BertLayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        else:
            self.before_plm_output_ln = None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        no_offset=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # [bsz, src_len]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            no_emb_offset=no_offset,
        )  # [bsz, src_len, hidden_dim]
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.before_plm_output_ln is not None:
            sequence_output = self.before_plm_output_ln(encoder_outputs[0])
        else:
            sequence_output = encoder_outputs[0]            

        if not return_dict:
            return (sequence_output, ) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def step(self, input_ids, decoder_state, streaming=False):
        device = input_ids.device

        if input_ids.size(1) > 1:
            input_ = input_ids[:, -1].unsqueeze(1)
        else:
            input_ = input_ids
        tgt_token_type = input_.ne(onmt.constants.TGT_PAD).long()  # [bsz, len]
        data_type = next(self.parameters()).dtype

        src_mask = decoder_state.src_mask.squeeze(1)  # [bsz, all_src_len] 确实要保持src的所有长度，问题是 batch size 要变了
        # print("src_mask", src_mask.size())

        extended_src_mask = self.invert_attention_mask(src_mask)

        mask_tgt = input_ids.ne(onmt.constants.TGT_PAD).byte()
        input_shape = input_ids.size()  # [bsz, sent_len]
        cur_pos = input_shape[-1]
        extended_tgt_mask = self.get_extended_attention_mask(mask_tgt, input_shape, device=device)

        extended_tgt_mask = extended_tgt_mask[:, :, -1, :].unsqueeze(-2)
        encoder_hidden_states = decoder_state.context.transpose(0, 1)  # [b, l, de_model]
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()

        head_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers, data_type)

        if self.dec_pretrained_model == "bert" or self.dec_pretrained_model == "roberta":
            embedding_output = self.embeddings.emb_step(cur_pos, input_, tgt_token_type)
        else:
            print("Warning: check dec_pretrained_model", self.dec_pretrained_model)
            exit(-1)

        hidden_states = embedding_output
        output_attentions = False
        buffers = decoder_state.attention_buffers

        for i, layer in enumerate(self.encoder.layer):
            buffer = buffers[i] if i in buffers else None
            layer_outputs, buffer = layer.bertlayer_step(
                hidden_states,
                extended_tgt_mask,
                head_mask[i],
                encoder_hidden_states,  # decoder_state.context
                extended_src_mask,  # decoder_state.src_mask
                output_attentions,
                buffer
            )

            hidden_states = layer_outputs[0]
            decoder_state.update_attention_buffer(buffer, i)

        output_dict = defaultdict(lambda: None)
        output_dict["hidden"] = hidden_states
        # output_dict["coverage"] = buffers[i]

        # return hidden_states, buffers[i]
        return output_dict

    def renew_buffer(self, new_len):

        # not sure about this
        # self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len+1, new_len+1)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)