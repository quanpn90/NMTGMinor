# coding=utf-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
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

"""PyTorch REFORMER model. Currently not working and might not worth trying for ASR"""

import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function


class ReverseSort(Function):
    """
        After chunked attention is applied which sorted clusters,
        original ordering has to be restored.
        Since customized backward function is used for Reformer (because of reversible network)
        the gradients of the output vectors have to be explicitely
        sorted here.

        Implementation note for myself: the number of forward arguments (except ctx) needs to match the backward outputs
        the number of backward arguments (except ctx) must match the forward output
    """

    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx):
        # save sorted_bucket_idx for backprop
        with torch.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx

            # undo sort to have correct order for next layer
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        # get parameters saved in ctx
        sorted_bucket_idx = ctx.sorted_bucket_idx

        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        # reverse sort of forward
        grad_out_vectors = torch.gather(grad_out_vectors, 2, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 2, sorted_bucket_idx)

        # return grad and `None` fillers for last 2 forward args
        return grad_out_vectors, grad_logits, None, None


class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """ Used to implement attention between consecutive chunks.
            Args:
                vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
                num_chunks_before: chunks before current chunk to include in attention
                num_chunks_after: chunks after current chunk to include in attention
            Returns:
                tensor of shape [num_chunks, N * chunk_length, ...], where
                N = (1 + num_chunks_before + num_chunks_after).
        """

        # if we don't look at the previous chunk or next chunk then return
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                # if i > 0:
                # the first term is all chunks from i. the second term is the last i chunks

                # if i < 0:
                # the first term is the last |i| chunks. the second term is all chunks before |i|
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))

        return torch.cat(slices, dim=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
            splits hidden_size dim into attn_head_size and num_attn_heads
        """

        # remove the last dim (hidden size) and add two more dims
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)

        # output size : [bsz x num_heads x seq_len x d_head]
        return x.transpose(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
            merges attn_head_size dim and num_attn_heads dim into hidden_size
        """

        # x should have size: batch_size * n_heads * length * head_size
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        """
            splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """

        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError("Input vector rank should be one of [3, 4], but is: {}".format(len(vectors.shape)))


class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, opt):

        super().__init__()
        self.opt = opt

        self.chunk_length = opt.chunk_length
        self.num_hashes = opt.num_hashes
        self.num_buckets = None

        self.num_chunks_before = opt.lsh_num_chunks_before
        self.num_chunks_after = opt.lsh_num_chunks_after

        self.dropout = opt.attn_dropout
        self.n_heads = opt.n_heads
        self.model_size = opt.model_size

        self.d_head = self.model_size // self.n_heads
        self.query_key = nn.Linear(self.model_size, self.model_size, bias=False)
        self.values = nn.Linear(self.model_size, self.model_size, bias=False)
        self.value_out = nn.Linear(self.model_size, self.model_size, bias=False)

    def _set_num_buckets(self, seq_len):

        # num buckets should be set to 2 * seq_len // chunk_length (recommended in paper)
        num_buckets_pow_2 = (2 * (seq_len // self.chunk_length)).bit_length() - 1
        num_buckets = 2 ** num_buckets_pow_2

        self.num_buckets = num_buckets
        return num_buckets

    def _hash_vectors(self, vectors, num_hashes, attn_mask):
        batch_size = vectors.shape[0]

        assert self.num_buckets % 2 == 0
        rotation_size = self.num_buckets
        num_buckets = self.num_buckets

        # remove the gradients, but why?
        vectors = vectors.detach()

        # [n_head x d_head x num_hashes x rotation_size//2]
        rotations_shape = (self.n_heads, vectors.shape[-1], num_hashes, rotation_size // 2)

        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)

        # rotated vectors: [bsz x n_head x num_hashes x seq_len x num_buckets/2]
        rotated_vectors = torch.einsum('bhtd,hdnr->bhntr', vectors, random_rotations)

        # TODO: understand why they only randomize for half of the buckets and take the negative for the other half
        rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        buckets = torch.argmax(rotated_vectors, dim=-1)

        # add an extra bucket for padding tokens only
        if attn_mask is not None:
            num_buckets = num_buckets + 1
            # assign padding tokens extra bucket

            buckets_mask = attn_mask.to(torch.uint8)[:, None, None, :].expand(buckets.shape)
            buckets = torch.where(buckets_mask, buckets,
                                  torch.Tensor(num_buckets-1, dtype=torch.long, device=buckets.device))

        # buckets is now [bsz x n_head x num_hashes x seq_len]
        # next we add offset so that bucket numbers from different hashing rounds don't overlap
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view(1, 1, -1, 1)

        # expand to batch_size and n_head
        offsets = offsets.expand((batch_size, self.n_heads) + offsets.shape[-2:])
        offsets_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offsets_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, seq_len, buckets, num_hashes):

        with torch.no_grad():
            # buckets should have size [bsz x nhead x (num_hashes * seq_len)]
            batch_size = buckets.shape[0]

            # arange and expand to get the original indices
            orig_indices = torch.arange(num_hashes * seq_len, device=buckets.device).view(1, 1, -1)
            orig_indices = orig_indices.expand(batch_size, self.n_heads, orig_indices.shape[-1])

            # scale buckets
            # why do we have to scale the buckets ???
            scaled_buckets = seq_len * buckets + (orig_indices % seq_len)
            scaled_buckets = scaled_buckets.detach()

            # hash-based sort
            # this should have size [bsz x nhead
            sorted_bucket_idx = torch.argsort(scaled_buckets, dim=-1)

            # create simple indices to scatter to, to have undo sort
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            # get undo sort
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
            expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)

    def _compute_attn_mask(self, query_indices, key_indices, attn_mask, query_key_dot_shape, seq_len):

        # attention mask for LSH
        if attn_mask is not None:
            # if chunked attention, the mask has to correspond to LSH order
            assert attn_mask.dim() == 2
            if seq_len > self.chunk_length:

                if attn_mask.dim() < 3:
                    attn_mask = attn_mask.unsqueeze(1)  # [ batch_size, 1, seq_len ]
                attn_mask = attn_mask.expand(query_indices.shape[:-1] + (-1, ))
                attn_mask = torch.gather(attn_mask, -1, key_indices)

            attn_mask = attn_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # we don't really need causal mask

        return

    def _attend(self, query_vectors, key_vectors, value_vectors, sorted_bucket_idx_per_hash,
                attention_mask, seq_len):

        # look at previous and following chunks if chunked attention
        if self.chunk_length < seq_len:
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits from dot-product
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free mem
        # del query_vectors, key_vectors

        # if chunked attention split bucket ids to query and key
        if self.chunk_length < seq_len:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash, -1, self.chunk_length, self.n_heads
            )

            key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        # get correct mask values during on precision
        mask = self._compute_attn_mask(query_bucket_idx, key_value_bucket_idx, attn_mask, query_key_dots.shape, seq_len)

        # # apply self-mask
        # # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # # " While attention to the future is not allowed, typical implementations of the
        # # Transformer do allow a position to attend to itself.
        # # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # # of a query vector with itself will almost always be greater than the dot product of a
        # # query vector with a vector at another position. We therefore modify the masking
        # # to forbid a token from attending to itself, except in situations
        # # where a token has no other valid attention targets (e.g. the first token in a sequence) "
        # self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)) \
        #     .to(query_bucket_idx.device)

        if mask is not None:
            query_key_dots = query_key_dots.float().masked_fill_(mask, -float('inf')).type_as(query_key_dots)

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attn_probs = torch.exp(query_key_dots - logits)

        # free mem
        # del query_key_dots

        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        # attend values
        out_vectors = torch.matmul(attn_probs, value_vectors)

        # free memory
        # del value_vectors

        # merge chunk
        if self.chunk_length < seq_len:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attn_probs


    def forward(self, x, attn_mask, buckets=None, **kwargs):
        """
        :param x: hidden states / embeddings of the previous layer [bsz x seq_len x hidden_size]
        :param attn_mask: attention mask
        :param buckets: TODO: findout about buckets
        :param kwargs:
        :return:
        """
        batch_size, seq_len = x.size(0), x.size(1)

        if attn_mask is not None:
            print(attn_mask.shape)
            if len(attn_mask.shape) == 3:
                attn_mask = attn_mask.squeeze(-1)

        num_hashes = self.num_hashes

        query_key = self.query_key(x)
        values = self.values(x)

        # del x

        query_key = self._split_hidden_size_dim(query_key, self.n_heads, self.d_head)
        values = self._split_hidden_size_dim(values, self.n_heads, self.d_head)

        # LSTM attention only makes sense if chunked attention should be performed
        if self.chunk_length < seq_len:

            # hash
            num_buckets = self._set_num_buckets(seq_len)
            if buckets is None:
                buckets = self._hash_vectors(query_key, num_hashes, attn_mask)

            assert (
                    int(buckets.shape[-1]) == num_hashes * seq_len
            ), "last dim of buckets is {}, but should be {}".format(buckets.shape[-1], num_hashes * seq_len)

            sorted_bucket_idx, undo_sorted_bucket_idx = \
                self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(seq_len, buckets, num_hashes)

            # make sure that bucket_idx is not longer than seq_len
            sorted_bucket_idx_per_hash = sorted_bucket_idx % seq_len

            # cluster query key vectors according to hashed buckets:
            query_key_vectors = self._gather_by_expansion(query_key, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(values, sorted_bucket_idx_per_hash, num_hashes)

            query_key_vectors = self._split_seq_length_dim_to(query_key_vectors, -1, self.chunk_length,
                                                              self.n_heads, self.d_head)
            value_vectors = self._split_seq_length_dim_to(value_vectors, -1, self.chunk_length,
                                                          self.n_heads, self.d_head)

        else:
            sorted_bucket_idx_per_hash = torch.arange(seq_len, device=query_key.device).repeat(
                batch_size, self.n_heads, 1
            )

            query_key_vectors = query_key
            value_vectors = values

        # scale the key vectors
        key_vectors = query_key_vectors * (self.d_head ** -0.5)

        # get attention probabilities
        out_vectors, logits, attention_probs = self._attend(
            query_key_vectors,
            key_vectors,
            value_vectors,
            sorted_bucket_idx_per_hash,
            attn_mask,
            seq_len=seq_len
        )

        # free memory
        # del query_key_vectors, key_vectors, value_vectors

        # re-order out-vectors and logits
        if self.chunk_length < seq_len:

            # sort the clusters back to correct ordering
            # out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)

            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(out_vectors, num_hashes, seq_len, self.n_heads, self.d_head)
                logits = self._split_seq_length_dim_to(logits, num_hashes, seq_len, self.n_heads, self.d_head)\
                    .unsqueeze(-1)

                prob_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * prob_vectors, dim=2)

                # free mem
                # del prob_vectors

        # del logits

        assert out_vectors.shape == (
            batch_size,
            self.n_heads,
            seq_len,
            self.d_head
        ), "out_vectors  have be of shape `[batch_size, n_head, seq_len, d_head]`."

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.n_heads, seq_len, self.d_head)

        out_vectors = self.value_out(out_vectors)

        return out_vectors, attention_probs

