import torch
import os
from contextlib import contextmanager
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from transformers.tokenization_utils import AddedToken,  BatchEncoding, PreTrainedTokenizer

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-50-one-to-many-mmt": "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-50-one-to-many-mmt": 1024,
}

class DeltaLMTokenizer(PreTrainedTokenizer):
    """
        Construct a MBart50 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).
        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.
        Users should refer to this superclass for more information regarding those methods.
        Args:
            vocab_file (`str`):
                Path to the vocabulary file.
            src_lang (`str`, *optional*):
                A string representing the source language.
            tgt_lang (`str`, *optional*):
                A string representing the target language.
            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.
            sep_token (`str`, *optional*, defaults to `"</s>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
            cls_token (`str`, *optional*, defaults to `"<s>"`):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            sp_model_kwargs (`dict`, *optional*):
                Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:
                - `enable_sampling`: Enable subword regularization.
                - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
                  - `nbest_size = {0,1}`: No sampling is performed.
                  - `nbest_size > 1`: samples from the nbest_size results.
                  - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                    using forward-filtering-and-backward-sampling algorithm.
                - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
                  BPE-dropout.
        Examples:
        ```python
        >>> from transformers import MBart50Tokenizer
        >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
        >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
        >>> tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> model_inputs = tokenizer(src_text, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...    labels = tokenizer(tgt_text, return_tensors="pt").input_ids
        >>> # model(**model_inputs, labels=labels) should work
        ```"""

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
            self,
            vocab_file,
            src_lang=None,
            tgt_lang=None,
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", [])
        # kwargs["additional_special_tokens"] += [
        #     code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        # ]

        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1

        self.sp_model_size = len(self.sp_model)
        # self.lang_code_to_id = {
        #     code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        # }
        # self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        # self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset
        #
        # self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # self._src_lang = "</s>"  # src_lang if src_lang is not None else
        # self.cur_lang_code_id = 2 # self.lang_code_to_id[self._src_lang]
        # self.tgt_lang = tgt_lang
        self._src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + self.fairseq_offset

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        assert new_src_lang in ["src", "tgt"], "DeltaLM tokenizer at the moment only supports src and tgt."
        self._src_lang = new_src_lang

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def get_vocab(self) -> Dict:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        return self.sp_model.decode(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        if self.src_lang in ["<s>", "src"]:
            self.prefix_tokens = []
        elif self.src_lang in ["</s>", "tgt"]:
            self.prefix_tokens = [self.eos_token_id]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos]."""
        # self.cur_lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = [self.eos_token_id]
        self.suffix_tokens = [self.eos_token_id]

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART-50 sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `[src_lang_code] X [eos]`
        - `labels`: (for decoder) `[tgt_lang_code] X [eos]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
            self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
            self,
            src_texts: List[str],
            src_lang: str = "en_XX",
            tgt_texts: Optional[List[str]] = None,
            tgt_lang: str = "ro_RO",
            **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

class MultilingualDeltaLMTokenizer(DeltaLMTokenizer):


    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
            self,
            vocab_file,
            src_lang=None,
            tgt_lang=None,
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> None:
        self.added_tokens_decoder = None
        self.added_tokens_encoder = None
        self.additional_special_tokens = None
        super(MultilingualDeltaLMTokenizer, self).__init__(vocab_file, src_lang=src_lang, tgt_lang=tgt_lang,eos_token=eos_token,
                                                           sep_token=sep_token,cls_token=cls_token,unk_token=unk_token,pad_token=pad_token,
                                                           mask_token=mask_token, sp_model_kwargs=sp_model_kwargs)


    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + self.fairseq_offset

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang

    def override_lang_list(self, lang_list):
        self.additional_special_tokens = lang_list
        start = 250001
        self.added_tokens_encoder.clear()
        self.added_tokens_encoder["<mask>"] = start
        for i, lang in enumerate(lang_list):
            self.added_tokens_encoder[lang] = start + i + 1

        self.added_tokens_decoder.clear()

        for word in self.added_tokens_encoder:
            self.added_tokens_decoder[self.added_tokens_encoder[word]] = word

    def _tokenize(self, text: str) -> List[str]:
        return [self.src_lang] + self.sp_model.encode(text, out_type=str)

    @classmethod
    def from_pretrained(cls, *args, lang_list=[], **kwargs, ):

        tokenizer = super(MultilingualDeltaLMTokenizer, cls).from_pretrained(*args, **kwargs)
        tokenizer.override_lang_list(lang_list)

        return tokenizer