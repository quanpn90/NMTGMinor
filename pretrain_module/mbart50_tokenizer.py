from transformers import MBart50TokenizerFast
import os
from contextlib import contextmanager
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-50-one-to-many-mmt": "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/sentencepiece.bpe.model",
    }
}

FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]


class MultilingualBart50TokenizerFast(MBart50TokenizerFast):
    """
        Construct a MBart50 tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

        This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
        Users should refer to this superclass for more information regarding those methods.

        Args:
            vocab_file (:obj:`str`):
                Path to the vocabulary file.
            src_lang (:obj:`str`, `optional`):
                A string representing the source language.
            tgt_lang (:obj:`str`, `optional`):
                A string representing the target language.
            eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
                The end of sequence token.
            sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
            cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
            unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
            pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            sp_model_kwargs (:obj:`dict`, `optional`):
                Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
                <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:

                - ``enable_sampling``: Enable subword regularization.
                - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.

                  - ``nbest_size = {0,1}``: No sampling is performed.
                  - ``nbest_size > 1``: samples from the nbest_size results.
                  - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                    using forward-filtering-and-backward-sampling algorithm.

                - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
                  BPE-dropout.

        Examples::

            >>> from transformers import MBart50Tokenizer
            >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
            >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
            >>> tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"
            >>> model_inputs = tokenizer(src_text, return_tensors="pt")
            >>> with tokenizer.as_target_tokenizer():
            ...    labels = tokenizer(tgt_text, return_tensors="pt").input_ids
            >>> # model(**model_inputs, labels=labels) should work
        """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []
