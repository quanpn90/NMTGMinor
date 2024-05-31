import torch
import os
from contextlib import contextmanager
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from transformers.tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from transformers import MBart50TokenizerFast

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
                          "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
                          "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL",
                          "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF",
                          "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA",
                          "gl_ES", "sl_SI"]  # fmt: skip


class MBart50ClusterTokenizer(MBart50TokenizerFast):

    # vocab_files_names = VOCAB_FILES_NAMES
    # model_input_names = ["input_ids", "attention_mask"]
    # slow_tokenizer_class = MBart50Tokenizer
    #
    # prefix_tokens: List[int] = []
    # suffix_tokens: List[int] = []

    def __init__(
            self,
            vocab_file=None,
            src_lang=None,
            tgt_lang=None,
            tokenizer_file=None,
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            num_clusters=4096,
            **kwargs,
    ):
        self.num_clusters = num_clusters

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token,
                                                                                     str) else mask_token

        all_codes = FAIRSEQ_LANGUAGE_CODES

        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        kwargs["additional_special_tokens"] += [
            code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        ]

        for i in range(num_clusters):

            token = "__" + str(i) + "__"
            cluster_token = AddedToken(token, lstrip=True, rstrip=False, single_word=True)

            kwargs["additional_special_tokens"].append(cluster_token)

        # print(kwargs["additional_special_tokens"])

        super().__init__(
            vocab_file,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.vocab_file = vocab_file

        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in all_codes
        }

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.tgt_lang = tgt_lang
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.set_src_lang_special_tokens(self._src_lang)

