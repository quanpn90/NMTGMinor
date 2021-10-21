import onmt


def split_line_by_char(line, word_list=["<unk>"]):
    chars = list()

    words = line.strip().split()

    for i, word in enumerate(words):
        if word in word_list:
            chars.append(word)
        else:
            for c in word:
                chars.append(c)

        if i < (len(words) - 1):
            chars.append(' ')

    return chars


class Tokenizer(object):

    def __init__(self, input_type='word', lower=False):
        self.input_type = input_type
        self.lower = lower

    def __call__(self, sentence):

        return self.tokenize(sentence)

    def tokenize(self, sentence):
        if self.input_type == "word":
            tokens = sentence.strip().split()
        elif self.input_type == "char":
            tokens = split_line_by_char(sentence)
        else:
            raise NotImplementedError("Input type not implemented")

        return tokens


FAIRSEQ_LANGUAGE_CODES = ["ar_AR",
                          "cs_CZ",
                          "de_DE",
                          "en_XX",
                          "es_XX",
                          "et_EE",
                          "fi_FI",
                          "fr_XX",
                          "gu_IN",
                          "hi_IN",
                          "it_IT",
                          "ja_XX",
                          "kk_KZ",
                          "ko_KR",
                          "lt_LT",
                          "lv_LV",
                          "my_MM",
                          "ne_NP",
                          "nl_XX",
                          "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR",
                          "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL",
                          "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK",
                          "xh_ZA", "gl_ES", "sl_SI"]


class HuggingFaceTokenizer(object):

    def __init__(self, pretrained_tokenizer):

        if pretrained_tokenizer == 'facebook/mbart-large-50':
            from transformers import MBart50TokenizerFast
            tokenizer_ = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX")

        else:
            raise NotImplementedError

        self._tokenizer = tokenizer_

    def tokenize(self, text, src_lang=None):

        if src_lang is not None:
            found = False
            for lang in FAIRSEQ_LANGUAGE_CODES:
                if lang[:2] == src_lang:
                    self._tokenizer.src_lang = lang
                    found = True
                    break

        if not found:
            print("Language code %s not found" % lang)
            raise NotImplementedError

        # add special tokens, etc
        tensor = self._tokenizer(text)['input_ids']

        # convert back to text
        tokens = self._tokenizer.convert_ids_to_tokens(tensor, skip_special_tokens=False)

        return tokens
