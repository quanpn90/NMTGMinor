import os
import onmt
import onmt.modules
from collections import defaultdict
try:
    from mosestokenizer import MosesDetokenizer, MosesTokenizer
except ImportError:
    # print("[WARNING] Moses tokenizer is not installed. Models with 'detokenize' option won't have Moses-detokenized outputs")
    MosesDetokenizer = None
    MosesTokenizer = None
import torch

from onmt.data.audio_utils import safe_readaudio, wav_to_fmel

class TranslatorParameter(object):

    def __init__(self, filename):

        self.model = ""
        self.src = "<stdin>"
        self.src_img_dir = ""
        self.tgt = ""
        self.output = "<stdout>"
        self.beam_size = 1
        self.batch_size = 1
        self.max_sent_length = 512
        self.min_sent_length = 1
        self.dump_beam = ""
        self.n_best = self.beam_size
        self.replace_unk = False
        self.gpu = -1
        self.cuda = 0
        self.verbose = False
        self.normalize = True

        self.beta = 0.0
        self.alpha = 0.0
        self.start_with_bos = False
        self.fp16 = False
        self.bf16 = False
        self.ensemble_op = 'mean'
        self.autoencoder = None
        self.encoder_type = 'text'
        self.lm = None

        self.src_lang = 'src'
        self.tgt_lang = 'tgt'
        self.bos_token = onmt.constants.BOS_WORD
        self.sampling = False
        self.attributes = None
        self.no_bos_gold = False
        self.no_repeat_ngram_size = 0
        self.no_buffering = False
        self.src_align_right = False
        self.dynamic_quantile = 0
        self.vocab_list = ""

        self.sub_model = ""
        self.sub_src = ""
        self.ensemble_weight = ""
        self.fast_translate = True
        self.vocab_id_list = None  # to be added if necessary

        self.pretrained_classifier = None
        self.detokenize = False
        self.external_tokenizer = "facebook/mbart-large-50"
        self.force_bos = False
        self.use_tgt_lang_as_source = False
        self.anti_prefix = ""
        self.min_filter = 0

        self.num_mel_bin = 0

        self.read_file(filename)


    def read_file(self, filename):

        f = open(filename)

        line = f.readline()

        while line:

            w = line.strip().split()

            if w[0] == "model":
                self.model = w[1]
            elif w[0] == "beam_size":
                self.beam_size = int(w[1])
                self.n_best = self.beam_size
            elif w[0] == "src_lang":
                self.src_lang = w[1]
            elif w[0] == "tgt_lang":
                self.tgt_lang = w[1]
            elif w[0] == "no_repeat_ngram_size":
                self.no_repeat_ngram_size = int(w[1])
            elif w[0] == "dynamic_quantile":
                self.dynamic_quantile = int(w[1])
            elif w[0] == "fp16":
                self.fp16 = True
            elif w[0] == "bf16":
                self.fp16 = True
            elif w[0] == "gpu":
                self.gpu = int(w[1])
                self.cuda = True
            elif w[0] == "detokenize":
                self.detokenize = True
            elif w[0] == "vocab_list":
                self.vocab_list = w[1]
            elif w[0] == "facebook/mbart-large-50":
                self.external_tokenizer = w[1]
            elif w[0] == "force_bos":
                self.force_bos = True
            elif w[0] == "use_tgt_lang_as_source":
                self.use_tgt_lang_as_source = True
            elif w[0] == "max_sent_length":
                self.max_sent_length = int(w[1])
            elif w[0] == "min_sent_length":
                self.min_sent_length = int(w[1])
            elif w[0] == "anti_prefix":
                self.anti_prefix = w[1]
            elif w[0] == "num_mel_bin":
                self.num_mel_bin = int(w[1])
            elif w[0] == "min_filter":
                self.min_filter = int(w[1])

            line = f.readline()


class RecognizerParameter(TranslatorParameter):

    def __init__(self, filename):
        super(RecognizerParameter, self).__init__(filename)

        # Lazy version of this

        self.src_lang = '<s>'
        self.tgt_lang = '<s>'
        self.bos_token = '<s>'

        self.external_tokenizer = "facebook/mbart-large-50"
        self.asr_format = "wav"
        self.encoder_type = "audio"



class OnlineTranslator(object):
    def __init__(self, model):
        opt = TranslatorParameter(model)
        from onmt.inference.fast_translator import FastTranslator
        self.translator = FastTranslator(opt)

        self.src_lang = "en"
        self.tgt_lang = "en"
        self.detokenize = opt.detokenize
        self.external_tokenizer = opt.external_tokenizer
        self.anti_prefix = opt.anti_prefix

    # def translate(self, input):
    #     predBatch, predScore, predLength, goldScore, numGoldWords, allGoldScores = \
    #         self.translator.translate([input.split()], [])
    #
    #     return " ".join(predBatch[0][0])

        self.use_tgt_lang_as_source = opt.use_tgt_lang_as_source


    def set_language(self, input_language, output_language, language_code_system="mbart50"):

        # override the input_language
        if self.use_tgt_lang_as_source:
            input_language = output_language

        if language_code_system == "mbart50":
            language_map_dict = {"en": "en_XX", "de": "de_DE", "fr": "fr_XX", "es": "es_XX",
                                 "pt": "pt_XX", "it": "it_IT", "nl": "nl_XX", "None": "<s>",
                                 "ja": "ja_XX", "zh": "zh_CN", "vn": "vi_VN"}

        else:
            language_map_dict = defaultdict(lambda self, missing_key: missing_key)

        input_lang = language_map_dict[input_language]
        output_lang = language_map_dict[output_language]

        self.translator.change_language(new_src_lang=input_lang, new_tgt_lang=output_lang, use_srclang_as_bos=False)

        self.src_lang = input_language
        self.tgt_lang = output_language

    def translate(self, input, prefix, memory=None):
        """
        Args:
            prefix:
            input: audio segment (torch.Tensor)

        Returns:

        """
        input = input.strip().split()

        if self.detokenize:
            prefixes = []
            for _prefix in prefix:
                if _prefix is not None:
                    with MosesTokenizer(self.tgt_lang) as tokenize:
                        __prefix = tokenize(_prefix)
                        __prefix = " ".join(__prefix)
                        _prefix = __prefix
                prefixes.append(_prefix)
            prefix = prefixes

        # 2 lists because the translator is designed to run with 1 audio and potentially 1 text
        src_batches = [[input]]  # ... about the input

        tgt_batch = []
        sub_src_batch = []
        past_src_batches = []

        if all(v is None for v in prefix):
            prefix = None

        anti_prefix = self.anti_prefix if len(self.anti_prefix) > 0 else None

        # perform beam search in the model
        pred_batch, pred_ids, pred_score, pred_pos_scores, pred_length, \
        gold_score, num_gold_words, all_gold_scores = self.translator.translate(
            src_batches, tgt_batch,
            prefix=prefix, anti_prefix=anti_prefix)

        # use the external sentencepiece model
        external_tokenizer = self.translator.external_tokenizer

        output_sentence = get_sentence_from_tokens(pred_batch[0][0], pred_ids[0][0], "word", external_tokenizer)

        # here if we want to use mosestokenizer, probably we need to split the sentence AFTER the sentencepiece/bpe
        # model applies their de-tokenization
        if self.detokenize and MosesDetokenizer is not None:
            output_sentence_parts = output_sentence.split()
            with MosesDetokenizer(self.tgt_lang) as detokenize:
                output_sentence = detokenize(output_sentence_parts)

        return output_sentence

    def translate_batch(self, inputs, prefixes, memory=None):
        """
        Args:
            inputs: list of audio tensors
            prefixes: list of prefixes
            memory:

        Returns:

        """
        inputs = [_input.strip().split() for _input in inputs]

        if self.detokenize:
            new_prefixes = []
            for _prefix in prefixes:
                if _prefix is not None:
                    with MosesTokenizer(self.tgt_lang) as tokenize:
                        tokenized_sentence = tokenize(_prefix)
                        tokenized_sentence = " ".join(tokenized_sentence)
                        _prefix = tokenized_sentence
                new_prefixes.append(_prefix)
            prefixes = new_prefixes

        # 2 list because the translator is designed to run with 1 audio and potentially 1 text
        src_batches = [inputs]  # ... about the input

        tgt_batch = []
        sub_src_batch = []
        past_src_batches = []

        if all(v is None for v in prefixes):
            prefixes = None

        anti_prefix = self.anti_prefix if len(self.anti_prefix) > 0 else None

        pred_batch, pred_ids, pred_score, pred_length, \
        gold_score, num_gold_words, all_gold_scores = self.translator.translate(
            src_batches, tgt_batch,
            prefix=prefixes, anti_prefix=anti_prefix)

        external_tokenizer = self.translator.external_tokenizer

        outputs = list()

        for pred, pred_id in zip(pred_batch, pred_ids):
            outputs.append(get_sentence_from_tokens(pred[0], pred_id[0], "word", external_tokenizer))

        if self.detokenize and MosesDetokenizer is not None:
            outputs_detok = []
            for output_sentence in outputs:
                # here if we want to use mosestokenizer, probably we need to split the sentence AFTER the sentencepiece/bpe
                # model applies their de-tokenization
                output_sentence_parts = output_sentence.split()
                with MosesDetokenizer(self.tgt_lang) as detokenize:
                    output_sentence = detokenize(output_sentence_parts)
                outputs_detok.append(output_sentence)
            return outputs_detok

        return outputs

# Checklist to integrate:

# 1. model file (model.averaged.pt)
# 2. the w2v and mbart50 config
# 3. mbart50 tokenizer
# 4. interface to translate with

def get_sentence_from_tokens(tokens, ids, input_type, external_tokenizer=None):
    if external_tokenizer is None:
        if input_type == 'word':
            sent = " ".join(tokens)
        elif input_type == 'char':
            sent = "".join(tokens)
        else:
            raise NotImplementedError

    else:
        sent = external_tokenizer.decode(ids, True, True).strip()

    return sent


class ASROnlineTranslator(object):

    def __init__(self, model):
        opt = RecognizerParameter(model)
        from onmt.inference.fast_translator import FastTranslator
        self.translator = FastTranslator(opt)

        self.src_lang = "en"
        self.tgt_lang = "en"
        self.detokenize = opt.detokenize
        self.anti_prefix = opt.anti_prefix
        self.num_mel_bin = opt.num_mel_bin
        self.min_filter = opt.min_filter

        print(self.num_mel_bin)

    def set_language(self, input_language, output_language, language_code_system="mbart50"):
        # TODO: check if the output language takes the form of "en+de"
        # TODO: in that case, we don't change language but rather use "<s>" and set the vocabulary limit

        if language_code_system == "mbart50":
            language_map_dict = {"en": "en_XX", "de": "de_DE", "fr": "fr_XX", "es": "es_XX",
                                 "pt": "pt_XX", "it": "it_IT", "nl": "nl_XX", "None": "<s>",
                                 "zh": "zh_CN", "ja": "ja_XX", "uk": "uk_UA"}

        else:
            language_map_dict = defaultdict(lambda self, missing_key: missing_key)

        # TODO:

        output_languages = output_language.split("+")
        input_languages = input_language.split("+")

        if len(output_languages) == 1 and len(input_languages) == 1:
            input_lang = language_map_dict[input_language]
            output_lang = language_map_dict[output_language]

            self.translator.change_language(new_src_lang=input_lang, new_tgt_lang=output_lang)

            self.src_lang = input_language
            self.tgt_lang = output_language

            self.translator.set_filter([])

        else:
            # here we have to handle multiple languages

            # TODO: should we set <s> or lang_map_dict["None"]
            input_lang = "<s>"
            output_lang = "<s>"

            self.translator.change_language(new_src_lang=input_lang, new_tgt_lang=output_lang)

            self.src_lang = input_language
            self.tgt_lang = output_language

            # but also we need to set the vocab ids restriction here ... based on some kind of list
            vocab_id_files = list()

            for _lang in output_languages:
                _lang_mapped = language_map_dict[_lang]
                vocab_id_file = os.path.join("vocabs", "%s.vocabids" % _lang_mapped)

                if os.path.exists(vocab_id_file):
                    vocab_id_files.append(vocab_id_file)

            print(vocab_id_files)

            self.translator.set_filter(vocab_id_files, min_occurance=self.min_filter)


    def translate(self, input, prefix, memory=None):
        """
        Args:
            prefix:
            input: audio segment (torch.Tensor)
            memory: for incrementally learning new tokens

        Returns:

        """

        if self.detokenize:
            prefixes = []
            for _prefix in prefix:
                if _prefix is not None:
                    with MosesTokenizer(self.tgt_lang) as tokenize:
                        __prefix = tokenize(_prefix)
                        __prefix = " ".join(__prefix)
                        _prefix = __prefix
                prefixes.append(_prefix)
            prefix = prefixes

        # 2 lists because the translator is designed to run with 1 audio and potentially 1 text
        if self.num_mel_bin > 1:
            input = wav_to_fmel(input, num_mel_bin=self.num_mel_bin)
        src_batches = [[input]]  # ... about the input

        tgt_batch = []


        anti_prefix = self.anti_prefix if len(self.anti_prefix) > 0 else None

        print("prefix", prefix)
        print("anti prefix:", anti_prefix)

        # use the external sentencepiece model
        external_tokenizer = self.translator.external_tokenizer

        if memory is not None and len(memory) > 0:
            memory_text_ids = [torch.as_tensor(external_tokenizer.encode(m)) for m in memory]
            memory = torch.ones(len(memory_text_ids), max(len(x) for x in memory_text_ids), dtype=torch.int64)
            for i, m in enumerate(memory_text_ids):
                memory[i, :len(m)] = m

        # perform beam search in the model
        pred_batch, pred_ids, pred_score, pred_pos_scores,  pred_length, \
        gold_score, num_gold_words, all_gold_scores = self.translator.translate(
            src_batches, tgt_batch, type='asr',
            prefix=prefix, anti_prefix=anti_prefix, memory=memory, input_size=self.num_mel_bin)

        output_sentence = get_sentence_from_tokens(pred_batch[0][0], pred_ids[0][0], "word", external_tokenizer)

        # here if we want to use mosestokenizer, probably we need to split the sentence AFTER the sentencepiece/bpe
        # model applies their de-tokenization
        if self.detokenize and MosesDetokenizer is not None:
            output_sentence_parts = output_sentence.split()
            with MosesDetokenizer(self.tgt_lang) as detokenize:
                output_sentence = detokenize(output_sentence_parts)

        # print(pred_ids[0][0], output_sentence)

        bpe_output = pred_batch[0][0]
        scores = pred_pos_scores[0][0]

        return output_sentence, bpe_output, scores

    def translate_batch(self, inputs, prefixes, memory=None):
        """
        Args:
            inputs: list of audio tensors
            prefixes: list of prefixes
            memory

        Returns:

        """

        if self.detokenize:
            new_prefixes = []
            for _prefix in prefixes:
                if _prefix is not None:
                    with MosesTokenizer(self.tgt_lang) as tokenize:
                        tokenized_sentence = tokenize(_prefix)
                        tokenized_sentence = " ".join(tokenized_sentence)
                        _prefix = tokenized_sentence
                new_prefixes.append(_prefix)
            prefixes = new_prefixes

        # 2 list because the translator is designed to run with 1 audio and potentially 1 text
        if self.num_mel_bin > 0:
            inputs = [wav_to_fmel(input, num_mel_bin=self.num_mel_bin) for input in inputs]
        src_batches = [inputs]  # ... about the input

        tgt_batch = []
        sub_src_batch = []
        past_src_batches = []

        # pred_score, pred_length, gold_score, num_gold_words, all_gold_scores = self.translator.translate(
        #     src_batches, tgt_batch,
        #     type='asr',
        #     prefix=prefix)

        anti_prefix = self.anti_prefix if len(self.anti_prefix) > 0 else None

        pred_batch, pred_ids, pred_score, pred_pos_scores, pred_length,  \
        gold_score, num_gold_words, all_gold_scores = self.translator.translate(
            src_batches, tgt_batch, type='asr',
            prefix=prefixes, anti_prefix=anti_prefix, input_size=self.num_mel_bin)

        external_tokenizer = self.translator.external_tokenizer

        outputs = list()

        for pred, pred_id in zip(pred_batch, pred_ids):
            outputs.append(get_sentence_from_tokens(pred[0], pred_id[0], "word", external_tokenizer))

        if self.detokenize and MosesDetokenizer is not None:
            outputs_detok = []
            for output_sentence in outputs:
                # here if we want to use mosestokenizer, probably we need to split the sentence AFTER the sentencepiece/bpe
                # model applies their de-tokenization
                output_sentence_parts = output_sentence.split()
                with MosesDetokenizer(self.tgt_lang) as detokenize:
                    output_sentence = detokenize(output_sentence_parts)
                outputs_detok.append(output_sentence)
            return outputs_detok

        bpe_outputs = list()

        for pred in pred_batch:
            bpe_outputs.append(pred[0])

        score_outputs = list()

        for pred_scores in pred_pos_scores:
            score_outputs.append(pred_scores[0])

        # print(pred, outputs)

        return outputs, bpe_outputs, score_outputs
