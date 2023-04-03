import onmt
import onmt.modules
from collections import defaultdict
try:
    from mosestokenizer import MosesDetokenizer, MosesTokenizer
except ImportError:
    # print("[WARNING] Moses tokenizer is not installed. Models with 'detokenize' option won't have Moses-detokenized outputs")
    MosesDetokenizer = None
    MosesTokenizer = None

class TranslatorParameter(object):

    def __init__(self, filename):

        self.model = ""
        self.src = "<stdin>"
        self.src_img_dir = ""
        self.tgt = ""
        self.output = "<stdout>"
        self.beam_size = 1
        self.batch_size = 1
        self.max_sent_length = 100
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
        self.external_tokenizer = ""
        self.fast_translate = True
        self.vocab_id_list = None  # to be added if necessary

        self.pretrained_classifier = None
        self.detokenize = False

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
            elif w[0] == "gpu":
                self.gpu = int(w[1])
                self.cuda = True
            elif w[0] == "detokenize":
                self.detokenize = True
            elif w[0] == "vocab_list":
                self.vocab_list = w[1]

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

    def translate(self, input):
        predBatch, predScore, predLength, goldScore, numGoldWords, allGoldScores = \
            self.translator.translate([input.split()], [])

        return " ".join(predBatch[0][0])


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

    def set_language(self, input_language, output_language, language_code_system="mbart50"):

        if language_code_system == "mbart50":
            language_map_dict = {"en": "en_XX", "de": "de_DE", "fr": "fr_XX", "es": "es_XX",
                                 "pt": "pt_XX", "it": "it_IT", "nl": "nl_XX", "None": "<s>"}

        else:
            language_map_dict = defaultdict(lambda self, missing_key: missing_key)

        input_lang = language_map_dict[input_language]
        output_lang = language_map_dict[output_language]

        self.translator.change_language(new_src_lang=input_lang, new_tgt_lang=output_lang)

        self.src_lang = input_language
        self.tgt_lang = output_language

    def translate(self, input, prefix):
        """
        Args:
            prefix:
            input: audio segment (torch.Tensor)

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
        src_batches = [[input]]  # ... about the input

        tgt_batch = []
        sub_src_batch = []
        past_src_batches = []

        # perform beam search in the model
        pred_batch, pred_ids, pred_score, pred_length, \
        gold_score, num_gold_words, all_gold_scores = self.translator.translate(
            src_batches, tgt_batch, type='asr',
            prefix=prefix)

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

    def translate_batch(self, inputs, prefixes):
        """
        Args:
            inputs: list of audio tensors
            prefixes: list of prefixes

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
        src_batches = [inputs]  # ... about the input

        tgt_batch = []
        sub_src_batch = []
        past_src_batches = []

        # pred_score, pred_length, gold_score, num_gold_words, all_gold_scores = self.translator.translate(
        #     src_batches, tgt_batch,
        #     type='asr',
        #     prefix=prefix)

        pred_batch, pred_ids, pred_score, pred_length, \
        gold_score, num_gold_words, all_gold_scores = self.translator.translate(
            src_batches, tgt_batch, type='asr',
            prefix=prefixes)

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
