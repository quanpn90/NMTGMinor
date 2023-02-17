import onmt
import onmt.modules
from collections import defaultdict


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

    def set_language(self, input_language, output_language, language_code_system="mbart50"):

        if language_code_system == "mbart50":
            language_map_dict = {"en": "en_XX", "de": "de_DE", "fr": "fr_XX", "es": "es_XX",
                                 "pt": "pt_XX", "it": "it_IT", "nl": "nl_XX"}

        else:
            language_map_dict = defaultdict(lambda self, missing_key: missing_key)

        input_lang = language_map_dict[input_language]
        output_lang = language_map_dict[output_language]

        self.translator.change_language(new_src_lang=input_lang, new_tgt_lang=output_lang)

    def translate(self, input, prefix):
        """
        Args:
            input: audio segment (torch.Tensor)

        Returns:

        """

        # 2 list because the translator is designed to run with 1 audio and potentially 1 text
        src_batches = [[input]]  # ... about the input

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
            prefix=prefix)

        external_tokenizer = self.translator.external_tokenizer

        return get_sentence_from_tokens(pred_batch[0][0], pred_ids[0][0], "word", external_tokenizer)

    def translate_batch(self, inputs, prefixes):
        """
        Args:
            inputs: list of audio tensors
            prefixes: list of prefixes

        Returns:

        """
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

        return outputs
