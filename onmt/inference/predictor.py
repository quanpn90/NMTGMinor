import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from onmt.model_factory import build_classifier
from ae.Autoencoder import Autoencoder
import torch.nn.functional as F
import sys
from onmt.constants import add_tokenidx
from onmt.options import backward_compatible

model_list = ['transformer', 'stochastic_transformer', 'fusion_network']


class Predictor(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.fp16 = opt.fp16
        self.attributes = opt.attributes  # attributes split by |. for example: de|domain1
        self.src_lang = opt.src_lang
        self.tgt_lang = opt.tgt_lang

        if self.attributes:
            self.attributes = self.attributes.split("|")

        self.models = list()
        self.model_types = list()

        # models are string with | as delimiter
        models = opt.model.split("|")

        print(models)
        self.n_models = len(models)
        self._type = 'text'

        for i, model_path in enumerate(models):
            checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)

            model_opt = checkpoint['opt']
            model_opt = backward_compatible(model_opt)
            if hasattr(model_opt, "enc_state_dict"):
                model_opt.enc_state_dict = None
                model_opt.dec_state_dict = None

            self.main_model_opt = model_opt
            dicts = checkpoint['dicts']

            # update special tokens
            onmt.constants = add_tokenidx(model_opt, onmt.constants, dicts)
            self.bos_token = model_opt.tgt_bos_word

            if i == 0:
                if "src" in checkpoint['dicts']:
                    self.src_dict = checkpoint['dicts']['src']
                else:
                    self._type = "audio"
                    # self.src_dict = self.tgt_dict

                self.tgt_dict = checkpoint['dicts']['tgt']

                print(self.tgt_dict.idxToLabel)

                if "langs" in checkpoint["dicts"]:
                    self.lang_dict = checkpoint['dicts']['langs']

                else:
                    self.lang_dict = {'src': 0, 'tgt': 1}

                # self.bos_id = self.tgt_dict.labelToIdx[self.bos_token]

            model = build_classifier(model_opt, checkpoint['dicts'])
            # optimize_model(model)
            if opt.verbose:
                print('Loading model from %s' % model_path)
            model.load_state_dict(checkpoint['model'])

            if model_opt.model in model_list:
                # if model.decoder.positional_encoder.len_max < self.opt.max_sent_length:
                #     print("Not enough len to decode. Renewing .. ")
                #     model.decoder.renew_buffer(self.opt.max_sent_length)
                model.renew_buffer(self.opt.max_sent_length)

            # model.convert_autograd()

            if opt.fp16:
                model = model.half()

            if opt.cuda:
                model = model.cuda()
            else:
                model = model.cpu()

            if opt.dynamic_quantile == 1:

                engines = torch.backends.quantized.supported_engines
                if 'fbgemm' in engines:
                    torch.backends.quantized.engine = 'fbgemm'
                else:
                    print("[INFO] fbgemm is not found in the available engines. Possibly the CPU does not support AVX2."
                          " It is recommended to disable Quantization (set to 0).")
                    torch.backends.quantized.engine = 'qnnpack'

                # convert the custom functions to their autograd equivalent first
                model.convert_autograd()

                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
                )

            model.eval()

            self.models.append(model)
            self.model_types.append(model_opt.model)

        # language model
        if opt.lm is not None:
            if opt.verbose:
                print('Loading language model from %s' % opt.lm)

            lm_chkpoint = torch.load(opt.lm, map_location=lambda storage, loc: storage)

            lm_opt = lm_chkpoint['opt']

            lm_model = build_language_model(lm_opt, checkpoint['dicts'])

            if opt.fp16:
                lm_model = lm_model.half()

            if opt.cuda:
                lm_model = lm_model.cuda()
            else:
                lm_model = lm_model.cpu()

            self.lm_model = lm_model

        self.cuda = opt.cuda
        self.ensemble_op = opt.ensemble_op

        if opt.autoencoder is not None:
            if opt.verbose:
                print('Loading autoencoder from %s' % opt.autoencoder)
            checkpoint = torch.load(opt.autoencoder,
                                    map_location=lambda storage, loc: storage)
            model_opt = checkpoint['opt']

            # posSize= checkpoint['autoencoder']['nmt.decoder.positional_encoder.pos_emb'].size(0)
            # self.models[0].decoder.renew_buffer(posSize)
            # self.models[0].decoder.renew_buffer(posSize)

            # Build model from the saved option
            self.autoencoder = Autoencoder(self.models[0], model_opt)

            self.autoencoder.load_state_dict(checkpoint['autoencoder'])

            if opt.cuda:
                self.autoencoder = self.autoencoder.cuda()
                self.models[0] = self.models[0].cuda()
            else:
                self.autoencoder = self.autoencoder.cpu()
                self.models[0] = self.models[0].cpu()

            self.models[0].autoencoder = self.autoencoder
        if opt.verbose:
            print('Done')


    def build_asr_data(self, src_data, tgt_sents):
        # This needs to be the same as preprocess.py.

        tgt_data = None
        if tgt_sents:
            tgt_data = [self.tgt_dict.convertToIdx(b,
                                                   onmt.constants.UNK_WORD,
                                                   onmt.constants.BOS_WORD,
                                                   onmt.constants.EOS_WORD) for b in tgt_sents]

        return onmt.Dataset(src_data, tgt_data,
                            batch_size_words=sys.maxsize,
                            data_type=self._type, batch_size_sents=self.opt.batch_size)

    def classify_batch(self, batches, sub_batches=None):

        with torch.no_grad():
            return self._classify_batch(batches, sub_batches=sub_batches)

    def _classify_batch(self, batches, sub_batches):
        batch = batches[0]

        beam_size = self.opt.beam_size
        bsz = batch_size = batch.size

        # require batch first for everything
        outs = dict()
        attns = dict()

        for i in range(self.n_models):
            # decoder output contains the log-prob distribution of the next step
            # decoder_output = self.models[i].step(tokens, decoder_states[i])
            model_outputs = self.models[i](batches[i])

            logits = model_outputs['logits']
            mask = model_outputs['src_mask']
            mask = mask.squeeze(1).transpose(0, 1)
            mask = mask.unsqueeze(-1)
            logits.masked_fill_(mask, 0)
            lengths = (1 - mask.long()).squeeze(-1).sum(dim=0, keepdim=False)
            clean_logits = logits.sum(dim=0, keepdim=False).div(lengths.unsqueeze(-1))
            probs = F.softmax(clean_logits.float(), dim=-1)
            outs[i] = probs

        probs = sum(outs.values())
        probs.div_(self.n_models)

        return probs

    def build_data(self, src_sents, tgt_sents, type='mt', past_sents=None):
        # This needs to be the same as preprocess.py.

        if type == 'mt':
            raise NotImplementedError
        #     if self.start_with_bos:
        #         src_data = [self.src_dict.convertToIdx(b,
        #                                                onmt.constants.UNK_WORD,
        #                                                onmt.constants.BOS_WORD)
        #                     for b in src_sents]
        #     else:
        #         src_data = [self.src_dict.convertToIdx(b,
        #                                                onmt.constants.UNK_WORD)
        #                     for b in src_sents]
        #     data_type = 'text'
        #     past_src_data = None
        elif type == 'asr':
            # no need to deal with this
            src_data = src_sents
            past_src_data = past_sents
            data_type = 'audio'
        else:
            raise NotImplementedError

        tgt_bos_word = self.opt.bos_token
        if self.opt.no_bos_gold:
            tgt_bos_word = None
        tgt_data = None
        if tgt_sents:
            tgt_data = [self.tgt_dict.convertToIdx(b,
                                                   onmt.constants.UNK_WORD,
                                                   tgt_bos_word,
                                                   onmt.constants.EOS_WORD) for b in tgt_sents]

        src_lang_data = [torch.Tensor([self.lang_dict[self.src_lang]])]
        # tgt_lang_data = [torch.Tensor([self.lang_dict[self.tgt_lang]])]
        tgt_lang_data = None

        return onmt.Dataset(src_data, tgt_data,
                            src_langs=src_lang_data, tgt_langs=tgt_lang_data,
                            batch_size_words=sys.maxsize,
                            data_type=data_type,
                            batch_size_sents=self.opt.batch_size,
                            src_align_right=self.opt.src_align_right,
                            past_src_data=past_src_data)

    def predict(self, src_data):
        type = 'asr'
        #  (1) convert words to indexes
        if isinstance(src_data[0], list) and type == 'asr':
            batches = list()
            for i, src_data_ in enumerate(src_data):

                dataset = self.build_data(src_data_, None, type=type, past_sents=None)
                batch = dataset.get_batch(0)
                batches.append(batch)
        else:
            dataset = self.build_data(src_data, None, type=type)
            batch = dataset.get_batch(0)  # this dataset has only one mini-batch
            batches = [batch] * self.n_models
            src_data = [src_data] * self.n_models

        batch_size = batches[0].size
        if self.cuda:
            for i, _ in enumerate(batches):
                batches[i].cuda(fp16=self.fp16)

        #  (2) translate
        #  each model in the ensemble uses one batch in batches
        probs = self.classify_batch(batches)

        #  (3) convert indexes to words
        pred_score = []
        for b in range(batch_size):
            pred_score.append(
                probs[b].tolist()
            )

        return pred_score

