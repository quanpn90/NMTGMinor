import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from onmt.ModelConstructor import build_model, build_language_model
from ae.Autoencoder import Autoencoder
import torch.nn.functional as F
import sys

model_list = ['transformer', 'stochastic_transformer', 'fusion_network']


class Rescorer(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None
        self.beta = opt.beta
        self.alpha = opt.alpha
        self.start_with_bos = opt.start_with_bos
        self.fp16 = opt.fp16
        self.attributes = opt.attributes  # attributes split by |. for example: de|domain1
        self.bos_token = opt.bos_token
        self.sampling = opt.sampling

        if self.attributes:
            self.attributes = self.attributes.split("|")

        self.models = list()
        self.model_types = list()

        # models are string with | as delimiter
        models = opt.model.split("|")

        print(models)
        self.n_models = len(models)
        self._type = 'text'

        for i, model in enumerate(models):
            if opt.verbose:
                print('Loading model from %s' % model)
            checkpoint = torch.load(model,
                                    map_location=lambda storage, loc: storage)

            model_opt = checkpoint['opt']

            if i == 0:
                if "src" in checkpoint['dicts']:
                    self.src_dict = checkpoint['dicts']['src']
                else:
                    self._type = "audio"
                self.tgt_dict = checkpoint['dicts']['tgt']

                if "atb" in checkpoint["dicts"]:
                    self.atb_dict = checkpoint['dicts']['atb']

                else:
                    self.atb_dict = None

                self.bos_id = self.tgt_dict.labelToIdx[self.bos_token]

            # Build model from the saved option
            # if hasattr(model_opt, 'fusion') and model_opt.fusion == True:
            #     print("* Loading a FUSION model")
            #     model = build_fusion(model_opt, checkpoint['dicts'])
            # else:
            #     model = build_model(model_opt, checkpoint['dicts'])
            model = build_model(model_opt, checkpoint['dicts'])
            model.load_state_dict(checkpoint['model'])

            if model_opt.model in model_list:
                # if model.decoder.positional_encoder.len_max < self.opt.max_sent_length:
                #     print("Not enough len to decode. Renewing .. ")
                #     model.decoder.renew_buffer(self.opt.max_sent_length)
                model.renew_buffer(self.opt.max_sent_length)

            if opt.fp16:
                model = model.half()

            if opt.cuda:
                model = model.cuda()
            else:
                model = model.cpu()

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

    def init_beam_accum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    # Combine distributions from different models
    def _combine_outputs(self, outputs):

        if len(outputs) == 1:
            return outputs[0]

        if self.ensemble_op == "logSum":
            output = (outputs[0])

            # sum the log prob
            for i in range(1, len(outputs)):
                output += (outputs[i])

            output.div_(len(outputs))

            # output = torch.log(output)
            output = F.log_softmax(output, dim=-1)
        elif self.ensemble_op == "mean":
            output = torch.exp(outputs[0])

            # sum the log prob
            for i in range(1, len(outputs)):
                output += torch.exp(outputs[i])

            output.div_(len(outputs))
            # output = torch.log(output)
            output = torch.log(output)
        elif self.ensemble_op == "max":
            output = outputs[0]

            for i in range(1, len(outputs)):
                output = torch.max(output,outputs[i])

        elif self.ensemble_op == "min":
            output = outputs[0]

            for i in range(1, len(outputs)):
                output = torch.min(output,outputs[i])

        elif self.ensemble_op == 'gmean':
            output = torch.exp(outputs[0])

            # geometric mean of the probabilities
            for i in range(1, len(outputs)):
                output *= torch.exp(outputs[i])

            # have to normalize
            output.pow_(1.0 / float(len(outputs)))
            norm_ = torch.norm(output, p=1, dim=-1)
            output.div_(norm_.unsqueeze(-1))

            output = torch.log(output)
        else:
            raise ValueError(
                'Emsemble operator needs to be "mean" or "logSum", the current value is %s' % self.ensemble_op)
        return output

    # Take the average of attention scores
    def _combine_attention(self, attns):

        attn = attns[0]

        for i in range(1, len(attns)):
            attn += attns[i]

        attn.div(len(attns))

        return attn

    def build_data(self, src_sents, tgt_sents):
        # This needs to be the same as preprocess.py.

        if self.start_with_bos:
            src_data = [self.src_dict.convertToIdx(b,
                                                   onmt.Constants.UNK_WORD,
                                                   onmt.Constants.BOS_WORD)
                        for b in src_sents]
        else:
            src_data = [self.src_dict.convertToIdx(b,
                                                   onmt.Constants.UNK_WORD)
                        for b in src_sents]

        tgt_bos_word = self.opt.bos_token
        tgt_data = None
        if tgt_sents:
            tgt_data = [self.tgt_dict.convertToIdx(b,
                                                   onmt.Constants.UNK_WORD,
                                                   tgt_bos_word,
                                                   onmt.Constants.EOS_WORD) for b in tgt_sents]

        src_atbs = None

        if self.attributes:
            tgt_atbs = dict()

            idx = 0
            for i in self.atb_dict:
                tgt_atbs[i] = [self.atb_dict[i].convertToIdx([self.attributes[idx]], onmt.Constants.UNK_WORD)
                               for _ in src_sents]
                idx = idx + 1

        else:
            tgt_atbs = None

        return onmt.Dataset(src_data, tgt_data,
                            src_atbs=src_atbs, tgt_atbs=tgt_atbs,
                            batch_size_words=sys.maxsize,
                            data_type=self._type,
                            batch_size_sents=sys.maxsize)

    def build_asr_data(self, src_data, tgt_sents):
        # This needs to be the same as preprocess.py.

        tgt_data = None
        if tgt_sents:
            tgt_data = [self.tgt_dict.convertToIdx(b,
                                                   onmt.Constants.UNK_WORD,
                                                   onmt.Constants.BOS_WORD,
                                                   onmt.Constants.EOS_WORD) for b in tgt_sents]

        return onmt.Dataset(src_data, tgt_data,
                            batch_size_words=sys.maxsize,
                            data_type=self._type, batch_size_sents=self.opt.batch_size)

    def build_target_tokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS

        return tokens

    def rescore_batch(self, batch):

        torch.set_grad_enabled(False)
        # Batch size is in different location depending on data.

        beam_size = self.opt.beam_size
        batch_size = batch.size

        gold_scores = batch.get('source').data.new(batch_size).float().zero_()
        gold_words = 0
        allgold_scores = []

        if batch.has_target:
            # Use the first model to decode
            model_ = self.models[0]

            gold_words, gold_scores, allgold_scores = model_.decode(batch)

        torch.set_grad_enabled(True)

        return gold_scores, gold_words, allgold_scores

    def rescore(self, src_data, tgt_data):
        #  (1) convert words to indexes
        dataset = self.build_data(src_data, tgt_data)
        batch = dataset.next()[0]
        if self.cuda:
            batch.cuda(fp16=self.fp16)
        batch_size = batch.size

        #  (2) translate
        gold_score, gold_words, allgold_words = self.rescore_batch(batch)

        return gold_score, gold_words, allgold_words

    def rescore_asr(self, src_data, tgt_data):
        #  (1) convert words to indexes
        dataset = self.build_asr_data(src_data, tgt_data)
        # src, tgt = batch
        batch = dataset.next()[0]
        if self.cuda:
            batch.cuda(fp16=self.fp16)
        batch_size = batch.size

        #  (2) translate
        gold_score, gold_words, allgold_words = self.rescore_batch(batch)

        return gold_score, gold_words, allgold_words


