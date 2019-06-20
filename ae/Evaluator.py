import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from onmt.ModelConstructor import build_model
import torch.nn.functional as F
from ae.Autoencoder import Autoencoder
import sys

model_list = ['transformer', 'stochastic_transformer']


class Evaluator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.start_with_bos = opt.start_with_bos
        self.fp16 = opt.fp16

        self.models = list()
        self.model_types = list()

        # models are string with | as delimiter
        models = opt.model.split("|")

        print(models)
        self.n_models = len(models)
        self._type = 'text'

        check_m = None;

        for i, model in enumerate(models):
            if opt.verbose:
                print('Loading model from %s' % model)
            checkpoint = torch.load(model,
                                    map_location=lambda storage, loc: storage)

            model_opt = checkpoint['opt']

            if i == 0:
                if ("src" in checkpoint['dicts']):
                    self.src_dict = checkpoint['dicts']['src']
                else:
                    self._type = "audio"
                self.tgt_dict = checkpoint['dicts']['tgt']

            # Build model from the saved option
            model = build_model(model_opt, checkpoint['dicts'])

            model.load_state_dict(checkpoint['model'])
            
            check_m = checkpoint['model'];


            if opt.cuda:
                model = model.cuda()
            else:
                model = model.cpu()

            if opt.fp16:
                model = model.half()

            model.eval()

            self.models.append(model)
            self.model_types.append(model_opt.model)

        self.cuda = opt.cuda




        ## Autoencoder

        if opt.verbose:
            print('Loading autoencoder from %s' % opt.autoencoder)
        checkpoint = torch.load(opt.autoencoder,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']


        posSize= checkpoint['autoencoder']['nmt.decoder.positional_encoder.pos_emb'].size(0)
        self.models[0].decoder.renew_buffer(posSize)


        # Build model from the saved option
        self.autoencoder = Autoencoder(self.models[0],model_opt)

        self.autoencoder.load_state_dict(checkpoint['autoencoder'])

        for k in checkpoint['autoencoder']:
            if(k.startswith("nmt") and k[4:] in check_m):
                n = checkpoint['autoencoder'][k]
                o = check_m[k[4:]]
                if(o.size() != n.size()):
                    print("Different size:",k[4:])
                elif((n - o).sum() != 0):
                    print("Different weight:",k[4:])

        if self.autoencoder.nmt.decoder.positional_encoder.len_max < self.opt.max_sent_length:
            self.autoencoder.nmt.decoder.renew_buffer(self.opt.max_sent_length)

        if opt.cuda:
            self.autoencoder = self.autoencoder.cuda()
        else:
            self.autoencoder = self.autoencoder.cpu()

        if opt.fp16:
            self.autoencoder = self.autoencoder.half()


        self.autoencoder.eval()


        if opt.verbose:
            print('Done')


    # Combine distributions from different models
    def _combineOutputs(self, outputs):

        if len(outputs) == 1:
            return outputs[0]

        if self.ensemble_op == "logSum":
            output = (outputs[0])

            # sum the log prob
            for i in range(1, len(outputs)):
                output += (outputs[i])

            output.div(len(outputs))

            # ~ output = torch.log(output)
            output = F.log_softmax(output, dim=-1)
        elif self.ensemble_op == "mean":
            output = torch.exp(outputs[0])

            # sum the log prob
            for i in range(1, len(outputs)):
                output += torch.exp(outputs[i])

            output.div(len(outputs))

            # ~ output = torch.log(output)
            output = torch.log(output)
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
    def _combineAttention(self, attns):

        attn = attns[0]

        for i in range(1, len(attns)):
            attn += attns[i]

        attn.div(len(attns))

        return attn

    def _getBatchSize(self, batch):
        #        if self._type == "text":
        return batch.size(1)

    #        else:
    #            return batch.size(0)

    def to_variable(self, data):
        for i, t in enumerate(data):
            if data[i] is not None:
                if self.cuda:
                    if (data[i].type() == "torch.FloatTensor" and self.fp16):
                        data[i] = data[i].half()
                    data[i] = Variable(data[i].cuda())
                else:
                    data[i] = Variable(data[i])
            else:
                data[i] = None
        return data

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.

        if self.start_with_bos:
            srcData = [self.src_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD)
                       for b in srcBatch]
        else:
            srcData = [self.src_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD)
                       for b in srcBatch]

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD) for b in goldBatch]


        return onmt.Dataset(srcData, tgtData, 9999,
                            data_type=self._type,
                            batch_size_sents=self.opt.batch_size)

    def buildASRData(self, srcData, goldBatch):
        # This needs to be the same as preprocess.py.

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, sys.maxsize,
                            [self.opt.gpu],
                            data_type=self._type, max_seq_num=self.opt.batch_size)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS

        return tokens

    def evalBatch(self, batch):

        torch.set_grad_enabled(False)
        # Batch size is in different location depending on data.

        if(self.autoencoder.representation == "EncoderDecoderHiddenState"):
            state,prediction = self.autoencoder.calcAlignment(batch)
        else:
            state, prediction = self.autoencoder(batch)
        return state,prediction

    def eval(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = dataset.next()[0]
        if self.cuda:
            batch.cuda()

        #  (2) eval
        state,prediction = self.evalBatch(batch)

        #  (3) convert indexes to words
        return self.calcDistance(state,prediction)



    def evalASR(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildASRData(srcBatch, goldBatch)
        batch = self.to_variable(dataset.next()[0])
        src, tgt = batch
        batchSize = self._getBatchSize(src)

        #  (2) eval
        state,prediction = self.evalBatch(src, tgt)

        #  (3) convert indexes to words
        return self.calcDistance(state,prediction)


    def calcDistance(self,state,prediction):

        if(self.autoencoder.representation == "EncoderDecoderHiddenState"):
            state = state.unsqueeze(0).expand(prediction.size(0),-1,-1,-1)
            prediction = prediction.unsqueeze(1).expand(-1,state.size(1),-1,-1)
            loss = state - prediction
            loss = loss.mul(loss)
            loss = loss.sum(-1)
        else:
            loss = state - prediction
            loss = loss.mul(loss)
            loss = loss.sum(1)
        return loss
