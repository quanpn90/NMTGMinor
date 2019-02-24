import numpy as np
import torch, math
import torch.nn as nn
import onmt
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder, TransformerDecodingState
from onmt.modules.BaseModel import NMTModel
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint
from onmt.modules.Utilities import mean_with_mask_backpropable as mean_with_mask
from onmt.modules.Utilities import max_with_mask

# ~ from onmt.modules.Checkpoint import checkpoint
from onmt.modules.Transformer.Layers import PrePostProcessing


def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward


class SimplifiedTransformerEncoder(TransformerEncoder):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, embedding, positional_encoder):

        self.layers = opt.layers

        # build_modules will be called from the inherited constructor
        super(SimplifiedTransformerEncoder, self).__init__(opt, embedding, positional_encoder)

        self.n_encoder_heads = opt.n_encoder_heads
        self.model_size = opt.model_size
        self.projector = nn.Linear(opt.model_size, opt.model_size * opt.n_encoder_heads)
        self.pooling = opt.var_pooling
        self.final_norm = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')

    def forward(self, input, freeze_embedding=False, return_stack=False, additional_sequence=None, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (wanna tranpose)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """

        add_emb = None
        if freeze_embedding:
            with torch.no_grad():
                emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

                """ Scale the emb by sqrt(d_model) """
                emb = emb * math.sqrt(self.model_size)

                if additional_sequence is not None:
                    add_input = additional_sequence
                    add_emb = embedded_dropout(self.word_lut, add_input,
                                               dropout=self.word_dropout if self.training else 0)

                    # emb = torch.cat([emb, add_emb], dim=0)
        else:
            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)

            """ Scale the emb by sqrt(d_model) """
            emb = emb * math.sqrt(self.model_size)

            if additional_sequence is not None:
                add_input = additional_sequence
                add_emb = embedded_dropout(self.word_lut, add_input, dropout=self.word_dropout if self.training else 0)

                # emb = torch.cat([emb, add_emb], dim=0)

        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        if add_emb is not None:
            add_emb = self.time_transformer(add_emb)

            # batch first
            emb = torch.cat([emb, add_emb], dim=1)
            input = torch.cat([input, additional_sequence], dim=1)

        emb = self.preprocess_layer(emb)

        mask_src = input.eq(onmt.Constants.PAD).unsqueeze(1)  # batch_size x 1 x len_src for broadcasting

        # time first
        context = emb.transpose(0, 1).contiguous()

        if return_stack == False:

            for i, layer in enumerate(self.layer_modules):

                if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
                    context = checkpoint(custom_layer(layer), context, mask_src)

                else:
                    context = layer(context, mask_src)  # batch_size x len_src x d_model

            # From Google T2T
            # if normalization is done in layer_preprocess, then it should also be done
            # on the output, since the output can grow very large, being the sum of
            # a whole stack of unnormalized layer outputs.
            context = self.postprocess_layer(context)

        else:
            raise NotImplementedError

        # project to different heads
        output = torch.nn.functional.elu(self.projector(context)) # T x B x h x H
        batch_size = output.size(1)
        output = output.view(output.size(0), output.size(1) ,self.n_encoder_heads, self.model_size)

        mask = input.eq(onmt.Constants.PAD).transpose(0, 1).unsqueeze(-1).unsqueeze(-1) # T x B x 1 x 1

        if self.pooling == 'mean':
            output = mean_with_mask(output, mask) # B x h x H
        elif self.pooling == 'max':
            output = max_with_mask(output, mask)
        else:
            raise NotImplementedError

        # self.encoder_heads, model_size
        output = self.final_norm(output.transpose(0, 1))

        mask_src = output.new(output.size(1), output.size(0)).fill_(onmt.Constants.EOS)

        return output, mask_src


class SimplifiedTransformer(NMTModel):
    """Main model in 'Attention is all you need' """

    def forward(self, batch, grow=False):
        """
        Inputs Shapes:
            src: len_src x batch_size
            tgt: len_tgt x batch_size

        Outputs Shapes:
            out:   a dictionary containing output hidden state and coverage

        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        tgt_attbs = batch.get('tgt_attbs')  # vector of length B

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        context, src_mask = self.encoder(src)

        fake_src = src.new(context.size(1), context.size(0)).fill_(onmt.Constants.BOS)
        output, coverage = self.decoder(tgt, tgt_attbs, context, fake_src)

        output_dict = dict()
        output_dict['hiddens'] = output
        output_dict['coverage'] = coverage
        return output_dict

    def decode(self, batch):

        src = batch.get('source')
        tgt_input = batch.get('target_input')
        tgt_attbs = batch.get('tgt_attbs')  # vector of length B
        tgt_output = batch.get('target_output')

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context, src_mask = self.encoder(src)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0

        fake_src = src.new(context.size(1), context.size(0)).fill_(onmt.Constants.BOS)

        output, coverage = self.decoder(tgt_input, tgt_attbs, context, fake_src)
        # output, coverage = self.decoder(tgt_input, context, fake_src)

        for dec_t, tgt_t in zip(output, tgt_output):
            gen_t = self.generator(dec_t)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.Constants.PAD).sum().item()

        return gold_words, gold_scores

    def create_decoder_state(self, batch, beam_size=1):

        src = batch.get('source')
        tgt_attbs = batch.get('tgt_attbs')  # vector of length B

        # transpose to have batch first
        src_transposed = src.transpose(0, 1)
        context, _ = self.encoder(src_transposed)

        fake_src = src.new(context.size(0), context.size(1)).fill_(onmt.Constants.EOS)
        decoder_state = TransformerDecodingState(fake_src, tgt_attbs, context, beam_size=beam_size)
        return decoder_state

