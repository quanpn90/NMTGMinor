import numpy as np
import torch, math
import torch.nn as nn
from collections import defaultdict
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, variational_dropout, \
    PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint
from onmt.modules.Utilities import mean_with_mask_backpropable as mean_with_mask
from onmt.modules.Utilities import max_with_mask



def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output

    return custom_forward


class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """

    def __init__(self, opt, embedding, positional_encoder, feature_embedding=None, share=None):

        super(TransformerEncoder, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.residual_dropout = opt.residual_dropout

        # lookup table for words
        self.word_lut = embedding

        # lookup table for features (here it's language embedding)
        self.feat_lut = feature_embedding

        if self.feat_lut is not None:
            self.enable_feature = True
            self.feature_projector = nn.Linear(opt.model_size * 2, opt.model_size)
        else:
            self.enable_feature = False

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.positional_encoder = positional_encoder

        self.build_modules(shared_encoder=share)

    def build_modules(self, shared_encoder=None):

        if shared_encoder is not None:

            print("* Shaing encoder parameters with another encoder")
            self.layer_modules = shared_encoder.layer_modules

            self.postprocess_layer = shared_encoder.postprocess_layer
        else:

            self.layer_modules = nn.ModuleList([EncoderLayer(self.n_heads, self.model_size, self.dropout,
                                                             self.inner_size, self.attn_dropout, self.residual_dropout)
                                                for _ in range(self.layers)])

            self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

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

        if not return_stack:

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

            return context, mask_src

        else:
            # return the normalized hidden representation of each layer
            output = list()

            for i, layer in enumerate(self.layer_modules):
                context, normalized_input = layer(context, mask_src, return_norm_input=True)

                if i > 0:
                    output.append(normalized_input)

            context = self.postprocess_layer(context)

            output.append(context)

            return output, mask_src


class TransformerDecoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: options for parameters
        embedding: torch.nn.Embedding instance for embedding
        positional encoder (sinusoid encoding)
        encoder to share: sharing parameters with encoder if needed
    """

    def __init__(self, opt, embedding, positional_encoder, feature_embedding=None, encoder_to_share=None):

        super(TransformerDecoder, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.residual_dropout = opt.residual_dropout
        self.copy_generator = opt.copy_generator
        self.pooling = opt.var_pooling
        self.fixed_target_length = 0
        if opt.fixed_target_length == "int":
            self.fixed_target_length = 1

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = embedding

        self.feat_lut = feature_embedding

        if self.feat_lut is not None:
            self.enable_feature = True
            self.feature_projector = nn.Linear(opt.model_size * 2, opt.model_size)
        else:
            self.enable_feature = False

        self.positional_encoder = positional_encoder

        if (self.fixed_target_length):
            self.length_lut =  nn.Embedding(8192,
                                     opt.model_size,
                                     padding_idx=onmt.Constants.PAD)

            self.length_projector = nn.Linear(opt.model_size * 2,opt.model_size);


        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max, len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

        self.build_modules(encoder_to_share=encoder_to_share)

    def build_modules(self, encoder_to_share=None):

        if encoder_to_share is None:
            self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size,
                                                             self.dropout, self.inner_size,
                                                             self.attn_dropout, self.residual_dropout)
                                                for _ in range(self.layers)])
        else:

            print("* Sharing Encoder and Decoder weights for self attention and feed forward layers ...")
            self.layer_modules = nn.ModuleList()

            for i in range(self.layers):
                decoder_layer = DecoderLayer(self.n_heads, self.model_size, self.dropout,
                                             self.inner_size, self.attn_dropout, self.residual_dropout,
                                             encoder_to_share=encoder_to_share.layer_modules[i])
                self.layer_modules.append(decoder_layer)

    def renew_buffer(self, new_len):

        self.positional_encoder.renew(new_len)

        if hasattr(self, 'mask'):
            del self.mask
        mask = torch.ByteTensor(np.triu(np.ones((new_len, new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def forward(self, input, input_attbs, context, src, freeze_embeddings=False, **kwargs):
        """
        Inputs Shapes: 
            input: (Tensor) batch_size x len_tgt (to be transposed)
            context: (Tensor) len_src x batch_size x d_model
        Outputs Shapes:
            out: tgt_len x batch_size x d_model
            coverage: batch_size x len_tgt x len_src
            
        """

        """ Embedding: batch_size x len_tgt x d_model """
        len_tgt = input.size(1)
        input_attbs = input_attbs.unsqueeze(1).repeat(1, len_tgt)
        returns = defaultdict(lambda: None)

        if freeze_embeddings:
            with torch.no_grad:
                emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
                attb_emb = self.feat_lut(input_attbs)
        else:
            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
            attb_emb = self.feat_lut(input_attbs)


        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]

        # B x T x H
        emb = self.preprocess_layer(emb)

        # expand B to B x T
        if(self.enable_feature):
            emb = torch.cat([emb, attb_emb], dim=-1)

            emb = torch.relu(self.feature_projector(emb))


        if(self.fixed_target_length):
            tgt_length = input.data.ne(onmt.Constants.PAD).sum(1).unsqueeze(1).expand_as(input.data)
            index = torch.arange(input.data.size(1)).unsqueeze(0).expand_as(tgt_length).type_as(tgt_length)
            tgt_length = (tgt_length - index) * input.data.ne(onmt.Constants.PAD).long()
            tgt_emb = self.length_lut(tgt_length);
            emb = torch.cat([emb, tgt_emb], dim=-1)

            emb = torch.relu(self.length_projector(emb))


        mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)


        # unused variable
        # pad_mask_src = src.data.ne(onmt.Constants.PAD)

        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)


        # transpose to T x B x H
        output = emb.transpose(0, 1).contiguous()

        for i, layer in enumerate(self.layer_modules):

            returns[i] = dict()

            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
                output_dict = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src)


            else:
                output_dict = layer(output, context, mask_tgt, mask_src)  # batch_size x len_src x d_model

            returns[i]['attn_out'] = output_dict['attn_out']

            #placeholder when new things need to be included to return

            output = output_dict['final_state']
            returns['coverage'] = output_dict['coverage']

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        returns['final_state'] = output

        # return output, None
        return returns

    def step(self, input, decoder_state):
        """
        Inputs Shapes: 
            input: (Variable) batch_size x len_tgt
            context: (Tensor) len_src x batch_size * beam_size x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src
            
        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        mask_src = decoder_state.src_mask
        input_attbs = decoder_state.tgt_attbs

        if decoder_state.concat_input_seq :
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)
            src = decoder_state.src.transpose(0, 1)

        input_ = input[:, -1].unsqueeze(1)

        # batch_size = input_.size(0)

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)

        emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb, t=input.size(1))

        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim

        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)

        input_attbs = input_attbs.unsqueeze(1)
        attb_emb = self.feat_lut(input_attbs)

        emb = torch.cat([emb, attb_emb], dim=-1)

        emb = torch.relu(self.feature_projector(emb))

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src
        if mask_src is None:
            mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        output = emb.contiguous()

        # FOR DEBUGGING
        # ~ decoder_state._debug_attention_buffer(0)

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            assert (output.size(0) == 1)
            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer)
            # batch_size x len_src x d_model

            decoder_state._update_attention_buffer(buffer, i)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        output = self.postprocess_layer(output)

        returns = dict()

        returns['hiddens'] = output
        returns['coverage'] = coverage
        returns['src'] = src.t()

        # return output, coverage

        return returns

class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """
    def __init__(self, encoder, decoder, generator=None, tgt_encoder=None):
        super().__init__(encoder, decoder, generator=generator)
        self.tgt_encoder = tgt_encoder
        self.pooling = self.decoder.pooling

    def forward(self, batch, grow=False):
        """
        The forward function served in training (for back propagation)

        Inputs Shapes: 
            batch (onmt.Dataset.Batch) an object containing tensors needed for training
        
        Outputs Shapes:
            out:   a dictionary containing output hidden state and coverage
            
        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        original_src = src
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        tgt_attbs = batch.get('tgt_attbs')  # vector of length B

        context, src_mask = self.encoder(src, grow=grow)

        decoder_output = self.decoder(tgt, tgt_attbs, context, src, grow=grow)

        output_dict = dict()
        output_dict['src'] = src
        output_dict['hiddens'] = decoder_output['final_state']
        output_dict['coverage'] = decoder_output['coverage']

        # additional loss term for multilingual
        # forcing the source and target context to be the same
        if self.tgt_encoder is not None:
            tgt_ = tgt[:, 1:]
            tgt_context, _ = self.tgt_encoder(tgt_)
            tgt_mask = tgt_.eq(onmt.Constants.PAD).transpose(0, 1).unsqueeze(2)

            if self.pooling == 'mean':
                tgt_context = mean_with_mask(tgt_context, tgt_mask)
            else:
                tgt_context = max_with_mask(tgt_context, tgt_mask)

            src_mask = src.eq(onmt.Constants.PAD).transpose(0, 1).unsqueeze(2)

            if self.pooling == 'mean':
                src_context = mean_with_mask(context, src_mask)
            else:
                src_context = mean_with_mask(context, src_mask)

            output_dict['src_context'] = src_context
            output_dict['tgt_context'] = tgt_context

        return output_dict

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
        """

        src = batch.get('source')
        tgt_input = batch.get('target_input')
        tgt_attbs = batch.get('tgt_attbs')  # vector of length B
        tgt_output = batch.get('target_output')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context, src_mask = self.encoder(src)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0

        decoder_output = self.decoder(tgt_input, tgt_attbs, context, src)

        output_dict = dict()
        output_dict['src'] = src.t()
        output_dict['hiddens'] = decoder_output['final_state']
        output_dict['coverage'] = decoder_output['coverage']

        gens = self.generator(output_dict)

        for gen_t, tgt_t in zip(gens, tgt_output):

            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.Constants.PAD).sum().item()
        # for dec_t, tgt_t in zip(output, tgt_output):
        #
        #
        #
        #     gen_t = self.generator(net_output)
        #     tgt_t = tgt_t.unsqueeze(1)
        #     scores = gen_t.gather(1, tgt_t)
        #     scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
        #     gold_scores += scores.squeeze(1).type_as(gold_scores)
        #     gold_scores += scores.squeeze(1).type_as(gold_scores)

        return gold_words, gold_scores

    def step(self, input, decoder_state):

        decoder_output = self.decoder.step(input, decoder_state)

        # decoder_hidden = decoder_output['hiddens'].squeeze(1)
        coverage = decoder_output['coverage'][:, -1, :].squeeze(1) # batch * beam x src_len

        log_dist = self.generator(decoder_output).squeeze(1)

        return log_dist, coverage
        # return self.decoder.step(input, decoder_state)

    def create_decoder_state(self, batch, beam_size):

        # from onmt.modules.ParallelTransformer.Models import ParallelTransformerEncoder, ParallelTransformerDecoder
        from onmt.modules.StochasticTransformer.Models import StochasticTransformerDecoder
        from onmt.modules.UniversalTransformer.Models import UniversalTransformerDecoder

        src = batch.get('source')
        tgt_attbs = batch.get('tgt_attbs')  # vector of length B

        # transpose to have batch first
        src_transposed = src.transpose(0, 1)
        context, _ = self.encoder(src_transposed)

        if isinstance(self.decoder, TransformerDecoder) or isinstance(self.decoder, StochasticTransformerDecoder) \
                or isinstance(self.decoder, UniversalTransformerDecoder):
            decoder_state = TransformerDecodingState(src, tgt_attbs, context, beam_size=beam_size)
        else:
            raise NotImplementedError
        # elif isinstance(self.decoder, ParallelTransformerDecoder):
        #     from onmt.modules.ParallelTransformer.Models import ParallelTransformerDecodingState
        #     decoder_state = ParallelTransformerDecodingState(src, context, mask_src, beam_size=beam_size)
        return decoder_state


class TransformerDecodingState(DecoderState):

    def __init__(self, src, tgt_attbs, context, beam_size=1):

        self.beam_size = beam_size

        self.input_seq = None
        self.attention_buffers = dict()
        self.original_src = src

        self.src = src.repeat(1, beam_size)
        self.context = context.repeat(1, beam_size, 1)
        self.beam_size = beam_size
        self.src_mask = None
        self.concat_input_seq = True
        self.tgt_attbs = tgt_attbs.repeat(beam_size)  # size: Bxb

    def _update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer  # dict of 2 keys (k, v) : T x B x H

    def _debug_attention_buffer(self, layer):

        if layer not in self.attention_buffers:
            return
        buffer = self.attention_buffers[layer]

        for k in buffer.keys():
            print(k, buffer[k].size())

    def _update_beam(self, beam, b, remaining_sents, idx):
        # here we have to reorder the beam data 
        # 
        for tensor in [self.src, self.input_seq]:
            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(
                1, beam[b].getCurrentOrigin()))

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_:
                    t_, br_, d_ = buffer_[k].size()
                    sent_states = buffer_[k].view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                    sent_states.copy_(sent_states.index_select(
                        1, beam[b].getCurrentOrigin()))

        state_ = self.tgt_attbs.view(self.beam_size, remaining_sents)[:, idx]

        state_.copy_(state_.index_select(0, beam[b].getCurrentOrigin()))

    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def _prune_complete_beam(self, active_idx, remaining_sents):

        model_size = self.context.size(-1)

        def update_active_with_hidden(t):
            if t is None:
                return t
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, model_size)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            return view.index_select(1, active_idx).view(*new_size)

        def update_active_without_hidden(t):
            if t is None:
                return t
            view = t.view(-1, remaining_sents)
            new_size = list(t.size())
            new_size[-1] = new_size[-1] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)

            return new_t

        self.context = update_active_with_hidden(self.context)

        self.input_seq = update_active_without_hidden(self.input_seq)

        self.src = update_active_without_hidden(self.src)

        self.tgt_attbs = update_active_without_hidden(self.tgt_attbs)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_:
                    buffer_[k] = update_active_with_hidden(buffer_[k])
