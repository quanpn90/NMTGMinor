import numpy as np
import torch, math
import torch.nn as nn
from onmt.modules.Transformer.Layers import EncoderLayer, DecoderLayer, PositionalEncoding, variational_dropout, PrePostProcessing
from onmt.modules.BaseModel import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint
from collections import defaultdict

def custom_layer(module):
    def custom_forward(*args):
        output = module(*args)
        return output
    return custom_forward


class MixedEncoder(nn.Module):


    def __init(self,text_encoder,audio_encoder):
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (wanna tranpose)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """

        if input.dim() == 2:
            return self.text_encoder.forward(input)
        else:
            return self.audio_encoder.forward(input)


class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'
    
    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)
        
    """
    
    def __init__(self, opt, embedding, positional_encoder, encoder_type='text'):
    
        super(TransformerEncoder, self).__init__()
        
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        if hasattr(opt,'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.time = opt.time
        self.version = opt.version
        self.input_type = encoder_type
        self.cnn_downsampling = opt.cnn_downsampling
        self.channels = 1
        feature_size = opt.input_size

        if encoder_type != "text":
            if not self.cnn_downsampling:
                self.audio_trans = nn.Linear(feature_size, self.model_size)
                torch.nn.init.xavier_uniform_(self.audio_trans.weight)
            else:
                channels = self.channels
                cnn = [nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(64),
                       nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(64)]
                self.audio_trans = nn.Sequential(*cnn)

                # self.model_size =
                feat_size = (((feature_size // channels) - 3) // 4) * 64
                assert self.model_size == feat_size, "The model dimension doesn't match with the feature dim, expecting %d " % feat_size
        else:
            self.word_lut = embedding

        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        elif opt.time == 'gru':
            self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        elif opt.time == 'lstm':
            self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)
        
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        
        self.positional_encoder = positional_encoder

        self.build_modules()

    def build_modules(self):

        self.layer_modules = nn.ModuleList([EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size, self.attn_dropout) for _ in range(self.layers)])

    def forward(self, input, **kwargs):
        """
        Inputs Shapes: 
            input: batch_size x len_src (wanna tranpose)
        
        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src 
            
        """

        """ Embedding: batch_size x len_src x d_model """
        if self.input_type == "text":
            mask_src = input.data.eq(onmt.Constants.PAD).unsqueeze(1)  # batch_size x len_src x 1 for broadcasting
            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        else:
            if not self.cnn_downsampling:
                mask_src = input.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
                input = input.narrow(2, 1, input.size(2) - 1)
                emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                        input.size(1), -1)
            else:
                long_mask =  input.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
                input = input.narrow(2, 1, input.size(2) - 1)

                # first resizing to fit the CNN format
                input = input.view(input.size(0), input.size(1), -1, self.channels)
                input = input.permute(0, 3, 1, 2)

                input = self.audio_trans(input)
                input = input.permute(0, 2, 1, 3).contiguous()
                input = input.view(input.size(0), input.size(1), -1)


                mask_src = long_mask[:, 0:input.size(1)*4:4].unsqueeze(1)
                emb = input

        """ Scale the emb by sqrt(d_model) """
        
        emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        emb = self.preprocess_layer(emb)

        # B x T x H -> T x B x H
        context = emb.transpose(0, 1).contiguous()
        
        for i, layer in enumerate(self.layer_modules):
            
            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:        
                context = checkpoint(custom_layer(layer), context, mask_src)

            else:
                context = layer(context, mask_src)      # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.    
        context = self.postprocess_layer(context)

        output_dict = { 'context': context, 'src_mask': mask_src }

        # return context, mask_src
        return output_dict


class TransformerDecoder(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt
        dicts


    """

    def __init__(self, opt, embedding, positional_encoder, attribute_embeddings=None, ignore_source=False):

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
        self.version = opt.version
        self.encoder_type = opt.encoder_type
        self.ignore_source = ignore_source
        self.encoder_cnn_downsampling = opt.cnn_downsampling


        if opt.time == 'positional_encoding':
            self.time_transformer = positional_encoder
        else:
            raise NotImplementedError

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d', static=False)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = embedding

        # Using feature embeddings in models
        if attribute_embeddings is not None:
            self.use_feature = True
            self.attribute_embeddings = attribute_embeddings
            self.feature_projector = nn.Linear(opt.model_size + opt.model_size * attribute_embeddings.size(),
                                               opt.model_size)
        else:
            self.use_feature = None

        self.positional_encoder = positional_encoder

        len_max = self.positional_encoder.len_max
        mask = torch.ByteTensor(np.triu(np.ones((len_max,len_max)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

        self.build_modules()

    def build_modules(self):
        self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size,
                                                         self.dropout, self.inner_size,
                                                         self.attn_dropout,
                                                         ignore_source=self.ignore_source) for _ in range(self.layers)])

    def renew_buffer(self, new_len):

        print(new_len)
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len,new_len)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def process_embedding(self, input, atbs=None):

        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        # Adding dropout
        emb = self.preprocess_layer(emb)

        if self.use_feature:
            atb_emb = self.attribute_embeddings(atbs).unsqueeze(1).expand_as(emb) #  B x H to 1 x B x H
            emb = torch.cat([emb, atb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))
            emb = emb * math.sqrt(self.model_size)

        return emb

    def forward(self, input, context, src, atbs=None, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """

        emb = self.process_embedding(input, atbs)

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0)*4:4].unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)

        output = emb.transpose(0, 1).contiguous()

        for i, layer in enumerate(self.layer_modules):

            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:

                output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src)
                                                                              # batch_size x len_src x d_model

            else:
                output, coverage = layer(output, context, mask_tgt, mask_src) # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = { 'hidden': output, 'coverage': coverage }

        # return output, None
        return output_dict

    def step(self, input, decoder_state):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (wanna tranpose)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None
        atbs = decoder_state.tgt_atb

        if decoder_state.input_seq is None:
            decoder_state.input_seq = input
        else:
            # concatenate the last input to the previous input sequence
            decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
        input = decoder_state.input_seq.transpose(0, 1)
        input_ = input[:,-1].unsqueeze(1)

        """ Embedding: batch_size x 1 x d_model """
        emb = self.word_lut(input_)

        """ Adding positional encoding """
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
            emb = self.time_transformer(emb, t=input.size(1))
        else:
            # prev_h = buffer[0] if buffer is None else None
            # emb = self.time_transformer(emb, prev_h)
            # buffer[0] = emb[1]
            raise NotImplementedError

        if isinstance(emb, tuple):
            emb = emb[0]
        # emb should be batch_size x 1 x dim

        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src
        if context is not None:
            if self.encoder_type == "audio":
                if src.data.dim() == 3:
                    if self.encoder_cnn_downsampling:
                        long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                    else:
                        mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
                elif self.encoder_cnn_downsampling:
                    long_mask =  src.eq(onmt.Constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
            else:
                mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):

            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)

            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        return output, coverage


class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None):
        super().__init__( encoder, decoder, generator)
        self.model_size = self.decoder.model_size

    def reset_states(self):
        return

    def forward(self, batch):
        """
        Inputs Shapes: 
            src: len_src x batch_size
            tgt: len_tgt x batch_size
        
        Outputs Shapes:
            out:      batch_size*len_tgt x model_size
            
            
        """
        src = batch.get('source')
        tgt = batch.get('target_input')
        tgt_atb = batch.get('target_atb')  # a dictionary of attributes

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)
        
        encoder_output = self.encoder(src)
        context = encoder_output['context']
        
        decoder_output = self.decoder(tgt, context, src, atbs=tgt_atb)
        output = decoder_output['hidden']

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['encoder'] = context
        output_dict['src_mask'] = encoder_output['src_mask']

        logprobs = self.generator[0](output)
        # output_dict['logprobs'] = logprobs

        return output_dict

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """

        src = batch.get('source')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context = self.encoder(src)['context']

        if hasattr(self,'autoencoder') and self.autoencoder \
                and self.autoencoder.representation == "EncoderHiddenState":
            context = self.autoencoder.autocode(context)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()

        decoder_output = self.decoder(tgt_input, context, src)['hidden']

        output = decoder_output

        if hasattr(self, 'autoencoder')  and self.autoencoder and \
                self.autoencoder.representation == "DecoderHiddenState":
            output = self.autoencoder.autocode(output)

        for dec_t, tgt_t in zip(output, tgt_output):
            if isinstance(self.generator,nn.ModuleList):
                gen_t = self.generator[0](dec_t)
            else:
                gen_t = self.generator(dec_t)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.Constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):
        self.decoder.renew_buffer(new_len)

    def step(self, input_t, decoder_state):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        hidden, coverage = self.decoder.step(input_t, decoder_state)
        # squeeze to remove the time step dimension
        log_prob = self.generator[0](hidden.squeeze(0))

        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict = defaultdict(lambda: None)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1):
        """
        Generate a new decoder state based on the batch input
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')

        tgt_atb = batch.get('target_atb')

        src_transposed = src.transpose(0, 1)
        encoder_output = self.encoder(src_transposed)

        decoder_state = TransformerDecodingState(src, tgt_atb, encoder_output['context'],
                                                 beam_size=beam_size, model_size=self.model_size)

        return decoder_state


class TransformerDecodingState(DecoderState):
    
    def __init__(self, src, tgt_atb, context, beam_size=1, model_size=512):

        # if audio only take one dimension since only used for mask
        self.original_src = src  # TxBxC

        if src is not None:
            if src.dim() == 3:
                self.src = src.narrow(2, 0, 1).squeeze(2).repeat(1, beam_size)
                print(self.src.size())
                # self.src = src.repeat(1, beam_size, 1) # T x Bb x c
            else:
                self.src = src.repeat(1, beam_size)
        else:
            self.src = None

        if context is not None:
            self.context = context.repeat(1, beam_size, 1)
        else:
            self.context = None
        self.beam_size = beam_size

        self.input_seq = None
        self.attention_buffers = dict()
        self.model_size = model_size

        if tgt_atb is not None:
            self.use_attribute = True
            self.tgt_atb = tgt_atb.repeat(beam_size)  # size: Bxb
        else:
            self.tgt_atb = None

    def update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer # dict of 2 keys (k, v) : T x B x H

    def update_beam(self, beam, b, remaining_sents, idx):

        for tensor in [self.src, self.input_seq]  :

            if tensor is None:
                continue

            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(
                1, beam[b].getCurrentOrigin()))

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            if buffer_ is None:
                continue

            for k in buffer_:
                t_, br_, d_ = buffer_[k].size()
                sent_states = buffer_[k].view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                sent_states.data.copy_(sent_states.data.index_select(
                            1, beam[b].getCurrentOrigin()))

    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def prune_complete_beam(self, active_idx, remaining_sents):

        model_size = self.model_size

        def update_active_with_hidden(t):
            if t is None:
                return t
            dim = t.size(-1)
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, dim)
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

        if self.src.dim() == 2:
            self.src = update_active_without_hidden(self.src)
        elif self.src.dim() == 3:
            t = self.src
            dim = t.size(-1)
            view = t.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            self.src = new_t

        self.tgt_atb = update_active_without_hidden(self.tgt_atb)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            for k in buffer_:
                buffer_[k] = update_active_with_hidden(buffer_[k])

