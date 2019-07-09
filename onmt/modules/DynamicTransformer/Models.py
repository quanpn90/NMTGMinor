import math
import torch
import onmt
from onmt.modules.DynamicTransformer.Dlcl import  DynamicLinearCombination
from onmt.modules.Transformer.Models import TransformerEncoder, TransformerDecoder
from onmt.modules.WordDrop import embedded_dropout
from torch.utils.checkpoint import checkpoint


class DlclTransformerEncoder(TransformerEncoder):
    """Transformer encoder."""

    def __init__(self, opt, dicts, positional_encoder, encoder_type='text'):

        super().__init__(opt, dicts, positional_encoder, encoder_type)

        self.history =  DynamicLinearCombination(self.model_size, self.layers, is_encoder=True)

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """
        # clean layer history
        self.history.clean()

        # Embedding: batch_size x len_src x d_model
        if self.input_type == "text":
            mask_src = input.data.eq(onmt.Constants.PAD).unsqueeze(1)  # batch_size x len_src x 1 for broadcasting
            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        else:
            mask_src = input.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
            input = input.narrow(2, 1, input.size(2) - 1)
            emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                    input.size(1), -1)

        # Scale the emb by sqrt(d_model)

        emb = emb * math.sqrt(self.model_size)

        # Adding positional encoding
        emb = self.time_transformer(emb)
        # Dropout
        emb = self.preprocess_layer(emb)

        # B x T x H -> T x B x H
        context = emb.transpose(0, 1).contiguous()

        self.history.push(context)

        for i, layer in enumerate(self.layer_modules):

            context = self.history.pop()

            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:
                context = checkpoint(custom_layer(layer), context, mask_src)

            else:
                context = layer(context, mask_src)  # batch_size x len_src x d_model

            self.history.push(context)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.history.pop()
        context = self.postprocess_layer(context)

        output_dict = {'context': context, 'src_mask': mask_src}

        # return context, mask_src
        return output_dict


class DlclTransformerDecoder(TransformerDecoder):

    def __init__(self, opt, dicts, positional_encoder, attribute_embeddings=None, ignore_source=False):

        super().__init__(opt, dicts, positional_encoder,
                         attribute_embeddings=attribute_embeddings, ignore_source=ignore_source)

        self.history =  DynamicLinearCombination(self.model_size, self.layers, is_encoder=False)

    def forward(self, input, context, src, atbs=None,  **kwargs):
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
        self.history.clean()

        emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = self.preprocess_layer(emb)

        if self.use_feature:
            atb_emb = self.attribute_embeddings(atbs).unsqueeze(1).repeat(1, emb.size(1)) #  B x H to 1 x B x H
            emb = torch.cat([emb, atb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))

        if context is not None:
            if self.encoder_type == "audio":
                mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        if context is not None:
            if self.encoder_type == "audio":
                mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)

        output = emb.transpose(0, 1).contiguous()

        self.history.push(output)

        for i, layer in enumerate(self.layer_modules):

            output = self.history.pop()

            if len(self.layer_modules) - i <= onmt.Constants.checkpointing and self.training:

                output, coverage = checkpoint(custom_layer(layer), output, context, mask_tgt, mask_src)
                                                                              # batch_size x len_src x d_model

            else:
                output, coverage = layer(output, context, mask_tgt, mask_src) # batch_size x len_src x d_model

            # write into memory
            self.history.push(output)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.history.pop()
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
        self.history.clean()
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

        if self.use_feature:
            atb_emb = self.attribute_embeddings(atbs).unsqueeze(1).expand_as(emb)  # B x H to 1 x B x H
            emb = torch.cat([emb, atb_emb], dim=-1)
            emb = torch.relu(self.feature_projector(emb))

        # Preprocess layer: adding dropout
        emb = self.preprocess_layer(emb)
        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src

        if context is not None:
            if self.encoder_type == "audio" and src.data.dim() == 3:
                mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.Constants.PAD).unsqueeze(1)
            else:
                mask_src = src.eq(onmt.Constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.data.eq(onmt.Constants.PAD).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        output = emb.contiguous()
        self.history.push(output)

        for i, layer in enumerate(self.layer_modules):

            output = self.history.pop()
            buffer = buffers[i] if i in buffers else None
            assert(output.size(0) == 1)

            output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer)

            decoder_state.update_attention_buffer(buffer, i)

            self.history.push(output)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.history.pop()
        output = self.postprocess_layer(output)

        return output, coverage