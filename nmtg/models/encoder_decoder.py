import torch
from torch import nn

from nmtg.models import Model


class Encoder(nn.Module):
    """Base class for encoders."""

    def forward(self, encoder_inputs, input_mask=None):
        """
        Run the encoder
        :param encoder_inputs: (FloatTensor) Input to the encoder
        :param input_mask: (ByteTensor) Optional mask for invalid inputs (i.e. padding)
        :return:
        """
        raise NotImplementedError

    def get_self_attention_bias(self, inputs, batch_first, input_mask=None):
        if input_mask is not None:
            self_attention_mask = input_mask if batch_first else input_mask.transpose(0, 1)
            self_attention_bias = inputs.new_full(self_attention_mask.size(), float('-inf'))\
                .masked_fill_(self_attention_mask, 0).unsqueeze(1)
        else:
            self_attention_bias = None
        return self_attention_bias


class Decoder(nn.Module):
    """Base class for decoders. Implementing classes should provide the constructor parameter
    encoder_to_share, to share parameters with an encoder instance"""

    def __init__(self, future_masking=True, encoder_to_share=None):
        super().__init__()
        self.future_masking = future_masking
        self.register_buffer('future_mask', None)
        self._register_load_state_dict_pre_hook(self._fix_future_mask)

    def get_future_mask(self, dim, device):
        if self.future_mask is None or self.future_mask.device != device:
            self.future_mask = torch.tril(torch.ones(dim, dim, dtype=torch.uint8, device=device), 0)
        elif self.future_mask.size(0) < dim:
            self.future_mask = torch.tril(self.future_mask.resize_(dim, dim).fill_(1), 0)
        return self.future_mask[:dim, :dim]

    def forward(self, decoder_inputs, encoder_outputs, input_mask=None, encoder_mask=None):
        """
        Run the decoder
        :param decoder_inputs: (FloatTensor) Input to the decoder
        :param encoder_outputs: (FloatTensor) Outputs from the encoder
        :param input_mask: (ByteTensor) Optional mask for invalid decoder inputs (i.e. padding)
        :param encoder_mask: (ByteTensor) Optional mask for invalid encoder outputs (i.e. padding)
        :return:
        """
        raise NotImplementedError

    def _fix_future_mask(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        if prefix + 'future_mask' in state_dict:
            old_mask = state_dict[prefix + 'future_mask']
            if self.future_mask is None:
                self.future_mask = old_mask.new(old_mask.size())
            else:
                self.future_mask.resize_(list(old_mask.size()))

    def get_self_attention_mask(self, decoder_inputs, batch_first, input_mask=None, future_masking=True):
        if future_masking and self.future_masking:
            len_tgt = decoder_inputs.size(1 if batch_first else 0)
            # future_mask: (len_tgt x len_tgt)
            future_mask = self.get_future_mask(len_tgt, decoder_inputs.device)
            if input_mask is None:
                self_attention_mask = future_mask.unsqueeze(0)
            else:
                # padding_mask: (batch_size x len_tgt)
                padding_mask = input_mask if batch_first else input_mask.transpose(0, 1)
                self_attention_mask = (padding_mask.unsqueeze(1) + future_mask).gt_(1)
        elif input_mask is not None:
            self_attention_mask = input_mask if batch_first else input_mask.transpose(0, 1)
            self_attention_mask = self_attention_mask.unsqueeze(1)
        else:
            self_attention_mask = None

        return self_attention_mask

    def get_self_attention_bias(self, decoder_inputs, batch_first, input_mask=None, future_masking=True):
        self_attention_mask = self.get_self_attention_mask(decoder_inputs, batch_first, input_mask, future_masking)

        if self_attention_mask is not None:
            # self_attention_mask: (batch_size x len_tgt x len_tgt) or broadcastable
            self_attention_bias = decoder_inputs.new_full(self_attention_mask.size(), float('-inf'))\
                .masked_fill_(self_attention_mask, 0)
        else:
            self_attention_bias = None

        return self_attention_bias

    def get_encoder_attention_bias(self, encoder_outputs, batch_first, encoder_mask=None):
        if encoder_mask is not None:
            # enc_padding_mask: (batch_size x len_src) or broadcastable
            enc_padding_mask = encoder_mask if batch_first else encoder_mask.transpose(0, 1)
            encoder_attention_bias = encoder_outputs.new_full(enc_padding_mask.size(), float('-inf'))\
                .masked_fill(enc_padding_mask, 0).unsqueeze(1)
        else:
            encoder_attention_bias = None

        return encoder_attention_bias


class IncrementalModule(nn.Module):
    def step(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        result = self._step(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        return result

    def _step(self, *input, **kwargs):
        raise NotImplementedError


class IncrementalDecoder(Decoder):
    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        seen = set()

        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state') \
                    and module not in seen:
                seen.add(module)
                module.reorder_incremental_state(incremental_state, new_order)

        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size') \
                        and module not in seen:
                    seen.add(module)
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size

    def step(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        result = self._step(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        return result

    def _step(self, decoder_inputs, encoder_outputs, incremental_state, input_mask=None, encoder_mask=None):
        """
        Run the decoder
        :param decoder_inputs: (FloatTensor) Input to the decoder
        :param encoder_outputs: (FloatTensor) Outputs from the encoder
        :param input_mask: (ByteTensor) Optional mask for invalid decoder inputs (i.e. padding)
        :param encoder_mask: (ByteTensor) Optional mask for invalid encoder outputs (i.e. padding)
        :param incremental_state: dict for storing state during incremental decoding
        :return:
        """
        raise NotImplementedError


class EncoderDecoderModel(Model):
    """Base class for Encoder/Decoder models. Implementing classes should construct encoders and
    decoders with the encoder_to_share parameter"""

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def add_options(parser):
        parser.add_argument('-share_enc_dec_weights', action='store_true',
                            help='Share the encoder and decoder weights (except for the src attention layer)')

    def forward(self, encoder_inputs, decoder_inputs, encoder_mask=None, decoder_mask=None):
        """
        Run a forward pass through encoder and decoder, forcing decoder input to the given parameter
        :param encoder_inputs: (FloatTensor) Inputs to the encoder
        :param decoder_inputs: (FloatTensor) Inputs to the decoder
        :param encoder_mask: (ByteTensor) Optional mask for invalid encoder input (i.e. padding)
        :param decoder_mask: (ByteTensor) Optional mask for invalid decoder input (i.e. padding)
        :return: (FloatTensor) Output from the decoder.
        """
        encoder_out = self.encoder(encoder_inputs, encoder_mask)
        decoder_out = self.decoder(decoder_inputs, encoder_out, decoder_mask, encoder_mask)
        return decoder_out

    @staticmethod
    def convert_state_dict(opt, state_dict):
        res = Model.convert_state_dict(opt, state_dict)
        res['decoder'] = {'future_mask': state_dict['decoder']['mask']}
        opt.no_future_masking = False
        return res
