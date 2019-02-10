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


class Decoder(nn.Module):
    """Base class for decoders. Implementing classes should provide the constructor parameter
    encoder_to_share, to share parameters with an encoder instance"""

    def __init__(self, future_masking=True, encoder_to_share=None):
        super().__init__()
        self.future_masking = future_masking
        self.future_mask = None

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

    def forward(self, decoder_inputs, encoder_outputs, input_mask=None, encoder_mask=None,
                incremental_state=None):
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
    decoders with the encoder_to_share paremeter"""

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
        res = super().convert_state_dict(state_dict)
        res['decoder'] = {'future_mask': state_dict['decoder']['mask']}
        return res
