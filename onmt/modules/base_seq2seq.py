import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt, math
from onmt.modules.optimized.linear import Linear, linear_function


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size, fix_norm=False, bias=True):
        
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        self.fix_norm = fix_norm
        self.must_softmax = False
        
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        
        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)
        
        self.linear.bias.data.zero_()

    def forward(self, output_dicts):
        """
        :param output_dicts: dictionary contains the outputs from the decoder
        :return: logits (the elements before softmax)
        """

        input = output_dicts['hidden']
        fix_norm = self.fix_norm
        target_mask = output_dicts['target_mask']

        if not fix_norm:
            logits = self.linear(input)
        else:
            normalized_weights = F.normalize(self.linear.weight, dim=-1)
            normalized_bias = self.linear.bias
            logits = F.linear(input, normalized_weights, normalized_bias)

        # softmax will be done at the loss function
        # output = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        output_dicts['logits'] = logits
        output_dicts['softmaxed'] = self.must_softmax
        return output_dicts
        

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator=None, rec_decoder=None, rec_generator=None,
                 mirror=False, ctc=False):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.rec_decoder = rec_decoder
        self.rec_generator = rec_generator
        self.ctc = ctc

        if self.rec_decoder:
            self.rec_decoder.word_lut.weight = self.encoder.word_lut.weight
            self.reconstruct = True
        else:
            self.reconstruct = False

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.decoder.word_lut.weight

    def post_backward(self, *args, **kwargs):

        pass

    def share_enc_dec_embedding(self):
        self.encoder.word_lut.weight = self.decoder.word_lut.weight
        
    def mark_pretrained(self):
        
        self.encoder.mark_pretrained()
        self.decoder.mark_pretrained()

    def load_state_dict(self, state_dict, strict=False):
        """
        override this method to have back-compatibility
        """
        def condition(param_name):
            # don't load these buffers (more like a bug)
            if 'positional_encoder' in param_name:
                return False
            if 'time_transformer' in param_name:
                if self.encoder is not None:
                    if getattr(self.encoder, "enc_pretrained_model", None) or self.encoder.time == 'positional_encoding':
                        return False
            if param_name == 'decoder.mask':
                return False
            if param_name == 'decoder.r_w_bias' or param_name == 'decoder.r_r_bias':
                if param_name in model_dict:
                    return True
                return False

            return True

        # restore old generated if necessary for loading
        if "generator.linear.weight" in state_dict and type(self.generator) is nn.ModuleList:
            self.generator = self.generator[0]

        model_dict = self.state_dict()

        # only load the filtered parameters
        filtered = {k: v for k, v in state_dict.items() if condition(k)}

        for k, v in model_dict.items():
            if k not in filtered:
                filtered[k] = v

        # removing the keys in filtered but not in model dict
        if strict:
            removed_keys = list()
            for k, v in filtered.items():
                if k not in model_dict:
                    removed_keys.append(k)

            for k in removed_keys:
                filtered.pop(k)

        # ctc weights can be ignored:
        if 'ctc_linear.weight' not in model_dict and 'ctc_linear.weight' in filtered:
            filtered.pop('ctc_linear.weight')
            if 'ctc_linear.bias' in filtered:
                filtered.pop('ctc_linear.bias')

        super().load_state_dict(filtered)

        # in case using multiple generators
        if type(self.generator) is not nn.ModuleList:
            self.generator = nn.ModuleList([self.generator])

    def convert_autograd(self):

        def attempt_to_convert(m):
            if hasattr(m, 'convert_autograd'):
                m.convert_autograd()

            for n, ch in m.named_children():
                attempt_to_convert(ch)

        attempt_to_convert(self.encoder)
        attempt_to_convert(self.decoder)


class Reconstructor(nn.Module):
    """
    This class is currently unused, but can be used to learn to reconstruct from the hidden states
    """
    
    def __init__(self, decoder, generator=None):
        super(Reconstructor, self).__init__()
        self.decoder = decoder
        self.generator = generator


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """

    def update_beam(self, beam, b, remaining_sents, idx):

        raise NotImplementedError

    def prune_complete_beam(self, active_idx, remaining_sents):

        raise NotImplementedError
