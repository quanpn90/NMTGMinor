import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from torch.autograd import Variable
from .tacotron_layers import LinearNorm, Prenet, Attention, get_mask_from_lengths, Postnet
import math
import onmt
from math import sqrt
from onmt.modules.base_seq2seq import NMTModel, DecoderState
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.dropout import embedded_dropout
from onmt.models.speech_recognizer.relative_transformer_layers import LIDFeedForward_small


class SpeechLSTMEncoder(nn.Module):
    def __init__(self, opt, embedding, encoder_type='audio'):
        super(SpeechLSTMEncoder, self).__init__()
        self.opt = opt
        self.model_size = opt.model_size

        if hasattr(opt, 'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers

        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout

        self.input_type = encoder_type
        self.cnn_downsampling = opt.cnn_downsampling

        self.switchout = opt.switchout
        self.varitional_dropout = opt.variational_dropout
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        self.time = opt.time
        self.lsh_src_attention = opt.lsh_src_attention
        self.reversible = opt.src_reversible

        self.temperature = math.sqrt(opt.model_size)

        if self.switchout > 0.0:
            self.word_dropout = 0.0

        feature_size = opt.input_size
        self.channels = 1

        self.lid_network = None

        if opt.gumbel_embedding or opt.lid_loss or opt.bottleneck:
            self.lid_network = LIDFeedForward_small(opt.model_size, opt.model_size, opt.n_languages,
                                                    dropout=opt.dropout)
            self.n_languages = opt.n_languages

        if opt.upsampling:
            feature_size = feature_size // 4

        if not self.cnn_downsampling:
            self.audio_trans = nn.Linear(feature_size, self.model_size)
            torch.nn.init.xavier_uniform_(self.audio_trans.weight)
        else:
            channels = self.channels
            cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
                   nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]

            feat_size = (((feature_size // channels) - 3) // 4) * 32
            # cnn.append()
            self.audio_trans = nn.Sequential(*cnn)
            self.linear_trans = nn.Linear(feat_size, self.model_size)
            # assert self.model_size == feat_size, \
            #     "The model dimension doesn't match with the feature dim, expecting %d " % feat_size

        # if use_cnn:
        #     cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)),
        #            nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std))]
        #     self.cnn = nn.Sequential(*cnn)
        #     input_size = ((((input_size - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1) * 32
        # else:
        #     self.cnn = None

        self.unidirect = self.opt.unidirectional

        self.rnn = nn.LSTM(input_size=self.model_size, hidden_size=self.model_size, num_layers=self.layers,
                           bidirectional=(not self.unidirect), bias=False, dropout=self.dropout, batch_first=True)

        # self.rnn_1 = nn.LSTM(input_size=self.model_size, hidden_size=self.model_size, num_layers=self.layers,
        #                      bias=False, dropout=self.dropout, batch_first=True)
        #
        # self.rnn_2 = nn.LSTM(input_size=self.model_size, hidden_size=self.model_size, num_layers=self.layers,
        #                      bias=False, dropout=self.dropout, batch_first=True)
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.varitional_dropout)
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

    def rnn_fwd(self, seq, mask, hid):
        if mask is not None:
            lengths = mask.sum(-1).float()
            seq = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
            seq, hid = self.rnn(seq, hid)
            seq = pad_packed_sequence(seq, batch_first=True)[0]
        else:
            seq, hid = self.rnn(seq)

        return seq, hid

    # def rnn_fwd_2(self, seq, mask, hid):
    #     # print(mask)
    #     # print(mask.shape)
    #     if mask is not None:
    #         # print(mask.shape)
    #         lengths = mask.sum(-1).float()
    #         lengths_2 = torch.ones(mask.shape[0]) * torch.max(lengths)
    #         seq_2 = pack_padded_sequence(seq, lengths_2, batch_first=True, enforce_sorted=False)
    #
    #         seq_2, hid = self.rnn(seq_2, hid)
    #         seq_2 = pad_packed_sequence(seq_2, batch_first=True)[0]
    #
    #     else:
    #         seq_2, hid = self.rnn(seq)
    #
    #     return seq_2, hid

    def forward(self, input, hid=None):
        # print(input)
        if not self.cnn_downsampling:
            mask_src = input.narrow(2, 0, 1).squeeze(2).gt(onmt.constants.PAD)
            input = input.narrow(2, 1, input.size(2) - 1)
            emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                    input.size(1), -1)
            emb = emb.type_as(input)
        else:

            long_mask = input.narrow(2, 0, 1).squeeze(2).gt(onmt.constants.PAD)
            input = input.narrow(2, 1, input.size(2) - 1)
            # first resizing to fit the CNN format
            input = input.view(input.size(0), input.size(1), -1, self.channels)
            input = input.permute(0, 3, 1, 2)

            input = self.audio_trans(input)
            input = input.permute(0, 2, 1, 3).contiguous()
            input = input.view(input.size(0), input.size(1), -1)
            input = self.linear_trans(input)

            mask_src = long_mask[:, 0:input.size(1) * 4:4]
            # the size seems to be B x T ?
            emb = input
        seq, hid = self.rnn_fwd(emb, mask_src, hid)

        if not self.unidirect:
            hidden_size = seq.size(2) // 2
            seq = seq[:, :, :hidden_size] + seq[:, :, hidden_size:]

        seq = self.postprocess_layer(seq)

        output_dict = {'context': seq.transpose(0, 1), 'src_mask': mask_src}

        return output_dict


class TacotronDecoder(nn.Module):
    def __init__(self, opt):
        super(TacotronDecoder, self).__init__()
        self.n_mel_channels = opt.n_mel_channels
        self.n_frames_per_step = opt.n_frames_per_step
        self.encoder_embedding_dim = opt.model_size
        self.attention_rnn_dim = opt.model_size
        self.decoder_rnn_dim = opt.model_size
        self.prenet_dim = opt.prenet_dim
        self.max_decoder_steps = opt.max_decoder_steps
        self.gate_threshold = 0.5
        self.p_attention_dropout = opt.attn_dropout
        self.p_decoder_dropout = opt.dropout
        self.encoder_type = opt.encoder_type

        self.prenet = Prenet(
            opt.n_mel_channels * opt.n_frames_per_step,
            [opt.prenet_dim, opt.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            opt.prenet_dim + opt.model_size,
            opt.model_size)

        self.attention_layer = Attention(
            opt.model_size, opt.model_size,
            opt.attention_dim, opt.attention_location_n_filters,
            opt.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            opt.model_size + opt.model_size,
            opt.model_size, 1)

        self.linear_projection = LinearNorm(
            opt.model_size + opt.model_size,
            opt.n_mel_channels * opt.n_frames_per_step)

        self.gate_layer = LinearNorm(
            opt.model_size + opt.model_size, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.contiguous().view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """

        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        #  B, prenet_dim + model_dim
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        #  B,  model_dim
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        #  B, 2 ,  Max_time
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        # B, 2* model_dim
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        # B, nmel * nframeperstep
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, src_mask, encoder_out, decoder_inputs, **kwargs):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(encoder_out).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        # (T_out + 1, B, n_mel_channels)
        decoder_inputs = self.prenet(decoder_inputs)
        # (T_out + 1, B, prenet_dim)

        lengths = src_mask.sum(-1).int()

        self.initialize_decoder_states(
            encoder_out, mask=~get_mask_from_lengths(lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            # B, prenet_dim
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, encoder_out):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(encoder_out)

        self.initialize_decoder_states(encoder_out, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > 0.3:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class SpeechAE(nn.Module):
    def __init__(self, encoder, decoder, opt):
        super(SpeechAE, self).__init__()
        self.mask_padding = True
        self.fp16_run = opt.fp16
        self.n_mel_channels = opt.n_mel_channels
        self.n_frames_per_step = opt.n_frames_per_step
        self.encoder = encoder
        self.tacotron_decoder = decoder
        self.postnet = Postnet(opt)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, self.n_frames_per_step)

            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))

            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            slice = torch.arange(0, mask.size(2), self.n_frames_per_step)

            outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)

        return outputs

    def forward(self, batch):
        src = batch.get('source')
        src_org = batch.get('source_org')
        src_lengths = batch.get('src_lengths')
        src_lengths_org = batch.get('src_lengths_org')
        src = src.transpose(0, 1)  # transpose to have batch first

        encoder_output = self.encoder(src)
        encoder_output = defaultdict(lambda: None, encoder_output)
        context = encoder_output['context']
        src_mask = encoder_output['src_mask']
        context = context.transpose(0, 1)

        decoder_input = src_org.narrow(2, 1, src_org.size(2) - 1)
        decoder_input = decoder_input.permute(1, 2, 0)
        mel_outputs, gate_outputs, alignments = self.tacotron_decoder(
            src_mask, context, decoder_input)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            src_lengths_org)

    def inference(self, input):
        encoder_output = self.encoder(input)
        context = encoder_output['context']
        context = context.transpose(0, 1)
        mel_outputs, gate_outputs, alignments = self.tacotron_decoder.inference(
            context)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
