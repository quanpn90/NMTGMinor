import torch.nn as nn
import torch
import math


#  Positional Embedding with discrete inputs
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(SinusoidalPositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, sin_first=True, bsz=None):
        """
        :param bsz: integer to repeat
        :param pos_seq: sequences of RELATIVE position indices (can be negative for future)
        :param sin_first: in Attention is all you need paper, sin is first then cosin
        """
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq.type_as(pos_seq))

        if sin_first:
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        else:
            pos_emb = torch.cat([sinusoid_inp.cos(), sinusoid_inp.sin()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].repeat(1, bsz, 1)
        else:
            return pos_emb[:, None, :]


class FastSinusoidalPositionalEncoding(nn.Module):
    """Adds positional embeddings to standard word embeddings
    This matches the original TensorFlow implementation at
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.

    Args:
        d_model: dimension of model
        p:       dropout probability
        len_max: max seq length for pre-calculated positional embeddings

    Inputs Shapes:
        word_emb: batch_size x len_seq x d_model

    Outputs Shapes:
        out:   batch_size x len_seq x d_model

    """

    def __init__(self, d_model, p=0, len_max=1024):
        # save a fixed positional embedding matrix up to len_max,
        # so that no need to recreate it everytime
        super(FastSinusoidalPositionalEncoding, self).__init__()
        self.len_max = len_max
        self.d_model = d_model
        self.data_type = None

        self.renew(len_max)
        self.p = p

    def renew(self, new_max_len):
        # detele the old variable to avoid Pytorch's error when register new buffer
        cuda = False
        if hasattr(self, 'pos_emb'):
            cuda = self.pos_emb.is_cuda
            # self.data_type = torch.type(self.pos_emb)
            del self.pos_emb

        position = torch.arange(0, new_max_len).float()

        num_timescales = self.d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)

        if cuda:
            pos_emb = pos_emb.cuda()

        if self.data_type is not None:
            pos_emb.type(self.data_type)
        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('pos_emb', pos_emb)
        # self.data_type = self.pos_emb.type()
        self.len_max = new_max_len

    def forward(self, word_emb, t=None):
        """
        :param word_emb: Tensor [BxTxH] (batch first)
        :param t: integer
        :return:
        """
        len_seq = t if t else word_emb.size(1)

        self.data_type = word_emb.type()

        if len_seq > self.len_max:
            self.renew(len_seq)

        if word_emb.size(1) == len_seq:
            time_ = self.pos_emb[:len_seq, :].type_as(word_emb)
            out = word_emb + time_
        else:
            # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
            time_emb = self.pos_emb[len_seq - 1, :]  # 1 x dim
            # out should have size bs x 1 x dim
            out = word_emb + time_emb.detach().unsqueeze(0).type_as(word_emb)
            # repeat(word_emb.size(0), 1, 1).type_as(word_emb)
        return out

    def get_positional_embeddings(self, word_emb, t=None):

        len_seq = t if t else word_emb.size(1)

        self.data_type = word_emb.type()
        if len_seq > self.len_max:
            self.renew(len_seq)

        if word_emb.size(1) == len_seq:
            time_emb = self.pos_emb[:len_seq, :].type_as(word_emb)

        else:
            time_emb = self.pos_emb[len_seq - 1, :].unsqueeze(0).type_as(word_emb)

        return time_emb
