import torch
import torch.nn as nn
import onmt

from onmt.modules.Transformer.Layers import PrePostProcessing, MultiHeadAttention, FeedForward, Linear
from onmt.modules.RelativeAttention import RelPartialLearnableMultiHeadAttn
from onmt.utils import flip


#   Positional Embedding with discrete inputs
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1):
        super(RelativeTransformerEncoderLayer, self).__init__()

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.preprocess_attn_rev = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)

        self.d_head = d_head = d_model // h
        self.multihead_fwd = RelPartialLearnableMultiHeadAttn(h//2, d_model, d_head, dropatt=attn_p)
        self.multihead_bwd = RelPartialLearnableMultiHeadAttn(h//2, d_model, d_head, dropatt=attn_p)
        self.attn_out = Linear(h * self.d_head , d_model)

        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        elif onmt.Constants.activation_layer == 'linear_swish_linear':
            ff_p = p
            feedforward = FeedForwardSwish(d_model, d_ff, ff_p, static=onmt.Constants.static)
        self.feedforward = feedforward

    def forward(self, input, pos, mask_fwd, mask_bwd):
        """
        :param input:
        :param pos:
        :param mask_fwd:
        :param mask_bwd:
        :return:
        """
        query_fwd = self.preprocess_attn(input)
        pos_rev = torch.clone(pos)

        # reverse the tensor at time dimension
        query_bwd = flip(input, 0)
        query_bwd = self.preprocess_attn_rev(query_bwd)
        out_fwd, _ = self.multihead_fwd(query_fwd, pos, mask_fwd, debug=False)  # T x B x d_head * h/2
        # print("OUTPUT FORWARD NAN", torch.isnan(out_fwd).sum() > 0)
        # print(torch.isnan(out_fwd).sum() > 0 )
        out_bwd, _ = self.multihead_bwd(query_bwd, pos_rev, mask_bwd, debug=True)  # T x B x d_head * h/2
        # print("OUTPUT BACKWARD NAN", torch.isnan(out_bwd).sum() > 0)

        # Flip the bwd states and the concatenate to the input before a final linear transformation
        out = torch.cat([out_fwd, torch.flip(out_bwd, [0])], dim=-1)
        # out = torch.cat([out_fwd, out_bwd], dim=-1)
        # out = self.attn_out(out)
        # out = torch.cat([out_fwd, out_fwd], dim=-1)
        out = self.attn_out(out)
        # out = out_fwd
        input = self.postprocess_attn(out, input)
        # print(torch.isnan(input).sum() > 0)

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input


class RelativeTransformerDecoderLayer(nn.Module):

    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, ignore_source=False):
        super(RelativeTransformerDecoderLayer, self).__init__()
        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
        self.ignore_source = ignore_source

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)
            self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=2)

        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', static=onmt.Constants.static)

        # self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, static=onmt.Constants.static, share=1)
        d_head = d_model // h
        self.multihead_tgt = RelPartialLearnableMultiHeadAttn(h, d_model, d_head, dropatt=attn_p)

        if onmt.Constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, static=onmt.Constants.static)
        elif onmt.Constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        elif onmt.Constants.activation_layer == 'linear_swish_linear':
            ff_p = p
            feedforward = FeedForwardSwish(d_model, d_ff, ff_p, static=onmt.Constants.static)
        self.feedforward = feedforward

    def forward(self, input, pos, context, mask_tgt, mask_src):

        """
        :param input: Embedding (from the last layer) T x B x H
        :param pos: Positional Encoding T x B x H
        :param context:
        :param mask_tgt:
        :param mask_src:
        :return:
        """

        # input and context should be time first ?

        query = self.preprocess_attn(input)

        out, _ = self.multihead_tgt(query, pos, mask_tgt)

        input = self.postprocess_attn(out, input)

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        if not self.ignore_source:
            query = self.preprocess_src_attn(input)
            out, coverage = self.multihead_src(query, context, context, mask_src)
            input = self.postprocess_src_attn(out, input)
        else:
            coverage = None

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, coverage

    def step(self, input, pos, context, mask_tgt, mask_src, buffer=None):
        """ Self attention layer
            layernorm > attn > dropout > residual
        """

        query = self.preprocess_attn(input)

        out, _, buffer = self.multihead_tgt.step(query, pos, mask_tgt, buffer=buffer)

        input = self.postprocess_attn(out, input)

        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        if not self.ignore_source:
            query = self.preprocess_src_attn(input)
            out, coverage, buffer = self.multihead_src.step(query, context, context, mask_src, buffer=buffer)
            input = self.postprocess_src_attn(out, input)
        else:
            coverage = None

        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))
        input = self.postprocess_ffn(out, input)

        return input, coverage, buffer


# class RelativeEncoderLayer(EncoderLayer):
