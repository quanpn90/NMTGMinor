import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.Transformer.Layers import XavierLinear as Linear
# Relative Multihead Attention
# Only for self-attention
class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, proj_out=True):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.qkv_net = Linear(d_model, 3 * n_head * d_head, bias=False)

        self.dropatt = nn.Dropout(dropatt)

        if proj_out:
            self.o_net = Linear(n_head * d_head, n_head * d_head, bias=False)
        else:
            self.o_net = None

        # self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        """
        :param h: Height (integer)
        :param w: Width (Integer)
        :param left: Boolean
        :return: Mask (torch.ByteTensor)
        """
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    # efficient computation of B and D term using shift
    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


# Relative Partially Learnable
class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.attn_type = 2

        print("* Attention type: ", self.attn_type)

    def forward(self, input, pos_enc, attn_mask=None, mems=None, debug=False):
        """
        :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param attn_mask:
        :param mems:
        :return: output: projected attention vector T x B x H
        """
        w = input
        r = pos_enc
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.attn_type == 2:
            w_heads = self.qkv_net(w + r)
        elif self.attn_type == 1:
            w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        if debug:
            print("Q nan", torch.isnan(w_head_q).sum() > 0)
            print("K nan", torch.isnan(w_head_k).sum() > 0)
            print("V nan", torch.isnan(w_head_v).sum() > 0)

        if self.attn_type == 2:
            attn_score = torch.einsum('ibnd,jbnd->ijbn', (w_head_q, w_head_k))
        elif self.attn_type == 1:
            r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head
            #### compute attention score
            rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
            AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

            rr_head_q = w_head_q + self.r_r_bias
            BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x

            # what is relative shift?
            # here the B and D are actually B~ and D~ in the paper
            # then shift them to efficiently get B and D
            BD = self._rel_shift(BD)

            attn_score = AC + BD

            if debug:
                print("AC nan", torch.isnan(AC).sum() > 0)
                print("B~D~ nan", torch.isnan(BD).sum() > 0)
                print("BD nan", torch.isnan(BD).sum() > 0)
                print("AC + BD nan", torch.isnan(attn_score).sum() > 0)


        # scale the attention score
        attn_score = attn_score * (self.scale)
        attn_score = attn_score.transpose(0, 2).transpose(1, 3).contiguous()
        # if debug:
        #     print("mask nan", torch.isnan(attn_mask).sum() > 0)
        #     print("attn score before mask nan", torch.isnan(attn_score).sum() > 0)
        if debug:
            attn_score = torch.clamp(attn_score, -0.5, 0.5)
            print(attn_score)
        attn_score = attn_score.float().masked_fill_(attn_mask.unsqueeze(-3), -float('inf')).type_as(attn_score)
        if debug:
            print("attn score after mask nan", torch.isnan(attn_score).sum() > 0)

        #### compute attention probability
        # print(attn_mask).size()
        # if attn_mask is not None and attn_mask.any().item():
        #     if attn_mask.dim() == 2:
        #         attn_score = attn_score.float().masked_fill(
        #             attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
        #     elif attn_mask.dim() == 3:
        #         attn_score = attn_score.float().masked_fill(
        #             attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        # attn_score = attn_score.transpose(0, 2).transpose(1, 3)

        attn_prob = F.softmax(attn_score.float(), dim=-1)
        if debug:
            print(attn_score.size())
            print("attn prob nan", torch.isnan(attn_prob).sum() > 0)
        attn_prob = attn_prob.transpose(0, 2).transpose(1, 3)
        coverage = torch.mean(attn_prob, dim=-1).transpose(0, 2)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        # This is the context vector
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        if self.o_net:
            attn_out = self.o_net(attn_vec)
        else:
            attn_out = attn_vec
        output = attn_out

        # if self.pre_lnorm:
        #     ##### residual connection
        #     output = w + attn_out
        # else:
        #     ##### residual connection + layer normalization
        #     output = self.layer_norm(w + attn_out)

        return output, coverage

    def step(self, input, pos_enc, attn_mask, buffer=None):
        """

        :param attn_mask:
        :param input: 1 x B x H (step by 1)
        :param pos_enc: T x B x H (relative position)
        :param buffer: a dictionary (or None) of buffered computation
        :return:
        """
        w = input
        r = pos_enc
        # qlen is probably 1 here
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.attn_type == 2:
            w_heads = self.qkv_net(w + r)
        elif self.attn_type == 1:
            w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        if buffer is not None and 'k' in buffer and 'v' in buffer and 'q' in buffer:
            w_head_q = torch.cat([buffer['q'], w_head_q], dim=0)  # time first
            buffer['q'] = w_head_q
            w_head_k = torch.cat([buffer['k'], w_head_k], dim=0)  # time first
            buffer['k'] = w_head_k
            w_head_v = torch.cat([buffer['v'], w_head_v], dim=0)  # time first
            buffer['v'] = w_head_v
            # len_key, b_ = proj_key.size(0), proj_key.size(1)
        else:
            if buffer is None:
                buffer = dict()
            buffer['k'] =  w_head_k
            buffer['v'] = w_head_v
            buffer['q'] = w_head_q

        klen = w_head_k.size(0)
        qlen = w_head_q.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        if self.attn_type == 2:
            attn_score = torch.einsum('ibnd,jbnd->ijbn', (w_head_q, w_head_k))
        elif self.attn_type == 1:
            r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head
            #### compute attention score
            rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
            AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
            rr_head_q = w_head_q + self.r_r_bias
            BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x
            # what is relative shift?
            # here the B and D are actually B~ and D~ in the paper
            # then shift them to efficiently get B and D
            BD = self._rel_shift(BD)

            attn_score = AC + BD

        # scale the attention score
        attn_score.mul_(self.scale)

        #### compute attention probability

        # attns = attns.view(b, self.h, len_query, len_key)
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)
        attn_score = attn_score.float().masked_fill_(attn_mask.unsqueeze(-3), -float('inf')).type_as(attn_score)
        # if attn_mask is not None and attn_mask.any().item():
        #     if attn_mask.dim() == 2:
        #         attn_score = attn_score.float().masked_fill(
        #             attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
        #     elif attn_mask.dim() == 3:
        #         attn_score = attn_score.float().masked_fill(
        #             attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)
        attn_prob = F.softmax(attn_score.float(), dim=1).type_as(attn_score)
        coverage = torch.mean(attn_prob, dim=-1).transpose(0, 2)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # take the final time step and then unsqueeze
        attn_vec = attn_vec[-1, :, :].unsqueeze(0)

        # linear projection
        if self.o_net:
            attn_out = self.o_net(attn_vec)
        else:
            attn_out = attn_vec

        output = attn_out

        return output, coverage, buffer


# From Shaw et al 2019
# Not easy to do because there is a max_klen hyperparameter (which is difficult)
class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output