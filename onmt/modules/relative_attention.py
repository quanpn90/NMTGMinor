import torch
import torch.nn as nn
import torch.nn.functional as F


def _rel_shift(x, zero_triu=False):
    # zero_pad size: [q_len, 1, bsz, n_head]

    zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                           device=x.device, dtype=x.dtype)

    x_padded = torch.cat([zero_pad, x], dim=1)

    x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

    x = x_padded[1:].view_as(x)

    # fills the 'unnecessary' parts with zeros
    if zero_triu:
        ones = torch.ones((x.size(0), x.size(1)))
        x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

    return x


def _rel_future_shift(x):
    """
    x input dimension: [qlen, klen, bsz, nhead]
    """

    qlen, klen = x.size(0), x.size(1)

    # adding the device here is MUCH faster than using device after expanding
    rel = torch.arange(klen - qlen, -qlen, -1, device=x.device).unsqueeze(0)
    shift = torch.arange(0, qlen, 1, device=x.device).unsqueeze(1)

    indices = klen - 1 - torch.abs(rel+shift)

    # expanding to the batch size and head dimensions
    for i in range(x.dim() - 2):
        indices = indices.unsqueeze(-1)

    indices = indices.expand_as(x)

    output_ = torch.gather(x, 1, indices)

    return output_


# Relative Multihead Attention
class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        # self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        # self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        # self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def _parallelogram_mask(self, h, w, left=False):
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
    # x dimension: [q_len, k_len, bsz, n_head]
    def _rel_shift(self, x, zero_triu=False):

        # zero_pad size: [q_len, 1, bsz, n_head]
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)

        x_padded = torch.cat([zero_pad, x], dim=1)

        # x_padded:
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


# Relative Partially Learnable (from Transformer XL)
class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, debug=False):
        """
        :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param r_w_bias: n_head * d_head
        :param r_r_bias: n_head * d_head (the global relative position bias)
        :param attn_mask:
        :param mems:
        :return:
        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            # w_heads = self.qkv_net(self.layer_norm(w))
            w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.contiguous().view(qlen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_k = w_head_k.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_v = w_head_v.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)

        w_head_q = w_head_q.view(bsz, self.n_head, qlen, self.d_head)
        w_head_k = w_head_k.view(bsz, self.n_head, klen, self.d_head)

        # r_head_k is the projected positions (not depending on the tensors)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head).transpose(0, 1)  # qlen x n_head x d_head

        #### compute attention score
        # r_w_bias is [n_head, d_head]
        rw_head_q = w_head_q + r_w_bias.unsqueeze(1)  # qlen x bsz x n_head x d_head
        AC = torch.matmul(rw_head_q, w_head_k.transpose(2, 3))

        rr_head_q = w_head_q + r_r_bias.unsqueeze(1)
        # [bsz, n_head, q_len, d] > [bsz, n_head, q_len, k_len]
        BD = torch.matmul(rr_head_q, r_head_k.transpose(1, 2))
        # [bsz, n_head, q_len, k_len] to [q_len, k_len, bsz, n_head]
        BD = BD.transpose(0, 2).transpose(1, 3)
        # relative_future_shift gives us 5 4 3 2 1 0 1 2 3 4 5 ... relatives for position at 0
        BD = _rel_future_shift(BD)
        BD = BD.transpose(0, 2).transpose(1, 3)

        # [bsz x n_head x qlen x klen]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # [qlen x klen x bsz x n_head]
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                        attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [bsz x n_head x qlen x klen] again
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)

        # [bsz x n_head x qlen x klen] again
        attn_prob = F.softmax(attn_score.float(), dim=-1)

        # nan will happen ... because of the first positions (aligned right) they will have nothing to attend to
        nan_mask = torch.isnan(attn_prob)
        attn_prob = attn_prob.masked_fill(nan_mask, 0).type_as(attn_score)

        if debug:
            n = nan_mask.byte().sum()
            total = attn_prob.numel()
            print("Total nan: %d %d " % (n, total))
        coverage = attn_prob

        attn_prob = self.dropatt(attn_prob)

        attn_prob = attn_prob.view(bsz * self.n_head, qlen, klen)
        # compute attention vector
        attn_vec = torch.bmm(attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.transpose(0, 1).contiguous().view(qlen, bsz, self.d_model)

        # linear projection
        attn_out = self.o_net(attn_vec)

        output = attn_out

        return output, coverage

    def step(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, buffer=None, debug=False):
        """
        :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param r_w_bias: n_head * d_head
        :param r_r_bias: n_head * d_head (the global relative position bias)
        :param attn_mask:
        :param mems:
        :return:
        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            raise NotImplementedError
        #     # cat = torch.cat([mems, w], 0)
        #     # w_heads = self.qkv_net(cat)
        #     # r_head_k = self.r_net(r)
        #     #
        #     # w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        #     # w_head_q = w_head_q[-qlen:]
        # else:
            # w_heads = self.qkv_net(self.layer_norm(w))
        w_heads = self.qkv_net(w)
        r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        if buffer is not None and 'k' in buffer and 'v' in buffer:
            w_head_k = torch.cat([buffer['k'], w_head_k], dim=0)  # time first
            buffer['k'] = w_head_k
            w_head_v = torch.cat([buffer['v'], w_head_v], dim=0)  # time first
            buffer['v'] = w_head_v
        else:
            if buffer is None:
                buffer = dict()
            buffer['k'] = w_head_k
            buffer['v'] = w_head_v

        klen = w_head_k.size(0)

        w_head_q = w_head_q.contiguous().view(qlen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_k = w_head_k.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_v = w_head_v.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)

        w_head_q = w_head_q.view(bsz, self.n_head, qlen, self.d_head)
        w_head_k = w_head_k.view(bsz, self.n_head, klen, self.d_head)

        # r_head_k is the projected positions (not depending on the tensors)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head).transpose(0, 1)  # qlen x n_head x d_head

        #### compute attention score
        # r_w_bias is [n_head, d_head]
        rw_head_q = w_head_q + r_w_bias.unsqueeze(1)  # qlen x bsz x n_head x d_head
        AC = torch.matmul(rw_head_q, w_head_k.transpose(2, 3))
        # AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        # w_head_q = [
        rr_head_q = w_head_q + r_r_bias.unsqueeze(1)
        # [bsz, n_head, q_len, d] > [bsz, n_head, q_len, k_len]
        BD = torch.matmul(rr_head_q, r_head_k.transpose(1, 2))

        BD = BD.transpose(0, 2).transpose(1, 3)
        # relative_future_shift gives us 5 4 3 2 1 0 1 2 3 4 5 ... relatives for position at 0
        BD = _rel_future_shift(BD)
        BD = BD.transpose(0, 2).transpose(1, 3)

        # [bsz x n_head x qlen x klen]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # [qlen x klen x bsz x n_head]
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                        attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [bsz x n_head x qlen x klen] again
        attn_score = attn_score.transpose(0, 2).transpose(1, 3)

        # [bsz x n_head x qlen x klen] again
        attn_prob = F.softmax(attn_score.float(), dim=-1)

        # nan will happen ... because of the first positions (aligned right) they will have nothing to attend to
        nan_mask = torch.isnan(attn_prob)
        attn_prob = attn_prob.masked_fill(nan_mask, 0).type_as(attn_score)

        if debug:
            n = nan_mask.byte().sum()
            total = attn_prob.numel()
            print("Total nan: %d %d " % (n, total))
        coverage = attn_prob

        attn_prob = self.dropatt(attn_prob)

        attn_prob = attn_prob.view(bsz * self.n_head, qlen, klen)
        # compute attention vector
        # attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = torch.bmm(attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.transpose(0, 1).contiguous().view(qlen, bsz, self.d_model)

        # linear projection
        attn_out = self.o_net(attn_vec)
        output = attn_out

        return output, coverage, buffer


# Learnable (no sin/cos position encoding)
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

        output = attn_out

        return output


# Relative Partially Learnable (from Transformer XL)
class SlowRelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(SlowRelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        """

            :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param r_w_bias: n_head * d_head
        :param r_r_bias: n_head * d_head (the global relative position bias)
        :param attn_mask:
        :param mems:
        :return:
        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            w_heads = self.qkv_net(self.layer_norm(cat))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(self.layer_norm(w))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        # what is relative shift?
        # R is actually the
        # here the B and D are actually B~ and D~ in the paper
        # then shift them to efficiently get B and D
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                        attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score.float(), dim=1)

        # nan will happen ... because of the first positions (aligned right) they will have nothing to attend to
        nan_mask = torch.isnan(attn_prob)
        attn_prob = attn_prob.masked_fill(nan_mask, 0).type_as(attn_score)

        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)

        output = attn_out

        return output


if __name__ == '__main__':

    bsz = 5
    n_head = 8

    qlen = 5
    klen = 12

    x = torch.arange(klen - 1, -1, -1.0).unsqueeze(0).repeat(qlen, 1)
    input = x.mul(10)
    print(input)
    # print(x)
    # x = x.unsqueeze(-1).unsqueeze(-1)
    # x = torch.Tensor(qlen, klen, bsz, n_head)
    # x.normal_(0, 1)

    output = _rel_shift(input, zero_triu=False)

    # print(x)
    # print(x.size())
    print("REL SHIFT 1 RESULT")
    print(output)

    idx = torch.arange(0, klen).unsqueeze(0).repeat(qlen, 1)
    print(idx)
    shifted_idx = _rel_shift(idx)
    print(shifted_idx)

    output_2 = torch.gather(input, 1, shifted_idx)
    print("REL SHIFT 2 RESULT")
    print(output_2)

    # a = torch.arange(klen - qlen, -qlen, -1).unsqueeze(0)
    # print(a)
    # a = torch.arange(-qlen + 1, klen - qlen + 1, 1).unsqueeze(0)
    # a = torch.cat([torch.arange(0, klen-qlen), torch.arange(qlen, -1, -1)]).unsqueeze(0)
    # print(a)
    # a = torch.arange(0, klen).unsqueeze(0)
    # print(a)

    # b = torch.arange(0, qlen, 1).unsqueeze(1)
    # print(b)

    # c = (a + b)
    # print(c)

    a = torch.arange(klen - qlen, -qlen, -1).unsqueeze(0)
    # print(a)

    b = torch.arange(0, qlen, 1).unsqueeze(1)
    # print(b)
    # print(c)
    # print(a+b)
    c = torch.abs(a+b)
    rearranged_idx = klen - 1 - c

    output_3 = torch.gather(input, 1, rearranged_idx)
    print("REL SHIFT 3 RESULT")
    print(output_3)

    # input_repeat = input.unsqueeze(-1).unsqueeze(-1)
    #
    # input_repeat = input_repeat.repeat(1, 1, bsz, n_head)
    #
    # idx = rearranged_idx.unsqueeze(-1).unsqueeze(-1).expand_as(input_repeat)
    # output_4 = torch.gather(input_repeat, 1, idx)

    print("REL SHIFT 4 RESULT")
    output_4 = _rel_future_shift(input)
    print(output_4)

    input_repeat = input.unsqueeze(-1).unsqueeze(-1)

    input_repeat = input_repeat.repeat(1, 1, bsz, n_head)
    output_5 = _rel_future_shift(input_repeat)
    print(output_5)
