import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.constants import double_precision


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

    indices = klen - 1 - torch.abs(rel + shift)

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
class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropatt=0, asynchronous=False,
                 tgt_len=None, ext_len=None, mem_len=None, shared_pos_across_heads=False):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        # self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)
        self.shared_pos_across_heads = shared_pos_across_heads
        self.asynchronous = asynchronous

        if not shared_pos_across_heads:
            self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

        # Parameters for the position biases
        # Each head has a different bias
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def compute_attention(self, r, w_head_q, w_head_k, w_head_v, attn_mask=None, debug=False):

        r_w_bias = self.r_r_bias
        r_r_bias = self.r_r_bias

        qlen, rlen, bsz = w_head_q.size(0), r.size(0), w_head_q.size(1)
        rsize = r.size(-1)
        klen = w_head_k.size(0)
        assert rlen >= klen  # can allocate more relative positions than klen

        # mapping d-model to d-head
        if rsize == self.d_model:
            r_head_k = self.r_net(r)
        elif rsize == self.d_head:
            # shared R for each head
            r_head_k = r.unsqueeze(-2).expand(rlen, 1, self.n_head, self.d_head)
        else:
            raise NotImplementedError

        w_head_q = w_head_q.contiguous().view(qlen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_k = w_head_k.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_v = w_head_v.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)

        w_head_q = w_head_q.view(bsz, self.n_head, qlen, self.d_head)
        w_head_k = w_head_k.view(bsz, self.n_head, klen, self.d_head)

        # r_head_k is the projected positions (not depending on the tensors)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head).transpose(0, 1)  # n_head  xrlen x d_head

        # compute attention score
        # r_w_bias is [n_head, d_head]
        rw_head_q = w_head_q + r_w_bias.unsqueeze(1)  # qlen x bsz x n_head x d_head
        AC = torch.matmul(rw_head_q, w_head_k.transpose(2, 3))

        rr_head_q = w_head_q + r_r_bias.unsqueeze(1)
        # [bsz, n_head, q_len, d] > [bsz, n_head, q_len, k_len]
        BD = torch.matmul(rr_head_q, r_head_k.transpose(1, 2))
        # [bsz, n_head, q_len, k_len] to [q_len, k_len, bsz, n_head]
        BD = BD.transpose(0, 2).transpose(1, 3)
        # relative_future_shift gives us 5 4 3 2 1 0 1 2 3 4 5 ... relatives for position at 0
        # BD = _rel_future_shift(BD)
        # Rel shift uses simple view which is faster than torch.gather

        # the input to rel_shift should have size: [qlen, klen, bsz, n_head]
        BD = _rel_shift(BD)

        BD = BD.transpose(0, 2).transpose(1, 3)
        # output size of BD: [bsz, n_head, q_len, k_len]

        # take the first klen results from BD (the rest might not be necessary)
        BD = BD[:, :, :, :klen]

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
        _dtype = torch.float64 if double_precision else torch.float32
        attn_prob = F.softmax(attn_score, dim=-1, dtype=_dtype).type_as(attn_score)

        # nan may happen ... because of the first positions (aligned right) will have nothing to attend to
        if debug:
            nan_mask = torch.isnan(attn_prob)
            attn_prob = attn_prob.masked_fill(nan_mask, 0).type_as(attn_score)

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

    def forward(self, w, r, attn_mask=None, debug=False, mems=None,
                incremental=False, incremental_cache=None):
        """
        :param mems:
        :param attn_mask:
        :param incremental_cache:
        :param incremental:
        :param debug:
        :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param attn_mask:
        :return:
        """

        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            w_heads = self.qkv_net(torch.cat([mems, w], 0))
        else:
            w_heads = self.qkv_net(w)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-qlen:]

        if incremental:
            if 'k' in incremental_cache and 'v' in incremental_cache:
                with torch.no_grad():
                    w_head_k = torch.cat([incremental_cache['k'], w_head_k], dim=0)  # time first
                    incremental_cache['k'] = w_head_k.detach()
                    w_head_v = torch.cat([incremental_cache['v'], w_head_v], dim=0)  # time first
                    incremental_cache['v'] = w_head_v.detach()
            else:
                incremental_cache['k'] = w_head_k.detach()
                incremental_cache['v'] = w_head_v.detach()

            # print(w_head_q.size(), w_head_k.size(), w_head_v.size())

        output, coverage = self.compute_attention(r, w_head_q, w_head_k, w_head_v, attn_mask=attn_mask, debug=debug)

        return output, coverage, incremental_cache


class LearnableRelMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropatt=0, max_len=64,
                 tgt_len=None, ext_len=None, mem_len=None):
        super(LearnableRelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.max_len = max_len

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.position_embedding = nn.Embedding(2 * self.max_len + 1, d_head)

        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.scale = 1 / (d_head ** 0.5)

    def generate_relative_positions(self, qlen, klen, device, caching=False):

        if caching:
            distance_mat = torch.arange(-klen+1, 1, 1).unsqueeze(0)  # 1 x T
            distance_mat = distance_mat.to(device)
        else:
            # assert qlen == klen
            range_vec = torch.arange(klen)  # klen
            range_vec = range_vec.to(device)
            range_mat = range_vec.unsqueeze(-1).expand(-1, klen).transpose(0, 1)
            distance_mat = range_mat - range_mat.transpose(0, 1)  # T x T

        distance_mat_clipped = torch.clamp(distance_mat, min=-self.max_len, max=self.max_len)

        relative_distance = distance_mat_clipped + self.max_len

        return relative_distance

    def forward(self, w, attn_mask=None, debug=False, mems=None,
                incremental=False, incremental_cache=None):
        """
        :param mems:
        :param attn_mask:
        :param incremental_cache:
        :param incremental:
        :param debug:
        :param w: input embeddings (E) T x B x H
        :param r: relative encodings (R)
        :param attn_mask:
        :return:
        """

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            w_heads = self.qkv_net(torch.cat([mems, w], 0))
        else:
            w_heads = self.qkv_net(w)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-qlen:]  # why ? # maybe some stupid thing related to streaming

        if incremental:
            if 'k' in incremental_cache and 'v' in incremental_cache:
                with torch.no_grad():
                    w_head_k = torch.cat([incremental_cache['k'], w_head_k], dim=0)  # time first
                    incremental_cache['k'] = w_head_k.detach()
                    w_head_v = torch.cat([incremental_cache['v'], w_head_v], dim=0)  # time first
                    incremental_cache['v'] = w_head_v.detach()
            else:
                incremental_cache['k'] = w_head_k.detach()
                incremental_cache['v'] = w_head_v.detach()

        q_len = w_head_q.size(0)
        k_len = w_head_k.size(0)

        device_ = w_head_q.device
        r_matrix = self.generate_relative_positions(q_len, k_len, device_, caching=incremental)
        # T x T x H
        r = self.position_embedding(r_matrix)

        output, coverage = self.compute_attention(r, w_head_q, w_head_k, w_head_v, attn_mask=attn_mask, debug=debug)

        return output, coverage, incremental_cache

    def compute_attention(self, r, w_head_q, w_head_k, w_head_v, attn_mask=None, debug=False):

        qlen, rlen, bsz = w_head_q.size(0), r.size(0), w_head_q.size(1)
        rsize = r.size(1)
        klen = w_head_k.size(0)
        assert rlen >= klen  # can allocate more relative positions than klen

        w_head_q = w_head_q.contiguous().view(qlen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_k = w_head_k.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)
        w_head_v = w_head_v.contiguous().view(klen, bsz * self.n_head, self.d_head).transpose(0, 1)

        w_head_q.mul_(self.scale)

        w_head_q = w_head_q.view(bsz, self.n_head, qlen, self.d_head)
        w_head_k = w_head_k.view(bsz, self.n_head, klen, self.d_head)
        w_head_v = w_head_v.view(bsz, self.n_head, klen, self.d_head)

        qk_score = torch.matmul(w_head_q, w_head_k.transpose(2, 3))

        w_head_q_t = w_head_q.permute(2, 0, 1, 3)  # qlen x bsz x n_head x d_head
        w_head_q_r = w_head_q_t.reshape(qlen, bsz * self.n_head, -1)
        r_t = r.transpose(1, 2)  # klen x dhead x klen

        qr_score = torch.matmul(w_head_q_r, r_t)
        qr_score = qr_score.reshape(klen, bsz, self.n_head, -1)
        qr_score = qr_score.permute(1, 2, 0, 3)

        attn_score = qk_score + qr_score

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
        dtype_ = torch.float64 if double_precision else torch.float32
        attn_prob = F.softmax(attn_score, dim=-1, dtype=dtype_).type_as(attn_score)

        # nan may happen ... because of the first positions (aligned right) will have nothing to attend to
        # nan_mask = torch.isnan(attn_prob)
        # attn_prob = attn_prob.masked_fill(nan_mask, 0).type_as(attn_score)

        coverage = attn_prob
        attn_prob = self.dropatt(attn_prob)

        context_org = torch.matmul(attn_prob, w_head_v)

        attn_t = attn_prob.permute(2, 0, 1, 3)
        attn_r = attn_t.reshape(klen, bsz * self.n_head, -1)

        # what is r size?
        context_pos = torch.matmul(attn_r, r)
        context_pos = context_pos.reshape(klen, bsz, self.n_head, -1)
        context_pos = context_pos.permute(1, 2, 0, 3)

        attn_vec = context_org + context_pos

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(bsz, qlen, self.n_head * self.d_head)
        # linear projection and transpose to T B D
        attn_out = self.o_net(attn_vec).transpose(0, 1)

        output = attn_out

        return output, coverage


if __name__ == '__main__':
    bsz = 1
    n_head = 8

    tgt_len = 10
    src_len = 6

    qlen = 5
    klen = 12

    pos = torch.arange(klen - 1, -klen, -1.0).unsqueeze(1).expand(-1, bsz)  # T x B
    pos = pos.unsqueeze(0).expand(klen, -1, -1)

    print(pos.size())
    pos = _rel_shift(pos)

    print(pos.size())
    print(pos.squeeze(-1))

    # x = torch.arange(klen - 1, -klen, -1.0).unsqueeze(0).repeat(qlen, 1)
    # input = x.mul(10)
    # print(input, input.size())

    # x = torch.randint(0, 100, (tgt_len, tgt_len))
    #
    # print(x)
    #
    # tgt_tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
    # tgt_src_mask = torch.zeros(tgt_len, src_len)
    #
    # tgt_mask = torch.cat([tgt_src_mask, tgt_tgt_mask], dim=-1)
    # print(tgt_mask)
    # # print(attn_mask)
    #
    # src_src_mask = torch.zeros(src_len, src_len)
    # src_tgt_mask = torch.ones(src_len, tgt_len)
    #
    # src_mask = torch.cat([src_src_mask, src_tgt_mask], dim=-1)
    #
    # print(src_mask)
    # print("FULL ATTENTION MASK")
    # attn_mask = torch.cat([src_mask, tgt_mask], dim=0)
    #
    # print(attn_mask)

