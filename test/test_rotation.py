import torch

import torch
from torch import nn, einsum
from einops import rearrange, repeat


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        """
        :param x: [bsz x time x hidden]
        :return:
        """
        # actually this module doesn't care about anything of x except x.size(1)

        n = x.shape[0]  # time dimension
        t = torch.arange(n, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


def rotate_every_two(x):

    # splits the last dimension in half
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)

    # stack negative x2 with x1
    x = torch.stack((-x2, x1), dim=-1)

    return rearrange(x, '... d j -> ... (d j)')


def rotate_backward(dx):

    dx = rearrange(dx, '... (d j) -> ... d j', j=2)

    dx2, dx1 = dx.unbind(dim=-1)

    dx = torch.stack((dx1, -dx2), dim=-1)

    dx = rearrange(dx, '... d j -> ... (d j)')

    return dx


# more like encodings because the position values are not learnablew weights
def apply_rotary_emb(q, sinu_pos):
    """
    :param q:  [bsz x time x hidden]
    :param k:  [bsz x time x hidden]
    :param sinu_pos:
    :return: q and k with applied position encoding
    """
    # splits the last dimension of the sinu_pos in half and grab sin and cos terms
    sinu_pos = rearrange(sinu_pos, 'n (j d) -> n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)

    # repeat the sin and cos terms with 2?
    sin, cos = map(lambda t: repeat(t, 'n d -> n (d j)', j=2), (sin, cos))

    # q' = (q * cos) + (rotate_every_two(q) * sin)
    # dl_dq = dl_dq' * (cos + sin * rotate'(q))
    print(q.size(), cos.size(), sin.size())
    q = q * cos.unsqueeze(1) + rotate_every_two(q) * sin.unsqueeze(1)
    # q = rotate_every_two(q)  # * sin

    # y = g(x) * a
    # dy/dx = dy/dg * dg/dx = a *
    # q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, sin, cos


BH = 1024 * 8
B = 1024
H = BH // B
Q = 75
K = 56
D = 64

pos_encoder = SinusoidalEmbeddings(D)
pos_encoder.cuda()

# create input
x = torch.randn((BH, Q, D), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)

# create the pos emb
pos_emb = pos_encoder(x)

rotate_grad = torch.Tensor([1, -1] * int(D / 2)).to(x.device)
rotate_grad = rotate_grad.unsqueeze(0).unsqueeze(1).repeat(BH, Q, 1)

#
r_x = rotate_every_two(x)
#
loss = r_x.sum() * 1
#
loss.backward()
#
print(x.grad - rotate_grad)
x.grad = None



x = torch.randn((Q, BH, D), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
grad_rx = torch.randn((Q, BH, D), dtype=torch.float32, device=torch.device("cuda"), requires_grad=False)

pos_emb = pos_encoder(x)
rotary_emb_x, sin, cos = apply_rotary_emb(x, pos_emb)
rotary_emb_x.backward(grad_rx)

print(x.grad)

rotate_grad = rotate_backward(x.new_ones(x.shape))

# grad_x = (cos + rotate_grad * sin) * grad_rx
grad_x = cos.unsqueeze(1) * grad_rx + rotate_backward(sin.unsqueeze(1) * grad_rx)

print(x.grad - grad_x)
