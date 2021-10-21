import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager


# helpers

def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)  # sqrt(2 * nb_features )

    # projection = repeat(projection_matrix, 'j d -> b j d', b=b)
    # projection = projection_matrix[None, None, :, :].expand(b, h, projection_matrix.size(0), projection_matrix.size(1))
    # .repeat(b, h, 1, 1)
    projection = projection_matrix
    projection = projection.type_as(data)

    # data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    data_dash = torch.matmul((data_normalizer * data), projection.transpose(0, 1))

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data -
                          torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    # data_dash = ratio * (torch.exp(data_dash - diag_data) + eps)
    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(),
                       kernel_epsilon=0.001, normalize_data=True, device=None):
    b, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b j d', b=b)
    # projection = projection_matrix[None, None, :, :].repeat(b, h, 1, 1)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def apply_scaling(scale, x):
    return torch.einsum("...n,...nd->...nd", scale, x)


def fast_attention(query, key, value):
    """
    :param query: bsz * n_heads x seq_len x nb_features
    :param key:  bsz * n_heads x seq_len x nb_features
    :param value:  bsz * n_heads x seq_len x head_dim
    :return:
    """
    buffer = torch.cat([key.transpose(1, 2).bmm(value), key.sum(1).unsqueeze(-1)], dim=-1)
    buffer = query.bmm(buffer)
    return apply_scaling(1 / buffer[:, :, -1], buffer[:, :, :-1])


# non-causal linear attention
def linear_attention(q, k, v):
    # print("[linear attention]", q.size(), k.size(), v.size())
    # bsz, heads, len_q, nb = q.size(0), q.size(1), q.size(2), q.size(3)
    # head_dim = v.size(-1)
    # k should be the same as q
    # k = k.view(bsz * heads, len_q, nb)

    # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # b h n d * b h d -> b h n
    # D_inv = 1. / torch.bmm(q.view, k)  #
    # b, h, d = q.size(0), q.size(1), q.size(-1)
    # q = q.view(bsz * heads, len_q, nb)
    # D_inv = 1. / torch.bmm(q, k_cumsum)
    # print("[linear attention dinv]", D_inv.size())

    # v = v.view(bsz * heads, len_q, head_dim)
    # print("[linear attention v]", v.size(), k.size())
    # context = torch.bmm(k.transpose(1, 2).contiguous(), v)   # BH * nb * len_q x BH * len_q * head_dim
    # print("[linear attention context]", context.size())

    #  -> BH * nb * head_dim
    # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)  # b h d e * b h n d * b h n -> b h n e
    # out = torch.bmm(q, context) * D_inv

    # out = out.view(bsz, heads, len_q, head_dim)

    # print("[linear attention out]", out.size())
    # k_cumsum = k.sum(dim=-2)  # B n d -> B d
    # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # b h n d * b h d -> b h n
    # context = torch.einsum('...nd,...ne->...de', k, v)  # b h n d * b h n e -> b h d e ( e = d = head_dim )
    # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    # return out
    query, key, value = q, k, v
    buffer = torch.cat([key.transpose(1, 2).bmm(value), key.sum(1).unsqueeze(-1)], dim=-1)
    buffer = query.bmm(buffer)
    return apply_scaling(1 / buffer[:, :, -1], buffer[:, :, :-1])


def apply_regular_feature_map(x, orf, epsilon=1e-6):
    m, d_k = orf.shape
    proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
    norm = (x ** 2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
    return (torch.exp(proj_x - norm) + epsilon) / math.sqrt(m)


def apply_hyperbolic_feature_map(x, orf, epsilon=1e-6):
    m, d_k = orf.shape
    proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
    proj_x = torch.cat([proj_x, -proj_x], dim=-1)
    norm = (x ** 2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
    return (torch.exp(proj_x - norm) + epsilon) / math.sqrt(2 * m)


def create_orf(d_k, m):
    blocks = torch.randn(math.ceil(m / d_k), d_k, d_k)
    blocks, _ = torch.qr(blocks)
    scale = torch.randn(m, d_k).norm(dim=1)
    return apply_scaling(scale, blocks.reshape(-1, d_k)[:m])


# TODO: maybe name this class FAVOR+ or FAVORSelfAttention
class Performer(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, generalized_attention=False,
                 kernel_fn=nn.ReLU(), no_projection=False):
        super().__init__()
        # nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        nb_features = 32

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_heads, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        self.causal = False

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        len_q, len_k = q.size(-2), k.size(-2)
        bh = q.size(0)   # , q.size(1)

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))

        else:
            # softmax approximation - default option
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        out = linear_attention(q, k, v)
        # print("[performer out]", out.size())
        # attn_weights = out.new(b, h, len_q, len_k).zero_()
        return out, None

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections


# a module for keeping track of when to update the projections

class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        # self.register_buffer('calls_since_last_redraw', torch.tensor(0))
        self.calls_since_last_redraw = 0

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)
            print("draw new random features ...")

            fast_attentions = find_modules(model, Performer)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            # self.calls_since_last_redraw.zero_()
            self.calls_since_last_redraw = 0
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented
