import torch.nn as nn
import copy
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, or_masks, create_mask
from triton.testing import do_bench
from functools import partial

torch.set_default_device('cuda')

T = 512

B = 8
H = 16
q, k, v = [torch.randn(B, 16, T, 64, requires_grad=True, dtype=torch.bfloat16) for _ in range(3)]

lengths_up = torch.tensor([300, 300, 300, 300, 300, 300, 300, 300])
length_down = torch.tensor([46, 50, 55, 13, 34, 67, 43, 66])


def causal_mask(b, h, q, kv):
    return q >= kv


def get_mask_from_lengths(lengths_up, length_down):
    max_len = T
    ids = torch.arange(0, max_len).to(lengths_up.device)

    mask = (ids < lengths_up.unsqueeze(1)) & (ids > length_down.unsqueeze(1))

    mask = mask.unsqueeze(1).unsqueeze(2).broadcast_to((B, H, T, T))
    return mask.cuda()


def prefix_full(b, h, q, kv, prefix_lengths):
    return kv <= prefix_lengths[b]


padding_mask = get_mask_from_lengths(lengths_up, length_down)
# print(padding_mask[0,0:300])

def padding_mask_(b, h, q, kv):
    return padding_mask[b, h, q, kv]


padding_sparsity_mask = create_block_mask(padding_mask_, B, H, T, T)

# short_prefixes = torch.randint(512, 1024, (B,), dtype=torch.int)
# short_prefix_mask = create_block_mask(or_masks(causal_mask, partial(prefix_full, prefix_lengths=short_prefixes)), B,
#                                       None, T, T)

# long_prefixes = torch.randint(2048, 2048 + 1024, (B,), dtype=torch.int)
# long_prefix_mask = create_block_mask(or_masks(causal_mask, partial(prefix_full, prefix_lengths=long_prefixes)), B, None,
#                                      3840, 3840)

# print("short prefixes: ", short_prefix_mask)
flex_attention = torch.compile(flex_attention)
print(": padding sparsity", do_bench(lambda: flex_attention(q, k, v, block_mask=padding_sparsity_mask).sum().backward()))

# mask = create_mask(or_masks(causal_mask, partial(prefix_full, prefix_lengths=short_prefixes)), B, 1, T, T)

mask = padding_mask

print(mask.size())

print("masking rate", mask.int().sum() / mask.numel())

print("xformers/sdpa with mask: ",
      do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask).sum().backward()))
print("FA (full): ", do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v).sum().backward()))