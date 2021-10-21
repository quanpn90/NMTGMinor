import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

len_q = 20
input_dim = 128
heads = 8
head_dim = input_dim // heads
output_dim = input_dim

k_proj = nn.Linear(input_dim, input_dim, bias=True)
v_proj = nn.Linear(input_dim, input_dim, bias=True)
q_proj = nn.Linear(input_dim, input_dim, bias=True)

# weight = Parameter(torch.Tensor(3 * input_dim, input_dim))

weight_t = torch.Tensor(3 * input_dim, input_dim)
bias_t = torch.Tensor(3 * input_dim)
# weight_t = weight_t.reshape(head_dim, 3, heads, input_dim)

w_q = q_proj.weight.clone()
w_k = k_proj.weight.clone()
w_v = v_proj.weight.clone()
print(torch.allclose(w_q, q_proj.weight))
weights = [w_q, w_k, w_v]
# with torch.no_grad():
#     weight_t[:, 0, :, :].reshape(input_dim, input_dim).copy_(q_proj.weight)
#     weight_t[:, 1, :, :].reshape(input_dim, input_dim).copy_(k_proj.weight)
#     weight_t[:, 2, :, :].reshape(input_dim, input_dim).copy_(v_proj.weight)
weight_ = torch.cat(weights, dim=0).contiguous()

b_q = q_proj.bias.clone()
b_k = k_proj.bias.clone()
b_v = v_proj.bias.clone()
biases = [b_q, b_k, b_v]

bias_ = torch.cat(biases, dim=0).contiguous()

weight_ = weight_.reshape(3 * head_dim * heads, input_dim).view(3, heads, head_dim, input_dim).transpose(0, 1).reshape(-1, input_dim)

bias_ = bias_.reshape(3 * head_dim * heads).view(3, heads, head_dim).transpose(0, 1).reshape(-1)
# weight_t = weight_t.reshape(3 * input_dim, input_dim)
weight_t.copy_(weight_)
bias_t.copy_(bias_)
weight = Parameter(weight_t)
bias = Parameter(bias_t)
bsz = 16
input = torch.randn(len_q, bsz, input_dim)

q_proj = q_proj.cuda()
k_proj = k_proj.cuda()
v_proj = v_proj.cuda()

weight = weight.cuda()
bias = bias.cuda()
input = input.cuda()

q = q_proj(input).view(len_q, bsz * heads, head_dim)
k = k_proj(input).view(len_q, bsz * heads, head_dim)
v = v_proj(input).view(len_q, bsz * heads, head_dim)

all = F.linear(input, weight, bias)

# all = all.view(len_q, bsz, 3, heads, head_dim)
#
# q_ = all[:, :,  0, :,:].reshape(len_q, bsz * heads, head_dim)
# k_ = all[:, :,  1, :,:].reshape(len_q, bsz * heads, head_dim)
# v_ = all[:, :,  2, :,:].reshape(len_q, bsz * heads, head_dim)

# all = all.view(len_q, bsz, 3, heads, head_dim).transpose(2, 3).contiguous()
all = all.view(len_q, bsz * heads, 3, head_dim)

q_ = all[:, :, 0, :] # .view(len_q, bsz * heads, head_dim)
k_ = all[:, :, 1, :] # .view(len_q, bsz * heads, head_dim)
v_ = all[:, :, 2, :] # .view(len_q, bsz * heads, head_dim)

# print(q - q_)
print("begin testing ...")
print(torch.allclose(q, q_))
print(torch.allclose(k, k_))
print(torch.allclose(v, v_))

# q_ = q.view(bsz * heads, head_dim)
# k_ = k.view(bsz * heads, head_dim)

# matmul1_results = torch.empty((queries.size(1), queries.size(0), keys.size(0)), dtype=queries.dtype,
#                                           device=queries.device)
# matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0, 1),
#                                 keys.transpose(0, 1).transpose(1, 2),
#                                 out=matmul1_results, beta=0.0, alpha=scale_t[0])

o = torch.bmm
