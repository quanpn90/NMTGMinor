import torch
import torch.nn.functional as F
from time import time

N_in = 1024
N_out = 4096
B = 16384
num_iters = 512

x = torch.randn(B, N_in, dtype=torch.float, requires_grad=True)

W = torch.randn(N_out, N_in, dtype=torch.float, requires_grad=True)

b = torch.randn(N_out, dtype=torch.float, requires_grad=True)

x = x.cuda()
W = W.cuda()
b = b.cuda()


y = F.linear(x, W, b)
y.sum().backward()

y2 = torch.mm(x, W.transpose(0, 1)) + b.unsqueeze(0)
y2.sum().backward()

print(y - y2)

r = torch.randn(1, N_in, dtype=torch.float, requires_grad=True)
s = torch.randn(1, N_out, dtype=torch.float, requires_grad=True)

r = r.cuda()
s = s.cuda()

y1 = F.linear(x, torch.mul(W, torch.mm(s.t(), r)), b)

# y2 = torch.mul(torch.mm(torch.mul(x, r)), s) + b.unsqueeze(0)

y2 = torch.mm(x * r, W.transpose(0, 1)) * s + b.unsqueeze(0)

print("Checking ")

print(y1.sum() / (B * N_out), y2.sum() / (B * N_out))
# print(torch.allclose(y1, y2, rtol=1e-05, atol=1e-08))

rank = 1
n_languages = 1024


r_table = torch.Tensor(n_languages, rank, N_in)
s_table = torch.Tensor(n_languages, rank, N_out)

# indices: [T x B x n_languages]
# r_output: T x B x rank x N_in
# s_output: T x B x rank x N_out
# apply the above equation. torch.mm(x * r, W.transpose(0, 1)) * s + b.unsqueeze(0)


# torch.cuda.profiler.start()
# torch.cuda.synchronize()
# start_time = time()
#
#
# for _ in range(num_iters):
#     y2 = torch.mm(x, W.transpose(0, 1)) + b.unsqueeze(0)
#     y2.sum().backward()
#
# torch.cuda.synchronize()
# stop_time = time()
#
# print(F"\nPseudo CMATMUL fp32 {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
# print("-----------------------------------------------------------------------------")
#
# torch.cuda.profiler.start()
# torch.cuda.synchronize()
# start_time = time()
#
# for _ in range(num_iters):
#     y = F.linear(x, W, b)
#     y.sum().backward()
#
# torch.cuda.synchronize()
# stop_time = time()
# print(F"\nPytorch CMATMUL fp32 time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
# print("-----------------------------------------------------------------------------")
#
#
# torch.cuda.profiler.start()
# torch.cuda.synchronize()
# start_time = time()
#
# with torch.cuda.amp.autocast(enabled=True):
#     for _ in range(num_iters):
#         y = F.linear(x, W, b)
#         y.sum().backward()
#
#
# torch.cuda.synchronize()
# stop_time = time()
#
# print(F"\nPytorch CMATMUL fp16 time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
# print("-----------------------------------------------------------------------------")
#
#
# torch.cuda.profiler.start()
# torch.cuda.synchronize()
# start_time = time()
#
#
# with torch.cuda.amp.autocast(enabled=True):
#     for _ in range(num_iters):
#         y2 = torch.mm(x, W.transpose(0, 1)) + b.unsqueeze(0)
#         y2.sum().backward()
#
# torch.cuda.synchronize()
# stop_time = time()
# print(F"\nPseudo CMATMUL fp16 {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
# print("-----------------------------------------------------------------------------")
