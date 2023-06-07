import torch
from time import time

B = 16384
N_in = 1024
N_out = 4096
num_iters = 200

x = torch.randn(B, N_in, dtype=torch.cfloat, requires_grad=True)
r = torch.randn(B, N_in, dtype=torch.float, requires_grad=True)
i = torch.randn(B, N_in, dtype=torch.float, requires_grad=True)

print(r.type())

r.data.copy_(x.real.data)
i.data.copy_(x.imag.data)

x = x.cuda()
r = r.cuda()
i = i.cuda()

x_2 = torch.randn(N_in, N_out, dtype=torch.cfloat, requires_grad=True)
r_2 = torch.randn(N_in, N_out, dtype=torch.float, requires_grad=True)
i_2 = torch.randn(N_in, N_out, dtype=torch.float, requires_grad=True)

r_2.data.copy_(x_2.real.data)
i_2.data.copy_(x_2.imag.data)

x_2 = x_2.cuda()
r_2 = r_2.cuda()
i_2 = i_2.cuda()

a = torch.mm(x, x_2)

with torch.no_grad():
    a = torch.mm(x, x_2)

    a_r = torch.mm(r, r_2) - torch.mm(i, i_2)
    a_i = torch.mm(r, i_2) + torch.mm(i, r_2)

    print(a.real - a_r)

    print(a.imag - a_i)

torch.cuda.profiler.start()
torch.cuda.synchronize()
start_time = time()

for _ in range(num_iters):

    a_r = torch.mm(r, r_2) - torch.mm(i, i_2)
    a_i = torch.mm(r, i_2) + torch.mm(i, r_2)

    (a_r.sum() + a_i.sum()).backward()

torch.cuda.synchronize()
stop_time = time()

print(F"\nPseudo CMATMUL fp32 {(stop_time - start_time) * 1000. / num_iters:.4f} ms")


torch.cuda.profiler.start()
torch.cuda.synchronize()
start_time = time()

for _ in range(num_iters):

    a = torch.mm(x, x_2)
    (a.real.sum() + a.imag.sum()).backward()

torch.cuda.synchronize()
stop_time = time()
print(F"\nPytorch CMATMUL fp32 time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")



torch.cuda.profiler.start()
torch.cuda.synchronize()
start_time = time()


with torch.cuda.amp.autocast(enabled=True):
    for _ in range(num_iters):

        a_r = torch.mm(r, r_2) - torch.mm(i, i_2)
        a_i = torch.mm(r, i_2) + torch.mm(i, r_2)

        (a_r.sum() + a_i.sum()).backward()

torch.cuda.synchronize()
stop_time = time()
print(F"\nPseudo CMATMUL fp16 {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

torch.cuda.profiler.start()
torch.cuda.synchronize()
start_time = time()

with torch.cuda.amp.autocast(enabled=True):
    for _ in range(num_iters):

        a = torch.mm(x, x_2)
        (a.real.sum() + a.imag.sum()).backward()


torch.cuda.synchronize()
stop_time = time()

print(F"\nPytorch CMATMUL fp16 time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")

