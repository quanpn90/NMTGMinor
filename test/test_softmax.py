import torch
import mask_softmax_dropout_cuda
import copy

BH = 1024 * 8
B = 1024
H = BH // B
Q = 75
K = 56

x = torch.randn((BH, Q, K) , dtype=torch.float16, device=torch.device("cuda"), requires_grad=True) * 100
x_ref = x.clone().detach().requires_grad_(True)

grado = torch.randn((BH, Q, K), dtype=torch.float16, device=torch.device("cuda"), requires_grad=True)

dropout_mask, softmax_results = mask_softmax_dropout_cuda.forward(True, 8, x, 0.0)
pytorch_output = torch.nn.functional.softmax(x_ref, dim=-1, dtype=torch.float32).type_as(x)

dif = softmax_results - pytorch_output
print(dif)
print(dif.double().sum().div_(x.numel()))

result = torch.allclose(softmax_results, pytorch_output, atol=1e-3, rtol=1e-3)

print(result)

print("Checking gradients ...")

grado2 = copy.deepcopy(grado)
grado3 = copy.deepcopy(grado)

pytorch_output.backward(grado)
gradx_ref = x_ref.grad
gradx = mask_softmax_dropout_cuda.backward(8, grado, softmax_results, dropout_mask, 0.0)

gradx2 = mask_softmax_dropout_cuda.backward_recompute(8, grado2, softmax_results, x, dropout_mask, 0.0)

dif = gradx - gradx_ref
print(dif.double().sum().div_(x.numel()))

result = torch.allclose(gradx, gradx_ref, atol=1e-3, rtol=1e-3)
print(result)

dif = gradx2 - gradx_ref
print(dif.double().sum().div_(x.numel()))

result = torch.allclose(gradx2, gradx_ref, atol=1e-3, rtol=1e-3)
print(result)

dif = gradx2 - gradx
print(dif.double().sum().div_(x.numel()))

result = torch.allclose(gradx2, gradx, atol=1e-3, rtol=1e-3)
print(result)




