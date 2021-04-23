import torch
import mask_softmax_dropout_cuda

BH = 1024 * 8
B = 1024
H = BH // B
Q = 75
K = 56

x = torch.randn((BH, Q, K), dtype=torch.float16, device=torch.device("cuda"), requires_grad=True)
x_ref = x.clone().detach().requires_grad_(True)

grado = torch.randn((BH, Q, K), dtype=torch.float16, device=torch.device("cuda"), requires_grad=True)

mask = x.new_zeros(B, K).bernoulli_(p=0.2).byte()
# mask = (torch.randn(self.sequences, self.seq_length)>0)

# null_tensor    = torch.tensor([])
# mask = null_tensor
dropout_result,  dropout_mask, softmax_results  = mask_softmax_dropout_cuda.forward(True, False, 8, x, mask, 0.0)
x_masked = x_ref.view(B, H, Q, K).masked_fill(mask.unsqueeze(1).unsqueeze(2).bool(), float('-inf')).view(BH, Q, K)
pytorch_output = torch.nn.functional.softmax(x_masked, dim=-1)
y_ref = torch._fused_dropout(pytorch_output, 1.0)


dif = softmax_results - pytorch_output
print(dif)
print(dif.double().sum().div_(x.numel()))


result = torch.allclose(softmax_results, pytorch_output, atol=1e-2, rtol=1e-2)

print(result)

print("Checking gradients ...")

pytorch_output.backward(grado)
gradx_ref = x_ref.grad
gradx = mask_softmax_dropout_cuda.backward(8, grado, softmax_results, dropout_mask, mask, 0.0)


dif = gradx - gradx_ref
print(dif.double().sum().div_(x.numel()))

result = torch.allclose(gradx, gradx_ref, atol=1e-1, rtol=1e-1)
print(result)

print("--------------------------------------------------------------------")
print("Time Mask testing for self-attention")

x = torch.randn((BH, Q, Q), dtype=torch.float16, device=torch.device("cuda"), requires_grad=True)
x_ref = x.clone().detach().requires_grad_(True)

grado = torch.randn((BH, Q, Q), dtype=torch.float16, device=torch.device("cuda"), requires_grad=True)

mask = x.new_zeros(Q, Q).bernoulli_(p=0.2).byte()

# istraining is True and time mask is True
dropout_result, dropout_mask, softmax_results = mask_softmax_dropout_cuda.forward(True, True, 8, x, mask, 0.0)
x_masked = x_ref.view(BH, Q, Q).masked_fill(mask.unsqueeze(0).bool(), float('-inf')).view(BH, Q, Q)
pytorch_output = torch.nn.functional.softmax(x_masked, dim=-1)
y_ref = torch._fused_dropout(pytorch_output, 1.0)

dif = softmax_results - pytorch_output
print(dif)
print(dif.double().sum().div_(x.numel()))


result = torch.allclose(softmax_results, pytorch_output, atol=1e-1, rtol=1e-1)

print(result)

print("Checking gradients ...")

pytorch_output.backward(grado)
gradx_ref = x_ref.grad
gradx = mask_softmax_dropout_cuda.backward(8, grado, softmax_results, dropout_mask, mask, 0.0)


dif = gradx - gradx_ref
print(dif.double().sum().div_(x.numel()))

result = torch.allclose(gradx, gradx_ref, atol=1e-2, rtol=1e-2)
print(result)


