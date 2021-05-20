import torch

try:
    import apex.amp as amp
    from apex.amp import half_function
except (ModuleNotFoundError, ImportError) as e:
    amp = None
    from .compat import half_function

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .compat import custom_fwd, custom_bwd

try:
    import fused_dropout_add_cuda
except (ModuleNotFoundError, ImportError) as e:
    fused_dropout_add_cuda = None


class FusedDropoutAdd(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, residual, dropout_prob, is_training):

        null_tensor = torch.tensor([])
        dropout_prob_t = torch.tensor([dropout_prob])

        if fused_dropout_add_cuda is not None and input.is_cuda and input.type() == 'torch.cuda.HalfTensor':
            # print("Fused dropout add")
            dropout_prob_ = dropout_prob if is_training else 0.0
            dropout_mask, output = fused_dropout_add_cuda.forward(True, input, residual, dropout_prob_)
        else:
            if is_training:
                dropout_results, dropout_mask = torch._fused_dropout(input, p=(1. - dropout_prob))
            else:
                dropout_mask = null_tensor
                dropout_results = input
            output = dropout_results + residual

        ctx.save_for_backward(dropout_mask, dropout_prob_t)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grads):

        dropout_mask, dropout_prob_t = ctx.saved_tensors

        if fused_dropout_add_cuda is not None and output_grads.is_cuda and output_grads.dtype == torch.float16:
            grad_input = fused_dropout_add_cuda.backward(output_grads, dropout_mask, dropout_prob_t[0])
        else:
            grad_input = torch._masked_scale(output_grads, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))

        return grad_input, output_grads, None, None


@half_function
def fused_dropout_add(input, residual, dropout, is_training):

    return FusedDropoutAdd.apply(input, residual, dropout, is_training)

#
# if __name__ == '__main__':
#
#     batch_size = 24568
#     hidden_size = 1024
#     num_iters = 10
#
#     class TestMLP(unittest.TestCase):
#
#         def test_creation(self):
#             MLP(mlp_sizes)

        # def test_numeric(self):
        #     dropout_rate = 0.0
        #
        #     for _ in range(1):
        #         bsz = random.randint(2850, batch_size // 8) * 8
        #         test_input = torch.empty(bsz, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
        #         ref_input = test_input.clone().detach().requires_grad_()
        #         mlp_out, dropout_mask = mlp(test_input)
        #         ref_out = ref_mlp.forward(ref_input, dropout_mask, ref=True)
        #
        #         print(dropout_mask.sum() / dropout_mask.numel())
        #         np.testing.assert_allclose(
        #             mlp_out.detach().cpu().numpy(),
        #             ref_out.detach().cpu().numpy(),
        #             atol=1e-5, rtol=1e-4)
        #
        #         # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        #         mlp_out.mean().mul(10.).backward()
        #         ref_out.mean().mul(10.).backward()
        #         np.testing.assert_allclose(
        #             test_input.grad.detach().cpu().numpy(),
        #             ref_input.grad.detach().cpu().numpy(),
        #             atol=1e-7, rtol=1e-5)
        #         np.testing.assert_allclose(
        #             mlp.biases[0].grad.detach().cpu().numpy(),
        #             ref_mlp.biases[0].grad.detach().cpu().numpy(),
        #             atol=1e-7, rtol=1e-5)
        #
        # def test_with_bias(self):
        #     for use_activation in ['relu']:
        #         mlp = MLP(mlp_sizes, activation=use_activation).cuda()
        #
        #         ref_mlp = deepcopy(mlp)
        #
        #         test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.).requires_grad_()
        #         ref_input = test_input.clone().detach().requires_grad_()
        #         mlp_out, dropout_mask = mlp(test_input)
        #         ref_out = ref_mlp(ref_input, dropout_mask, ref=True)
        #         np.testing.assert_allclose(
        #             mlp_out.detach().cpu().numpy(),
        #             ref_out.detach().cpu().numpy(),
        #             atol=1e-7, rtol=1e-5)
        #
        #         # Use mean value as scalar loss. Multiply 10 to make it big enough not zero out
        #         mlp_out.mean().mul(10.).backward()
        #         ref_out.mean().mul(10.).backward()
        #         np.testing.assert_allclose(
        #             test_input.grad.detach().cpu().numpy(),
        #             ref_input.grad.detach().cpu().numpy(),
        #             atol=0, rtol=1)
        #
        #         for l in range(mlp.num_layers):
        #             np.testing.assert_allclose(
        #                 mlp.weights[l].grad.detach().cpu().numpy(),
        #                 ref_mlp.weights[l].grad.detach().cpu().numpy(),
        #                 atol=1e-7, rtol=1)
        #             np.testing.assert_allclose(
        #                 mlp.biases[l].grad.detach().cpu().numpy(),
        #                 ref_mlp.biases[l].grad.detach().cpu().numpy(),
        #                 atol=1e-7, rtol=1e-5)
        #
        # def test_no_grad(self):
        #     mlp = MLP(mlp_sizes).cuda()
        #     ref_mlp = deepcopy(mlp)
        #
        #     test_input = torch.empty(batch_size, mlp_sizes[0], device="cuda").uniform_(-1., 1.)
        #     ref_input = test_input.clone().detach()
        #     mlp_out, dropout_mask = mlp(test_input)
        #
        #     ref_out = ref_mlp(ref_input, dropout_mask, ref=True)
        #     np.testing.assert_allclose(
        #         mlp_out.detach().cpu().numpy(),
        #         ref_out.detach().cpu().numpy(),
        #         atol=1e-7, rtol=1e-5)
        #
        # def test_performance_half(self):
        #     mlp = MLP(mlp_sizes).cuda().half()
        #
        #     mlp_layers = []
        #     for i in range(mlp.num_layers):
        #         linear = nn.Linear(mlp_sizes[i], mlp_sizes[i + 1])
        #         mlp.weights[i].data.copy_(linear.weight)
        #         mlp.biases[i].data.copy_(linear.bias)
        #         mlp_layers.append(linear)
        #         if i < mlp.num_layers - 1:
        #             # mlp_layers.append(nn.ReLU(inplace=True))
        #             mlp_layers.append(torch.nn.GELU())
        #             mlp_layers.append(nn.Dropout(0.25))
        #
        #     ref_mlp = nn.Sequential(*mlp_layers).cuda().half()
        #
        #     test_input = torch.empty(
        #         batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()
        #     ref_input = torch.empty(
        #         batch_size, mlp_sizes[0], device="cuda", dtype=torch.half).fill_(10.).requires_grad_()
        #
        #     # Warm up GPU
        #     for _ in range(100):
        #         ref_out = ref_mlp(ref_input)
        #         ref_loss = ref_out.mean()
        #         ref_mlp.zero_grad()
        #         ref_loss.backward()
        #         mlp_out, _ = mlp(test_input)
        #         test_loss = mlp_out.mean()
        #         mlp.zero_grad()
        #         test_loss.backward()
        #
        #     torch.cuda.profiler.start()
        #     torch.cuda.synchronize()
        #     start_time = time()
        #     for _ in range(num_iters):
        #         ref_out = ref_mlp(ref_input)
        #         ref_loss = ref_out.mean()
        #         ref_mlp.zero_grad()
        #         ref_loss.backward()
        #     torch.cuda.synchronize()
        #     stop_time = time()
        # print(F"\nPytorch MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
        #
        #     torch.cuda.synchronize()
        #     start_time = time()
        #     for _ in range(num_iters):
        #         mlp_out, _ = mlp(test_input)
        #         test_loss = mlp_out.mean()
        #         mlp.zero_grad()
        #         test_loss.backward()
        #     torch.cuda.synchronize()
        #     stop_time = time()
        #     print(F"C++ MLP time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
        #     torch.cuda.profiler.stop()

    # unittest.main()