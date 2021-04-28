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

        if fused_dropout_add_cuda is not None and input.is_cuda and input.dtype == torch.float16:
            dropout_mask, output = fused_dropout_add_cuda.forward(is_training, input, residual, dropout_prob)
        else:
            if is_training:
                dropout_results, dropout_mask = torch._fused_dropout(input, p=(1. - dropout_prob))
            else:
                dropout_mask = null_tensor
                dropout_results = input
            dropout_results.add_(residual)
            output = dropout_results

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
