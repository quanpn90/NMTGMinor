import torch
import torch.nn.functional as F


try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError) as e:
    from .compat import custom_fwd, custom_bwd

try:
    import mask_softmax_dropout_cuda
except (ModuleNotFoundError, ImportError) as e:
    mask_softmax_dropout_cuda = None


class AttentionSoftmaxDropout(object):

    @staticmethod
    def forward(inputs, double_precision, dropout_prob, is_training, heads):
        """
        :param heads: 
        :param is_training: 
        :param dropout_prob: 
        :param inputs: 
        :param double_precision: 
        :return: 
        """

        len_k = inputs.size(-1)
        if mask_softmax_dropout_cuda and len_k <= 2048 and inputs.type() == 'torch.cuda.HalfTensor':

            dropout_mask, softmax_results, dropout_results = \
                mask_softmax_dropout_cuda.forward(is_training, heads, inputs, dropout_prob)

            if is_training:
                dropout_results = softmax_results

        else:
            dtype_ = torch.float64 if double_precision else torch.float32
            softmax_results = F.softmax(inputs, dim=-1, dtype=dtype_)

            # Dropout - is not executed for inference
            if is_training:
                dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=(1. - dropout_prob_t[0]))
            else:
                dropout_results = softmax_results
                dropout_mask = torch.tensor([])

        return dropout_mask, softmax_results, dropout_results

    @staticmethod
    def backward(grad_outputs, softmax_results, dropout_prob_t, heads_t, dropout_mask):
        len_key = softmax_results.size(-1)

        if mask_softmax_dropout_cuda is not None and grad_outputs.type() == 'torch.cuda.HalfTensor' \
                and len_key <= 2048:

            softmax_grads = mask_softmax_dropout_cuda.backward_recompute(heads_t[0], grad_outputs, softmax_results,
                                                                         dropout_mask, dropout_prob_t[0])

        else:
            dropout_grads = torch._masked_scale(grad_outputs, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))

            # be careful we overwrite into "softmax_results" memory here
            softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)

        return softmax_grads
