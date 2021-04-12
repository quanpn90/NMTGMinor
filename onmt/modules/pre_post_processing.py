import torch
import torch.nn as nn
from .layer_norm import LayerNorm, MultilingualLayerNorm
import onmt
from onmt.modules.dropout import VariationalDropout
from onmt.modules.bottle import Bottle


class PrePostProcessing(nn.Module):
    """Applies processing to tensors
    Args:
        d_model: dimension of model
        p:       dropout probabolity
        sequence of processing steps:
            n = normalization
            d = dropout
            a = adding previous input to output (residual)
    """

    def __init__(self, d_model, dropout_p, sequence='nda', variational=False, elementwise_affine=True,
                 multilingual=False, n_languages=1):
        super(PrePostProcessing, self).__init__()
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.multilingual = multilingual

        self.steps = list(sequence)

        if onmt.constants.residual_type == 'gated':
            # gated residual
            # initialize k with one
            self.k = nn.Parameter(torch.ones(1))

        if 'n' in self.steps:
            if not multilingual:
                ln = LayerNorm((self.d_model,), elementwise_affine=elementwise_affine)
                self.layer_norm = Bottle(ln)
            else:
                ln = MultilingualLayerNorm((self.d_model,), eps=1e-5, elementwise_affine=True, n_languages=n_languages)
                self.layer_norm = ln
        if 'd' in self.steps:
            if variational:
                self.dropout = VariationalDropout(self.dropout_p, batch_first=False)
            else:
                self.dropout = nn.Dropout(self.dropout_p)
        if 'z' in self.steps:
            # Rezero residual method
            self.g = nn.Parameter(torch.tensor(0.0))

    def forward(self, tensor, input_tensor=None, mask=None, factor=None):
        """
        :param tensor: input tensor [BxTxH] or [TxBxH (most likely)]
        :param input_tensor: previous tensor for residual
        :param mask: unused
        :param factor: tensor size 1, for multilingual
        :return:
        """

        output = tensor
        for step in self.steps:
            if step == 'n':
                # this cast is needed for O1 and FusedLayerNorm
                if self.multilingual:
                    output = self.layer_norm(output, factor)
                    output = output
                else:
                    output = self.layer_norm(output)
            if step == 'd':
                output = self.dropout(output)
            if step == 'a':
                if input_tensor is not None:
                    if onmt.constants.residual_type != 'gated':
                        output = output + input_tensor
                    else:
                        output = F.relu(self.k) * output + input_tensor
            if step == 'z':  # rezero-residual but scaling the output with initially small g
                output = output * self.g
                if input_tensor is not None:
                    output = output + input_tensor
        return output
