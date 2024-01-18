from torch import Tensor


def apply_padding_mask(
    seqs: Tensor, padding_mask, pad_value = 0
) -> Tensor:
    """Apply the specified padding mask to ``seqs``.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        the batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param padding_mask:
        The padding mask to apply. *Shape:* :math:`(N,S)`, where :math:`N` is
        the batch size and :math:`S` is the sequence length.
    :param pad_value:
        The value for padded positions.

    :returns:
        The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    if padding_mask is None:
        return seqs

    m = padding_mask

    for _ in range(seqs.ndim - m.ndim):
        m = m.unsqueeze(-1)

    # should we do in-place pad masking?
    return seqs.where(m, pad_value)
