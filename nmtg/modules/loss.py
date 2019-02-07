import torch
from torch import nn, Tensor


class NMTLoss(nn.Module):
    def __init__(self, output_size, padding_idx, label_smoothing=0.0):
        super().__init__()
        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing
        weight = torch.ones(output_size)
        weight[padding_idx] = 0
        self.loss = nn.NLLLoss(weight, reduction='sum')

    def forward(self, lprobs: Tensor, targets: Tensor):
        gtruth = targets.view(-1)  # (batch * time,)
        lprobs = lprobs.view(-1, lprobs.size(-1))  # (batch * time, vocab_size)

        if self.label_smoothing > 0:  # label smoothing
            non_pad_mask = gtruth.ne(self.padding_idx)
            nll_loss = -lprobs.gather(1, gtruth.unsqueeze(1))[non_pad_mask]
            nll_loss = nll_loss.sum()
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.label_smoothing / (self.output_size - 1)
            loss = (1 - self.label_smoothing) * nll_loss + eps_i * smooth_loss
            loss_data = nll_loss.item()
        else:
            loss = self.loss(lprobs.float(), gtruth)
            loss_data = loss.item()

        return loss, loss_data
