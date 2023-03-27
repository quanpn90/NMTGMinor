import torch
import torch.nn as nn


class LengthAdapter(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthAdapter, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, input_size * 2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(input_size * 2, input_size * 2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(input_size * 2, output_size, kernel_size=3, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: input [T x B x H]
            mask: [B x T]
        Returns:

        """

        # from [T x B x H] to [B x T x H] to [B x H x T]
        x = x.permute(1, 2, 0)
        # x = x.transpose(1, 2).contiguous()  # convert to [batch_size, input_size, seq_length]
        x = self.conv_layers(x)

        # from    [B x H x T] to [B x T x H] to [T x B x H]
        x = x.permute(2, 0, 1)

        return x