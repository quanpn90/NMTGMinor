class Generator(nn.Module):

    def __init__(self, hidden_size, output_size, fix_norm=False):

        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        self.fix_norm = fix_norm

        stdv = 1. / math.sqrt(self.linear.weight.size(1))

        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)

        self.linear.bias.data.zero_()

    def forward(self, output_dicts):
        """
        :param output_dicts: dictionary contains the outputs from the decoder
        :return: logits (the elements before softmax)
        """

        input = output_dicts['hidden']
        fix_norm = self.fix_norm
        target_mask = output_dicts['target_mask']

        if not fix_norm:
            logits = self.linear(input).float()
        else:
            normalized_weights = F.normalize(self.linear.weight, dim=-1)
            normalized_bias = self.linear.bias
            logits = F.linear(input, normalized_weights, normalized_bias)

        # softmax will be done at the loss function
        # output = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        return logits
