import torch
import torch.nn as nn


class VariationalLayer(nn.Module):

    def __init__(self, inputSize, outputSize):
        super(VariationalLayer, self).__init__()

        print("Variational layer")

        self.inputSize = inputSize
        self.outputSize = outputSize

        self.meanLL= nn.Linear(self.inputSize, self.outputSize)
        self.stdLL = nn.Linear(self.inputSize, self.outputSize)

        self.meanAct = nn.Tanh()
        self.stdAct = nn.Softplus()


    def forward(self, input):

        mean = self.meanLL(input)
        mean = self.meanAct(mean)
        self.mean = mean
        if(self.training):
            std = self.stdLL(input)
            std = self.stdAct(std)
            random = torch.randn(std.size())
            if(std.is_cuda):
                random = random.cuda()
            self.std = std
            mean = mean + random * std
        return mean

