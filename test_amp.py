import torch
from apex import amp
from apex.normalization.fused_layer_norm import FusedLayerNorm
torch.cuda.set_device(1)

class NeuralNet(torch.nn.Module):

    def __init__(self, d_in, d_out):
        self.d_in = d_in
        self.d_out = d_out
        super().__init__()

        self.norm = torch.nn.LayerNorm(d_in)
        self.norm2 = FusedLayerNorm(d_out)
        # self.norm2 = torch.nn.LayerNorm(d_out)
        self.linear = torch.nn.Linear(d_in, d_out)
        self.linear2 = torch.nn.Linear(d_out, d_out)

    def forward(self, input):

        input = self.norm(input)
        print(input.type())
        output = self.linear(input)
        print(output.type())
        output = torch.relu(output)
        print(output.type())
        output = self.norm2(output)
        output = self.linear2(output)
        print(output.type())
        output = torch.nn.functional.log_softmax(output)
        print("end")
        return output

model = NeuralNet(500, 1000)
model = model.cuda()
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

for i in range(1000):
    x = torch.rand(128, 500).cuda()
    o = model(x).float()
    y = torch.randint(low=0, high=999, size=(128, )).cuda()
    loss = loss_function(o, y)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    optimizer.step()
    optimizer.zero_grad()
