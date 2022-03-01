import torch
import torch.nn.functional as F
from time import time

class ParameterRef(object):

    def __init__(self, weight_buf, offset, length, size):

        self.weight_buf = weight_buf
        self.offset = offset
        self.length = length
        self.size = size

    def __call__(self):

        return self.weight_buf[self.offset:self.offset+self.length].view(*self.size)


def find_weight(m, _weight_list):

    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.Parameter:
            weight = target_attr
            if weight.ndim == 2:
                _weight_list.append(weight)

    for n, ch in m.named_children():
        find_weight(ch, _weight_list)

    return _weight_list

def flatten_weight(m, _weight_buf, _offset):

    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.Parameter:
            weight = target_attr
            size = weight.size()
            numel = weight.numel()
            _weight_buf.data[offset:offset+numel].copy_(weight.data.view(-1))
            # print(_weight_buf[offset:offset+numel].view_as(weight))
            setattr(m, attr_str, None)
            del m._parameters[attr_str]
            setattr(m, attr_str, ParameterRef(_weight_buf, _offset, numel, size))
            _offset = _offset + numel
            del weight

    for n, ch in m.named_children():
        _offset = find_weight(ch, _weight_buf, _offset)

    return _offset

class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_hiddens=2):

        super(MLP, self).__init__()
        self.weight_buf = None
        self.input_weight = torch.nn.Parameter(torch.randn(hidden_size, input_size))

        self.hidden_weight = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.output_weight = torch.nn.Parameter(torch.randn(output_size, hidden_size))

    def set_buffer(self, _weight_buf):
        self.weight_buf = _weight_buf
        
    def forward(self, x):

        try:
            x = F.linear(x, self.input_weight, None)
            x = torch.relu(x)

            x = F.linear(x, self.hidden_weight, None)
            x = torch.relu(x)

            x = F.linear(x, self.output_weight, None)
        except TypeError as e:
            x = F.linear(x, self.input_weight(), None)
            x = torch.relu(x)

            x = F.linear(x, self.hidden_weight(), None)
            x = torch.relu(x)

            x = F.linear(x, self.output_weight(), None)

        return x

mlp = MLP(1024, 4096, 1024).cuda()

x = torch.rand(128, 1024).cuda()

torch.cuda.profiler.start()
torch.cuda.synchronize()
start_time = time()

for i in range(32):
    y = mlp(x)
    y.sum().backward()
    mlp.zero_grad()

torch.cuda.synchronize()
stop_time = time()
print(F"\nPytorch default MLP time {(stop_time - start_time) * 1000. / 32:.4f} ms")

weight_list = list()
find_weight(mlp, weight_list)

numels = sum([w.numel() for w in weight_list])

weight_buf = torch.nn.Parameter(torch.zeros(numels)).cuda()
offset = 0

with torch.no_grad():
    offset = flatten_weight(mlp, weight_buf, offset)

print(offset)

mlp.set_buffer(weight_buf)

torch.cuda.profiler.start()
torch.cuda.synchronize()
start_time = time()

for i in range(32):
    y = mlp(x)
    y.sum().backward()
    mlp.zero_grad()

torch.cuda.synchronize()
stop_time = time()
print(F"\nPytorch flattened MLP time {(stop_time - start_time) * 1000. / 32:.4f} ms")