import torch
import timeit

import sys
cuda = len(sys.argv) > 1 and sys.argv[1] == "--cuda"

setup = "inputs = torch.rand(5, 6, 7); mask = torch.rand(5, 6).ge(0.5)"
mask_select = "inputs.masked_select(mask.unsqueeze(-1))"
nonzero = "inputs.view(-1, 7).index_select(0, torch.nonzero(mask.view(-1)).squeeze(1))"

if cuda:
    setup += "; inputs = inputs.cuda(); mask = mask.cuda(); torch.cuda.synchronize()"
    mask_select += "; torch.cuda.synchronize()"
    nonzero += "; torch.cuda.synchronize()"

timerA = timeit.Timer(mask_select, setup, globals=globals())
timerB = timeit.Timer(nonzero, setup, globals=globals())

print("Mask select: {:.2f}ms".format(sum(timerA.repeat(10, 100)) / 10 * 1000))
print("Nonzero: {:.2f}ms".format(sum(timerB.repeat(10, 100)) / 10 * 1000))

setup = "inputs = torch.rand(5, 6); mask = torch.rand(5, 6).ge(0.5)"
mask_select = "inputs.masked_select(mask)"
nonzero = "inputs.view(-1).index_select(0, torch.nonzero(mask.view(-1)).squeeze(1))"

if cuda:
    setup += "; inputs = inputs.cuda(); mask = mask.cuda(); torch.cuda.synchronize()"
    mask_select += "; torch.cuda.synchronize()"
    nonzero += "; torch.cuda.synchronize()"

timerA = timeit.Timer(mask_select, setup, globals=globals())
timerB = timeit.Timer(nonzero, setup, globals=globals())

print("Mask select: {:.2f}ms".format(sum(timerA.repeat(10, 100)) / 10 * 1000))
print("Nonzero: {:.2f}ms".format(sum(timerB.repeat(10, 100)) / 10 * 1000))
