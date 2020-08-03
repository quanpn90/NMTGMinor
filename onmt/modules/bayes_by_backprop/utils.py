import torch


def flatten_list(tensors):

    flat = list()
    indices = list()
    shapes = list()

    s = 0
    for tensor in tensors:
        shapes.append(tensor.shape)
        flat_t = torch.flatten(tensor)
        size = flat_t.shape[0]

        flat.append(flat_t)
        indices.append((s, s+size))
        s += size
    flat = torch.cat(flat).view(-1)

    return flat, indices, shapes


def unflatten(flat, indices, shapes):

    params = [flat[s:e] for (s, e) in indices]
    for i, shape_p in enumerate(shapes):
        params[i] = params[i].view(*shape_p)

    return tuple(params)
