
def unflatten_state_dict(state_dict):
    res = {}
    for key, value in state_dict.items():
        key = key.split('.')
        current = res
        for part in key[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[key[-1]] = value
    return res


def flatten_state_dict(state_dict, prefix=''):
    res = {}
    for k, v in state_dict.items():
        key = prefix + k
        value = v
        if isinstance(value, dict):
            res.update(flatten_state_dict(value, key + '.'))
        else:
            res[key] = value
    return res
