import random
import struct
import numpy as np
import os


def write_ark(ark, dic, scp=None, append=False):
    # Write ark
    mode = 'ab' if append else 'wb'
    pos_list = []
    with open(ark, mode) as fd:
        pos = fd.tell() if append else 0
        for key in dic:
            encode_key = (key + ' ').encode()
            fd.write(encode_key)
            pos += len(encode_key)
            pos_list.append(pos)
            data = dic[key]
            pos += write_array(fd, data)

    # Write scp
    if scp is not None:
        mode = 'a' if append else 'w'
        with open(scp, mode) as fd:
            for key, position in zip(dic, pos_list):
                fd.write(key + ' ' + ark + ':' + str(position) + os.linesep)


def write_ark_file(ark_file, scp_file, dic, scp=None):
    pos_lst, len_lst = [], []
    pos = ark_file.tell()
    for key in dic:
        encode_key = (key + ' ').encode()
        ark_file.write(encode_key)
        pos += len(encode_key)
        pos_lst.append(pos)
        data = dic[key]
        len_lst.append(len(data))
        pos += write_array(ark_file, data)

    ark = ark_file.name
    for key, ps, ln in zip(dic, pos_lst, len_lst):
        scp_file.write(key + ' ' + ark + ':' + str(ps) + ' ' + str(ln) + os.linesep)


def write_array(fd, array):
    size = 0
    assert isinstance(array, np.ndarray), type(array)
    fd.write(b'\0B')
    size += 2
    dt = array.dtype
    if dt == np.float32 or dt == np.float16:
        atype = b'FM ' if dt == np.float32 else b'HM '
        if len(array.shape) == 2:
            fd.write(atype)
            size += 3
            fd.write(b'\4')
            size += 1
            fd.write(struct.pack('<i', len(array)))  # Rows
            size += 4

            fd.write(b'\4')
            size += 1
            fd.write(struct.pack('<i', array.shape[1]))  # Cols
            size += 4
        fd.write(array.tobytes())
        size += array.nbytes
    else:
        raise ValueError('Unsupported array type: {}'.format(dt))
    return size
