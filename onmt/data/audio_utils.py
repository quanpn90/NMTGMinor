import numpy as np
from contextlib import contextmanager
import io
from io import TextIOBase
import os
import subprocess
import sys
import warnings
from functools import partial
from io import BytesIO
from io import StringIO
import re
import struct
import sys
import warnings

from .kaldiio.compression_header import GlobalHeader
from .kaldiio.compression_header import PerColHeader
from .kaldiio.utils import default_encoding
from .kaldiio.utils import LazyLoader
from .kaldiio.utils import MultiFileDescriptor
from .kaldiio.utils import open_like_kaldi
from .kaldiio.utils import open_or_fd
from .kaldiio.utils import seekable
from .kaldiio.wavio import read_wav
from .kaldiio.wavio import write_wav

PY3 = sys.version_info[0] == 3


if PY3:
    from collections.abc import Mapping
    binary_type = bytes
    string_types = str,
else:
    from collections import Mapping
    binary_type = str
    string_types = basestring,  # noqa: F821


# load scp function
# audio downsampling function
def _parse_arkpath(ark_name):
    """Parse arkpath
    Args:
        ark_name (str):
    Returns:
        Tuple[str, int, Optional[Tuple[slice, ...]]]
    Examples:
        >>> _parse_arkpath('a.ark')
        'a.ark', None, None
        >>> _parse_arkpath('a.ark:12')
        'a.ark', 12, None
        >>> _parse_arkpath('a.ark:12[3:4]')
        'a.ark', 12, (slice(3, 4, None),)
        >>> _parse_arkpath('cat "fo:o.ark" |')
        'cat "fo:o.ark" |', None, None
    """

    if ark_name.rstrip()[-1] == '|' or ark_name.rstrip()[0] == '|':
        # Something like: "| cat foo" or "cat bar|" shouldn't be parsed
        return ark_name, None, None

    slices = None
    if ':' in ark_name:
        fname, offset = ark_name.split(':', 1)
        if '[' in offset and ']' in offset:
            offset, Range = offset.split('[')
            # Range = [3:6,  10:30]
            Range = Range.replace(']', '').strip()
            slices = _convert_to_slice(Range)
        offset = int(offset)
    else:
        fname = ark_name
        offset = None
    return fname, offset, slices


def read_int32vector(fd, endian='<', return_size=False):
    assert fd.read(2) == b'\0B'
    assert fd.read(1) == b'\4'
    length = struct.unpack(endian + 'i', fd.read(4))[0]
    array = np.empty(length, dtype=np.int32)
    for i in range(length):
        assert fd.read(1) == b'\4'
        array[i] = struct.unpack(endian + 'i', fd.read(4))[0]
    if return_size:
        return array, (length + 1) * 5 + 2
    else:
        return array


def read_matrix_or_vector(fd, endian='<', return_size=False):
    """Call from load_kaldi_file
    Args:
        fd (file):
        endian (str):
        return_size (bool):
    """
    size = 0
    assert fd.read(2) == b'\0B'
    size += 2

    Type = str(read_token(fd))
    size += len(Type) + 1

    # CompressedMatrix
    if 'CM' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size
        per_col_header = PerColHeader.read(fd, global_header)
        size += per_col_header.size

        # Read data
        buf = fd.read(global_header.rows * global_header.cols)
        size += global_header.rows * global_header.cols
        array = np.frombuffer(buf, dtype=np.dtype(endian + 'u1'))
        array = array.reshape((global_header.cols, global_header.rows))

        # Decompress
        array = per_col_header.char_to_float(array)
        array = array.T

    elif 'CM2' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size

        # Read matrix
        buf = fd.read(2 * global_header.rows * global_header.cols)
        array = np.frombuffer(buf, dtype=np.dtype(endian + 'u2'))
        array = array.reshape((global_header.rows, global_header.cols))

        # Decompress
        array = global_header.uint_to_float(array)

    elif 'CM3' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size

        # Read matrix
        buf = fd.read(global_header.rows * global_header.cols)
        array = np.frombuffer(buf, dtype=np.dtype(endian + 'u1'))
        array = array.reshape((global_header.rows, global_header.cols))

        # Decompress
        array = global_header.uint_to_float(array)

    else:
        if Type == 'FM' or Type == 'FV':
            dtype = endian + 'f'
            bytes_per_sample = 4
        elif Type == 'DM' or Type == 'DV':
            dtype = endian + 'd'
            bytes_per_sample = 8
        else:
            raise ValueError(
                'Unexpected format: "{}". Now FM, FV, DM, DV, '
                'CM, CM2, CM3 are supported.'.format(Type))

        assert fd.read(1) == b'\4'
        size += 1
        rows = struct.unpack(endian + 'i', fd.read(4))[0]
        size += 4
        dim = rows
        if 'M' in Type:  # As matrix
            assert fd.read(1) == b'\4'
            size += 1
            cols = struct.unpack(endian + 'i', fd.read(4))[0]
            size += 4
            dim = rows * cols

        buf = fd.read(dim * bytes_per_sample)
        size += dim * bytes_per_sample
        array = np.frombuffer(buf, dtype=np.dtype(dtype))

        if 'M' in Type:  # As matrix
            array = np.reshape(array, (rows, cols))

    if return_size:
        return array, size
    else:
        return array


def read_ascii_mat(fd, return_size=False):
    """Call from load_kaldi_file
    Args:
        fd (file): binary mode
        return_size (bool):
    """
    string = []
    size = 0

    # Find '[' char
    while True:
        b = fd.read(1)
        try:
            char = b.decode(encoding=default_encoding)
        except UnicodeDecodeError:
            raise ValueError('File format is wrong?')
        size += 1
        if char == ' ' or char == '\n':
            continue
        elif char == '[':
            hasparent = True
            break
        else:
            string.append(char)
            hasparent = False
            break

    # Read data
    ndmin = 1
    while True:
        char = fd.read(1).decode(encoding=default_encoding)
        size += 1
        if hasparent:
            if char == ']':
                char = fd.read(1).decode(encoding=default_encoding)
                size += 1
                assert char == '\n' or char == ''
                break
            elif char == '\n':
                ndmin = 2
            elif char == '':
                raise ValueError(
                    'There are no corresponding bracket \']\' with \'[\'')
        else:
            if char == '\n' or char == '':
                break
        string.append(char)
    string = ''.join(string)
    assert len(string) != 0

    # Examine dtype
    match = re.match(r' *([^ \n]+) *', string)
    if match is None:
        dtype = np.float32
    else:
        ma = match.group(0)
        # If first element is integer, deal as interger array
        try:
            float(ma)
        except ValueError:
            raise RuntimeError(
                ma + 'is not a digit\nFile format is wrong?')
        if '.' in ma:
            dtype = np.float32
        else:
            dtype = np.int32
    array = np.loadtxt(StringIO(string), dtype=dtype, ndmin=ndmin)
    if return_size:
        return array, size
    else:
        return array


def read_token(fd):
    """Read token
    Args:
        fd (file):
    """
    token = []
    # Keep the loop until finding ' ' or end of char
    while True:
        c = fd.read(1)
        if c == b' ' or c == b'':
            break
        token.append(c)
    if len(token) == 0:  # End of file
        return None
    decoded = b''.join(token).decode(encoding=default_encoding)
    return decoded


def read_kaldi(fd, endian='<', return_size=False):
    """Load kaldi
    Args:
        fd (file): Binary mode file object. Cannot input string
        endian (str):
        return_size (bool):
    """
    assert endian in ('<', '>'), endian
    binary_flag = fd.read(4)
    assert isinstance(binary_flag, binary_type), type(binary_flag)

    if seekable(fd):
        fd.seek(-4, 1)
    else:
        fd = MultiFileDescriptor(BytesIO(binary_flag), fd)

    if binary_flag[:4] == b'RIFF':
        # array: Tuple[int, np.ndarray]
        array, size = read_wav(fd, return_size=True)

    # Load as binary
    elif binary_flag[:2] == b'\0B':
        if binary_flag[2:3] == b'\4':  # This is int32Vector
            array, size = read_int32vector(fd, endian, return_size=True)
        else:
            array, size = read_matrix_or_vector(fd, endian, return_size=True)
    # Load as ascii
    else:
        array, size = read_ascii_mat(fd, return_size=True)
    if return_size:
        return array, size
    else:
        return array


class ArkLoader(object):

    def __init__(self, fastest=True):

        self.current_ark = None
        self.reader = None
        self.readers = dict()
        self.fastest = fastest

    def load_mat(self, ark_name, endian='<', as_bytes=False):
        assert endian in ('<', '>'), endian
        ark, offset, slices = _parse_arkpath(ark_name)

        if not self.fastest:
            if self.current_ark != ark:
                if self.reader is not None:
                    self.reader.close()
                self.reader = open_like_kaldi(ark, 'rb')
                self.current_ark = ark

            return self.read_mat(self.reader, offset, slices, endian=endian, as_bytes=as_bytes)

        else:
            if ark not in self.readers:
                self.readers[ark] = open_like_kaldi(ark, 'rb')

            fd = self.readers[ark]
            return self.read_mat(fd, offset, slices, endian=endian, as_bytes=as_bytes)

    def read_mat(self, fd, offset, slices, endian='<', as_bytes=False):

        if offset is not None:
            fd.seek(offset)
        if not as_bytes:
            array = read_kaldi(fd, endian)
        else:
            array = fd.read()

        if slices is not None:
            if isinstance(array, (tuple, list)):
                array = (array[0], array[1][slices])
            else:
                array = array[slices]

        return array

    def close(self):

        if self.reader is not None:
            self.reader.close()

        for k in self.readers:
            self.readers[k].close()
