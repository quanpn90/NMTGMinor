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
import soundfile
import math
import torch
import torchaudio

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
        elif Type == 'HM':
            dtype = endian + 'e'
            bytes_per_sample = 2
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


def safe_readaudio_from_cache(file_, start=0.0, end=0.0, sample_rate=16000):
    offset = math.floor(sample_rate * start)
    num_frames = -1 if end <= start else math.ceil(sample_rate * (end - start))

    dtype = "float32"
    frames = file_._prepare_read(offset, None, num_frames)
    waveform = file_.read(frames, dtype, always_2d=True)
    sample_rate_ = file_.samplerate
    tensor = torch.from_numpy(waveform)
    tensor = tensor[:, 0].unsqueeze(1)
    return tensor


class WavLoader(object):

    def __init__(self, cache_size=512):
        """
        :param scp_path_list: list of path to the ark matrices
        """
        if cache_size > 0:
            self.cache = dict()
            self.usage = dict()
        else:
            self.cache = None
        self.cache_size = cache_size

    def load_wav(self, wav_path, start, end, sample_rate=16000):

        # take the object in cache if exists
        if wav_path in self.cache:
            file_ = self.cache[wav_path]
            self.usage[wav_path] = self.usage[wav_path] + 1
        else:
            # read the audio file
            # print(os.path.exists(wav_path), wav_path)
            file_ = soundfile.SoundFile(wav_path, 'r')
            if len(self.cache) > self.cache_size:
                # remove 1 file from cache based on lowest usage, maybe?
                min_key = min(self.usage, key=self.usage.get)
                if min_key != wav_path:  # don't close the current file
                    self.cache[min_key].close()
                    self.cache.pop(min_key, None)
                    self.usage.pop(min_key, None)

            # add the object to the cache
            self.cache[wav_path] = file_
            self.usage[wav_path] = 1

        data = safe_readaudio_from_cache(file_, start, end, sample_rate)

        return data

    def close(self):

        for wav_path in self.cache:
            self.cache[wav_path].close()


# this function reads wav file based on the timestamp in seconds
def safe_readaudio(wav_path, start=0.0, end=0.0, sample_rate=16000):

    offset = math.floor(sample_rate * start)
    num_frames = -1 if end <= start else math.ceil(sample_rate * (end - start))

    # by default torchaudio normalizes the read tensor
    tensor, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=num_frames,
                                normalize=True, channels_first=False)
    tensor = tensor[:, 0].unsqueeze(1)

    # tensor has size [length, num_channel] in which channel should be 1 for wav2vec
    return tensor


def wav_to_fmel(wav: torch.Tensor, num_mel_bin=80, waveform_scale=2**15, standardized=True):
    """
    Args:
        wav:
        num_mel_bin:
        waveform_scale: in fairseq 2.0 it is used as 2**15
        standardized:

    Returns:

    """
    # extract log mel filterbank features
    # wav = wav.squeeze(1)
    # print(wav.size())
    # assert(wav.ndim == 1)
    wav = wav.squeeze(1).unsqueeze(0)
    feature_vector = torchaudio.compliance.kaldi.fbank(wav * waveform_scale,
                                            num_mel_bins=num_mel_bin)
    # normalize
    if standardized:
        std, mean = torch.std_mean(feature_vector, 0)

        feature_vector = feature_vector.subtract(mean).divide(std)

    return feature_vector

# this function reads wav file based on the timestamp in seconds
def safe_readaudio_from_cache(file_, wav_path, start=0.0, end=0.0, sample_rate=16000):
    offset = math.floor(sample_rate * start)
    num_frames = -1 if end <= start else math.ceil(sample_rate * (end - start))

    if file_ is not None:
        dtype = "float32"
        frames = file_._prepare_read(offset, None, num_frames)
        waveform = file_.read(frames, dtype, always_2d=True)
        sample_rate_ = file_.samplerate
        tensor = torch.from_numpy(waveform)
        tensor = tensor[:, 0].unsqueeze(1)
    else:

        tensor = tensor[:, 0].unsqueeze(1)

    # select the first channel?
    # tensor has size [length, num_channel] in which channel should be 1 for wav2vec

    return tensor

