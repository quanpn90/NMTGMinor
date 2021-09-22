import torch
import torchaudio as taudio
from functools import lru_cache
from onmt.utils import safe_readaudio
import numpy as np


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path_list, cache=False):
        """
        :param scp_path_list: list of path to the ark matrices
        """
        self.wav_path_list = wav_path_list
        self._sizes = len(self.wav_path_list)
        self._dtype = torch.float32
        if cache:
            self.cache = dict()
        else:
            self.cache = None

    @property
    def dtype(self):
        # I'm not sure when this function is called
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    def __len__(self):
        return self._sizes

    def __getitem__(self, i):
        wav_info = self.wav_path_list[i]
        # it should be a tuple (wav_file, start, end)

        if self.cache is not None and wav_info in self.cache:
            return self.cache[wav_info]

        wavpath, start, end, sample_rate = wav_info
        data = safe_readaudio(wavpath, start, end, sample_rate)

        # mean normalization
        # x = data.numpy()
        # x = data
        # x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        # data = torch.from_numpy(x).unsqueeze(-1)

        if self.cache is not None:
            self.cache[wav_info] = data

        return data
