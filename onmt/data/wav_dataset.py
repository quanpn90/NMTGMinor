import torch
import torchaudio as taudio
from functools import lru_cache
from .audio_utils import safe_readaudio, wav_to_fmel, safe_readaudio_from_cache
import numpy as np
import soundfile
import math
import torchaudio
import os


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path_list, cache_size=0,
                 wav_path_replace=None, num_mel_bin=0):
        """
        param wav_path_list: list of path to the audio
        param cache_size: size of the cache to load wav files (in case multiple wavefiles are used)
        param wav_path_replace:
        param num_mel_bin: if larger than zero, convert the audio to logmel features
        """
        self.wav_path_list = wav_path_list
        self._sizes = len(self.wav_path_list)
        self._dtype = torch.float32
        if cache_size > 0:
            self.cache = dict()
            self.usage = dict()
        else:
            self.cache = None
        self.cache_size = cache_size
        self.num_mel_bin = num_mel_bin

        if wav_path_replace is not None and wav_path_replace[0] != "None":
            found = False
            for old, new in zip(wav_path_replace[::2], wav_path_replace[1::2]):
                if old in self.wav_path_list[0][0]:
                    found = True

                self.wav_path_list = [(x[0].replace(old, new), *x[1:]) for x in self.wav_path_list]

            if not found:
                print(wav_path_replace, self.wav_path_list[0][0])
                print("WARNING: Could not replace wav path")

    def flush_cache(self):

        if self.cache is not None:
            for wav_path in self.cache:
                self.cache[wav_path].close()
                self.cache[wav_path] = None

        self.cache = dict()
        self.usage = dict()

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

        wav_path, start, end, sample_rate = wav_info

        # there are many utterances sharing the save wavfiles -> we can keep the same object in memory
        if self.cache is not None and wav_path.endswith("wav"):

            # take the object in cache if exists
            if wav_path in self.cache:
                file_ = self.cache[wav_path]
                self.usage[wav_path] = self.usage[wav_path] + 1
            else:
                # read the audio file
                # print(os.path.exists(wav_path), wav_path)

                try:
                    file_ = soundfile.SoundFile(wav_path, 'r')
                except RuntimeError as e:

                    print("Wavpath invalid:", wav_path, os.path.exists(wav_path))
                    raise e
                if len(self.cache) > self.cache_size:
                    # remove 1 file from cache based on lowest usage, maybe?
                    min_key = min(self.usage, key=self.usage.get)
                    if min_key != file_:
                        self.cache[min_key].close()
                        self.cache.pop(min_key, None)
                        self.usage.pop(min_key, None)

                # add the object to the cache
                self.cache[wav_path] = file_
                self.usage[wav_path] = 1

            data = safe_readaudio_from_cache(file_, wav_path, start=start, end=end, sample_rate=sample_rate)
        else:
            file_ = None
            data = safe_readaudio(wav_path, start=start, end=end, sample_rate=sample_rate)

        if self.num_mel_bin > 0:
            data = wav_to_fmel(data, num_mel_bin=self.num_mel_bin)

        return data