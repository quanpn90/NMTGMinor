import torch
import torchaudio as taudio
from functools import lru_cache
# from onmt.utils import safe_readaudio
import numpy as np
import soundfile
import math


# this function reads wav file based on the timestamp in seconds
def safe_readaudio_from_cache(file_, wav_path, start=0.0, end=0.0, sample_rate=16000):

    offset = math.floor(sample_rate * start)
    num_frames = -1 if end <= start else math.ceil(sample_rate * (end - start))

    if file_ is not None:
        dtype = "float32"
        frames = file_._prepare_read(offset, None, num_frames)
        waveform = file_.read(frames, dtype, always_2d=True)
        sample_rate_ = file_.samplerate
    else:
        with soundfile.SoundFile(wav_path, "r") as file_:
            dtype = "float32"
            frames = file_._prepare_read(frame_offset, None, num_frames)
            waveform = file_.read(frames, dtype, always_2d=True)
            sample_rate_ = file_.samplerate

    tensor = torch.from_numpy(waveform)
    tensor = tensor[:, 0].unsqueeze(1)  # select the first channel?
    # tensor has size [length, num_channel] in which channel should be 1 for wav2vec

    return tensor


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path_list, cache=True, cache_size=2048):
        """
        :param scp_path_list: list of path to the ark matrices
        """
        self.wav_path_list = wav_path_list
        self._sizes = len(self.wav_path_list)
        self._dtype = torch.float32
        if cache:
            self.cache = dict()
            self.usage = dict()
        else:
            self.cache = None
        self.cache_size = cache_size

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
        if self.cache is not None:

            # take the object in cache if exists
            if wav_path in self.cache:
                file_ = self.cache[wav_path]
                self.usage[wav_path] = self.usage[wav_path] + 1
            else:
                # read the audio file
                file_ = soundfile.SoundFile(wav_path, 'r')
                if len(self.cache) > self.cache_size:
                    # remove 1 file from cache based on lowest usage, maybe?
                    min_key = max(self.usage, key=self.usage.get)
                    self.cache[min_key].close()
                    self.cache.pop(min_key, None)
                    self.usage.pop(min_key, None)

                # add the object to the cache
                self.cache[wav_path] = file_
                self.usage[wav_path] = 1
        else:
            file_ = None

        data = safe_readaudio_from_cache(file_, wav_path, start, end, sample_rate)

        if self.cache is not None:
            self.cache[wav_info] = data

        return data
