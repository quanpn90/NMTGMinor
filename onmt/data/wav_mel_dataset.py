# Copied from https://github.com/openai/whisper/blob/main/whisper/audio.py


import os
from functools import lru_cache
from typing import Optional, Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import soundfile

from .wav_dataset import safe_readaudio_from_cache, safe_readaudio


def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Parameters
    ----------
    file: str
        The audio file to open
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
            os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
        audio: Union[str, np.ndarray, torch.Tensor],
        n_mels: int = N_MELS,
        padding: int = 0,
        device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of
    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz
    n_mels: int
        The number of Mel-frequency filters, only 80 is supported
    padding: int
        Number of zero samples to pad to the right
    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT
    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WavMelDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path_list, cache_size=0):
        """
        :param scp_path_list: list of path to the ark matrices
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
        if self.cache is not None:

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

            data = safe_readaudio_from_cache(file_, wav_path, start, end, sample_rate)
        else:
            file_ = None
            data = safe_readaudio(wav_path, start, end, sample_rate)

        # convert waveforms -> logmel
        # summing on the 2nd dimension to "emulate" single channel
        data = log_mel_spectrogram(data.sum(dim=1))

        return data
