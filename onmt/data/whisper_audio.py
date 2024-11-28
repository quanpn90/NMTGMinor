
import torchaudio

# import from huggingface for ease
from transformers import AutoProcessor, WhisperFeatureExtractor
import math
import torch

class WhisperAudioProcessor:

    def __init__(self, model_id):

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.feature_extractor = self.processor.feature_extractor

    def __call__(self, audio_file, start_time=0, end_time=0, sampling_rate=16000):

        feature_extractor = self.feature_extractor

        offset = math.floor(sampling_rate * start_time)
        num_frames = -1 if end_time <= start_time else math.ceil(sampling_rate * (end_time - start_time))
        a, sr = torchaudio.load(audio_file, frame_offset=offset, num_frames=num_frames, normalize=True)

        # sum(0) just in case there are multiple channels
        # maybe its not a good idea?
        x = feature_extractor(a.sum(0), sampling_rate=sr)

        x = x['input_features'][0]
        x = torch.from_numpy(x)

        # squeeze(0) to remove the batch dimension
        # transpose(0, 1) to make the channel dimension last
        x = x.squeeze(0).transpose(0, 1).contiguous()

        return x

    # TODO: spec augment applied here
    def extract_feature(self, x, sampling_rate=16000):
        """
        Convert wav (x) to logmel features

        Args:
            x:
            sampling_rate:

        Returns:
        """

        x = self.feature_extractor(x, sampling_rate=sampling_rate)
        x = x['input_features'][0]
        x = torch.from_numpy(x)

        x = x.squeeze(0).transpose(0, 1).contiguous()

        return x

