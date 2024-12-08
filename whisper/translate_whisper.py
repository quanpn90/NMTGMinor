from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import librosa
import sys
import argparse

import logging
import warnings

# Suppress warnings and set logging level
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='translate_whisper.py')

parser.add_argument('-src', required=True,
                    help='Source file to decode (one line per sequence)')


model_card = model_id = "openai/whisper-large-v3-turbo"

# Load model and processor
processor = WhisperProcessor.from_pretrained(model_card)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = WhisperForConditionalGeneration.from_pretrained(model_card, torch_dtype=torch_dtype,
                                                        low_cpu_mem_usage=True, use_safetensors=True)


model.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

generate_kwargs = {
    "max_new_tokens": 444,
    "num_beams": 4,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
    "language": "english"
}

opt = parser.parse_args()

src_file = opt.src

c = 0

for line in open(src_file).readlines():
    c += 1

    audio_file = line.split()[1]

    decode_result = pipe(audio_file, generate_kwargs=generate_kwargs)

    transcript = decode_result["text"]

    print('PRED %d: %s' % (c, transcript))
    print()

    