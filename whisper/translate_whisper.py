from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import librosa
import sys
import argparse
import torchaudio
import logging
import warnings

from memory_efficient_whisper import (MemoryEfficientWhisper,
                                      MemoryEfficientWhisperEncoderLayer,
                                      MemoryEfficientLayerNorm,
                                      create_whisper_model)

# Suppress warnings and set logging level
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='translate_whisper.py')

parser.add_argument('-model_path', required=True,
                    help="Path to the model checkpoint")
parser.add_argument('-src', required=True,
                    help='Source file to decode (one line per sequence)')
parser.add_argument('-batch_size', type=int, default=8,
                    help='Batch size during decoding ')
parser.add_argument('-beam_size', type=int, default=4,
                    help='Beam size during decoding')
parser.add_argument('-output_file', required=True,
                    help="Path to the output_file to be written")
parser.add_argument('-target_file', required=False,
                    help="Path to the reference file. If provided word error rate will be computed")

args = parser.parse_args()

model_card = args.model_path

tokenizer_card = "openai/whisper-large-v3-turbo"

if model_card == "none":
    model_card = tokenizer_card
# Load model and processor
processor = WhisperProcessor.from_pretrained(tokenizer_card)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = create_whisper_model(model_card, torch_dtype=torch_dtype,
                             attn_implementation='flash_attention_2')

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
    "max_new_tokens": 444, # limited by whisper
    "num_beams": args.beam_size,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,  # long-form whisper needs this
    "language": "english"
}

src_file = args.src
writer = open(args.output_file, 'w')

audio_list = list()

def process_audio_list(counter = 0):

    decode_results = pipe(audio_list, generate_kwargs=generate_kwargs, batch_size=len(audio_list))
    # print(decode_results)

    for decode_result in decode_results:
        transcript = decode_result["text"]
        counter = counter + 1
        print('PRED %d: %s' % (counter, transcript))
        print()
        writer.write(transcript)
        writer.write("\n")

def process_long_audio(audio_file, counter = 0):

    decode_result = pipe(audio_file, generate_kwargs=generate_kwargs)
    transcript = decode_result["text"]
    counter = counter + 1
    print('PRED %d: %s' % (counter, transcript))
    print()
    writer.write(transcript)
    writer.write("\n")

c = 0
for line in open(src_file).readlines():

    audio_file = line.split()[1]

    waveform, sample_rate = torchaudio.load(audio_file)
    duration = waveform.size(1) / sample_rate

    # if the duration is too long:
    if duration > 30.000:
        # process the current batch if not empty
        if len(audio_list) > 0:
            process_audio_list(c)
            c = c + len(audio_list)
            audio_list = list()

        # use long-form whisper to deal with the long audio file
        process_long_audio(audio_file, c)
        c = c + 1

    else:
        audio_list.append(audio_file)

        # TODO: group audio files in a list for batch decoding
        if len(audio_list) >= args.batch_size:
            # print(audio_list)
            process_audio_list(c)
            c = c + len(audio_list)
            audio_list = list()

# finish the last batch if not empty
if len(audio_list) > 0:
    process_audio_list(c)

writer.close()
    