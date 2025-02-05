import evaluate
import sacrebleu
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import os

from sklearn.metrics import classification_report
from datasets import load_dataset, ClassLabel, Features, Value, Dataset, Audio, concatenate_datasets
from pydub import AudioSegment
from tqdm import tqdm
import re
import transformers
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    AutoProcessor, AutoTokenizer, EarlyStoppingCallback, SeamlessM4Tv2ForSpeechToText
from torch.utils.data import DataLoader
from jiwer import wer, cer
import jiwer
import sacrebleu
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
import torch, gc
from torch import nn
import sys
import argparse

# sys.path.append('/home/eugan/repos/yapay-net/src/hug/net/')
# from whispermt import WhisperForConditionalGenerationMTOnlyNext as WhisperForConditionalGenerationMT
# from whispermt import WhisperForConditionalGeneration2 as WhisperForConditionalGenerationMT

from decode_utils import (DataCollatorSpeechSeq2SeqWithPadding,
                          load_asr_dataset,
                          compute_metrics,
                          remove_special_characters)

from memory_efficient_whisper import create_whisper_model

parser = argparse.ArgumentParser(description='translate_whisper.py')

parser.add_argument('-model_path', required=True, default="", type=str,
                    help="Path to the model checkpoint")
parser.add_argument('-lora_path', required=False, default="", type=str,
                    help="Path to the model checkpoint")
parser.add_argument('-test_stm', required=True, default="test_length.cl_lc.stm",
                    help='Source file to decode (one line per sequence)')
# parser.add_argument('-huggingface_dataset', required=False, default="",
#                     help="If src is none, using huggingface dataset")
parser.add_argument('-batch_size', type=int, default=8,
                    help='Batch size during decoding ')
parser.add_argument('-beam_size', type=int, default=4,
                    help='Beam size during decoding')
parser.add_argument('-output_file', required=False, default="",
                    help="Path to the output_file to be written")
parser.add_argument('-target_file', required=False, default="",
                    help="Path to the reference file. If provided word error rate will be computed")

parser.add_argument('-no_progress_bar', action='store_true',
                    help="Use spec augmentation")

args = parser.parse_args()

test_path = args.test_stm

# for arzen it should be ar en
test_dataset = load_asr_dataset(test_path, language=["ar", "en"])
print(test_dataset)

# feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
# tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
# processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)


# print(ASD)

upstream_model = "openai/whisper-large-v3-turbo"

processor = AutoProcessor.from_pretrained(upstream_model)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, text_processor=processor.tokenizer)

metric = evaluate.load("wer")

# seam_model_path = "../model/checkpoint-2400/"
model_path = args.model_path

# model = WhisperForConditionalGeneration.from_pretrained(model_path)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")
device = device if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = create_whisper_model(model_path, torch_dtype,
                             attn_implementation="flash_attention_2",
                             low_cpu_mem_usage=True,
                             device_map={"": device})

if args.lora_path is not None and len(args.lora_path) > 0:
    print("[INFO] Loading LORA weights from {}".format(args.lora_path))
    from peft import PeftModel
    lora_weights_path = args.lora_path

    # 2. Load the LoRA adapter weights onto the base model
    model = PeftModel.from_pretrained(model, lora_weights_path)

    # 3. Merge the LoRA weights into the base model's weights and unload the adapter
    model.merge_and_unload()

zh_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
en_id = processor.tokenizer.convert_tokens_to_ids("<|en|>")
es_id = processor.tokenizer.convert_tokens_to_ids("<|es|>")
de_id = processor.tokenizer.convert_tokens_to_ids("<|de|>")

transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
notimestamps_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
# print(f"zh: {zh_id} en: {en_id} es: {es_id} de: {de_id}")
forced_decoder_ids = list()
# print(model.generation_config.forced_decoder_ids[0], flush=True)
# print(ASD)
forced_decoder_ids.append(model.generation_config.forced_decoder_ids[0])
forced_decoder_ids.append([2, transcribe_id])
forced_decoder_ids.append([3, notimestamps_id])

# model = WhisperForConditionalGeneration.from_pretrained(_model)#, use_flash_attention_2=True)
# model = SeamlessM4Tv2ForSpeechToText.from_pretrained(_model)
# model = model.to_bettertransformer()
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# print("pad_token_id: {}".format(model.config.pad_token_id))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# device="cpu"
model.to(device)
if device == "cuda":  # and False:
    typ = torch.bfloat16
    model.to(torch.bfloat16)
else:
    typ = torch.float32

model.eval()

data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    collate_fn=data_collator,
    shuffle=False)


def to_cuda_recursive(data, device, dtype=torch.float32):
    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=dtype)
    elif isinstance(data, list):
        return [to_cuda_recursive(item, device, dtype=dtype) for item in data]
    elif isinstance(data, dict):
        return {key: to_cuda_recursive(value, device, dtype=dtype) for key, value in data.items()}
    elif isinstance(data, transformers.feature_extraction_utils.BatchFeature):
        return {key: to_cuda_recursive(value, device, dtype=dtype) for key, value in data.items()}
    else:
        return data


target_lst = list()
predictions_lst = list()
target_language_lst = list()

model.generation_config.forced_decoder_ids = forced_decoder_ids
model.generation_config.num_return_sequences = 1
model.generation_config.num_beams = args.beam_size
model.generation_config.no_repeat_ngram_size = 4
model.generation_config.max_new_tokens = 255

language_tokens = [t for t in processor.tokenizer.additional_special_tokens if len(t) == 6]
print(language_tokens)
language_ids = [processor.tokenizer.convert_tokens_to_ids(x) for x in language_tokens]
print(language_ids)

no_progress_bar = args.no_progress_bar

for idx, batch_ in tqdm(enumerate(data_loader), total=len(data_loader), disable=no_progress_bar):
    path_lst = batch_.pop("paths")
    target = batch_.pop("tgt_txt")
    uid = batch_.pop("uid")
    language = batch_.pop("lang_labels")
    # target_language_lst.extend(processor.batch_decode(language, skip_special_tokens=False))
    # print(batch_)
    for t in target:
        t = t.split(" ", 1)[-1].replace("<|endoftext|>", "")
        target_lst.append(t)
    # print(target_lst)
    # target_lst.extend(target)
    batch = to_cuda_recursive(batch_, device, typ)

    output_tokens = model.generate(input_features=batch["input_features"],
                                   generation_config=model.generation_config)
    # forced_decoder_ids=forced_decoder_ids, num_return_sequences=1, num_beams=10, no_repeat_ngram_size=4,max_new_tokens=255)#, task="transcribe")
    # print(output_tokens)
    pred_transcript = processor.batch_decode(output_tokens, skip_special_tokens=True)
    # print(uid)
    predictions_lst.extend(pred_transcript)
    if idx < 3:
        print(output_tokens)
        print(f"target: {target_lst}")
        # print(f"target_language_lst: {target_language_lst}")
        print(f"pred_language: {predictions_lst}")
    else:
        continue
    # break
    # print(ADSW)

clean_predictions = list()
for text in predictions_lst:
    text = remove_special_characters(text)
    clean_predictions.append(text)

with open("hypos.txt", "w") as f:
    for line in predictions_lst:
        f.write("{}\n".format(line.strip()))

with open("target.txt", "w") as f:
    for line in target_lst:
        f.write("{}\n".format(line.strip()))

with open("hypos.norm.txt", "w") as f:
    for line in clean_predictions:
        f.write("{}\n".format(line.strip()))

wer_error = wer(target_lst, clean_predictions)
cer_error = cer(target_lst, clean_predictions)

print("WER: {}".format(wer_error))
print("CER: {}".format(cer_error))

out = jiwer.process_words(target_lst,
                          clean_predictions)  # , reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)
with open("w.eval.txt", "w") as f:
    f.write(jiwer.visualize_alignment(out))

out = jiwer.process_characters(target_lst,
                               clean_predictions)  # , reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)
with open("c.eval.txt", "w") as f:
    f.write(jiwer.visualize_alignment(out))
