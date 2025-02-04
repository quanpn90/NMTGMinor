import re
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
from dataclasses import dataclass
from typing import Any, Dict, List, Union


def detect_language(text, mapper):
    def detect_language_word(word, mapper):
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        mandarin_pattern = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]')
        latin_pattern = re.compile(r'[a-zA-Z]')
        number_pattern = re.compile(r'^-?\d+(\.\d+)?$')

        if number_pattern.match(word):
            return mapper["en"]

        contains_arabic = arabic_pattern.search(word)
        contains_latin = latin_pattern.search(word)
        contain_mandarin = mandarin_pattern.search(word)

        if contain_mandarin and contains_latin and contains_arabic:
            print("WTF all 3")
            print(ASD)
        elif contains_arabic and contains_latin:
            print(ASD)
            return mapper['mix']
        elif contain_mandarin:
            return mapper["zh"]
        elif contains_arabic:
            print(AR)
            return mapper['ar']
        elif contains_latin:
            return mapper['en']
        else:
            print(f"word: {word}")
            # print(UNK)
            return mapper["en"]
            # return mapper["<unk>"]

    tmp = list()
    for word in text.split():
        tmp.append(detect_language_word(word, mapper))
    return tmp


def normalize_text(utterance, language):
    arabic_filter = re.compile(r'[OUM]+/*|\u061F|\?|\!|\.')
    english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.')  # |\.
    cyrillic_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.')
    japanese_filter = re.compile(
        r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.|\u3002|\u300C|\u300D|\uFF08|\uFF09|\uFF0C|\uFF1F|\uFF01|\uFF1A|\uFF1B')

    # english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:')#|\.

    if language == "ar":
        return re.subn(arabic_filter, '', utterance)[0]
    elif language == "en" or language == "de" or language == "es" or language == "tr":
        return re.subn(english_filter, '', utterance)[0].lower()
    elif language == "uk":
        return re.subn(cyrillic_filter, '', utterance)[0].lower()
    elif language == "zh" or language == "ja":
        return re.subn(japanese_filter, '', utterance)[0]
    else:
        raise ValueError(f'Text normalization for {language} is not supported')


CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ", "]",
                   "[", "-", "#"]

chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters(text):
    if chars_to_ignore_re is not None:
        return re.sub(chars_to_ignore_re, "", text).lower()
    else:
        return text.lower()


mapper_orig = {
    "de": "<|de|>",
    "en": "<|en|>",
    "es": "<|es|>",
    "ar": "<|ar|>",
    "ua": "<|ua|>",
    "ja": "<|ja|>",
    "zh": "<|zh|>",
    "tr": "<|tr|>",
    "mix": "mix",
    "<unk>": "<unk>",
    "<eos>": "<eos>",
}

mapper_new = {
    "de": 2,
    "en": 0,
    "es": 3,
    "ar": 4,
    "ua": 5,
    "ja": 6,
    "zh": 1,
    "tr": 8,
    "mix": 0,
    "<unk>": 10,
    "<eos>": 11,
}


# Define a filter function
def filter_function(example):
    # Replace "audio" with the actual key in your dataset
    array_length = len(example["audio"]["array"])

    # Set your desired length range
    min_length = 8000
    max_length = 20 * 16000

    # Return True if the length is within the desired range
    return min_length <= array_length <= max_length


def load_asr_dataset(file_path, language):
    # Read the content of the STM file
    with open(file_path, 'r', encoding='utf-8') as stm_file:
        lines = stm_file.readlines()

    # Process lines to create a list of dictionaries
    data = []
    skipped = 0
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) < 3:
            # print("Skipping line" + line)
            skipped += 1
            continue
        # transcript, _, wav_path = parts
        if len(parts) < 6:
            continue
        uid, wav_path, start, end, duration, transcript = parts

        # if "/" not in wav_path:
        #     skipped += 1; continue

        # language_ids = detect_language(transcript, mapper_orig)
        language_ids = [mapper_orig[language[0]] for _ in transcript.split()]

        transcript = remove_special_characters(transcript)
        # if transcript.strip() == "":
        #     skipped += 1; continue
        transcript = " ".join(transcript.split())

        transcript = "<|startoftranscript|><|transcribe|><|notimestamps|> {}<|endoftext|>".format(transcript)

        if not wav_path.lower().endswith(".wav"): wav_path = wav_path + ".wav"

        # if len(parts) >= 3:  # Assuming at least 5 columns in STM file
        data.append({
            # 'audio': audio.raw_data,
            # 'audio_path': parts[0],
            "uid": uid,
            'audio': wav_path,
            'transcript': transcript,
            'start': start,
            'end': end,
            'language_ids': language_ids,
        })

        # Create a Hugging Face Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("{}/{} skipped".format(skipped, len(lines)))
    return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    text_processor: Any

    def get_language_for_token(self, words_batch, language_label_lst, decoder_input_ids):
        bpe_languages_batch = list()
        for idx, words in enumerate(words_batch):
            bpe_languages = ["<|transcribe|>", "<|notimestamps|>"]
            # bpe_languages = []
            words = words.split(" ", 1)[-1].replace("<|endoftext|>", "")

            languages = language_label_lst[idx]  # self.detect_language(words)
            len_ids = len(decoder_input_ids[idx])
            # print("===")
            toks = list()
            for word, language in zip(words.split(), languages):
                tok = self.text_processor(" " + word, return_tensors="pt", padding=False, add_special_tokens=False)
                #   print("word: {}".format(tok.input_ids))
                #    toks.extend(tok.input_ids)
                bpe_languages.extend([language] * tok.input_ids.shape[1])
            bpe_languages.append(bpe_languages[-1])
            # if len_ids != len(bpe_languages):
            #     print(words_batch)
            #     print(language_label_lst)
            #     print(bpe_languages)
            #     print(toks)
            #     print(decoder_input_ids)
            #     input("Press Enter to continue...")
            # print(bpe_languages)
            bpe_languages_batch.append("".join(bpe_languages))
            # bpe_languages_batch.append(torch.tensor(bpe_languages))

        padded_tensor = self.text_processor(bpe_languages_batch, return_tensors="pt", add_special_tokens=False,
                                            padding=True)
        # print("words_batch {} ".format(words_batch))
        # print("decoder_input_ids:  {} shape {}".format(decoder_input_ids, decoder_input_ids.shape))
        # padded_tensor = pad_sequence(bpe_languages_batch, batch_first=True, padding_value=-100)
        # print("bpe_languages_batch: {} ".format(bpe_languages_batch))
        # print(padded_tensor.input_ids, flush=True);
        # print(type(padded_tensor))
        return padded_tensor  # .input_ids

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        sr = 16000
        audio_lst = list()
        label_lst = list()
        path_lst = list()
        uid_lst = list()
        language_ids_lst = list()

        for el in features:
            audio = el["audio"]["array"]
            start = int(float(el["start"]))
            end = int(float(el["end"]))
            if len(audio) < 1:
                print("DDDDD")
                print(audio.shape)
                print(el["audio"], flush=True)
                continue
            if start >= 0 and end > 0:
                audio = audio[start * 16:end * 16]
            #  start = int(int(float(el["start"])*1000)*16)
            #  end = int(float(el["end"])*sr)
            #  audio = audio[start : end ]
            audio_lst.append(audio)
            path_lst.append(el["audio"]["path"])
            label_lst.append(el["transcript"])
            uid_lst.append(el["uid"])
            language_ids_lst.append(el["language_ids"])

        # batch = self.processor.feature_extractor(audio_lst, sampling_rate=sr, return_tensors="pt", padding=False)
        batch = self.processor.feature_extractor(audio_lst, sampling_rate=sr, return_tensors="pt")
        # print(batch)
        # print("==")
        # print(ADS)
        labels_batch = self.text_processor(label_lst, return_tensors="pt", add_special_tokens=False, padding=True)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        lang_labels_batch = self.get_language_for_token(label_lst, language_ids_lst, labels_batch.input_ids[:, 1:])
        lang_labels = lang_labels_batch["input_ids"].masked_fill(lang_labels_batch.attention_mask.ne(1), -100)
        # print(lang_labels, flush=True)
        lang_labels[:, :2] = -100

        # batch["labels"] = labels
        batch["paths"] = path_lst
        batch["tgt_txt"] = label_lst
        batch["decoder_input_ids"] = labels_batch.input_ids[:, :-1]
        batch["decoder_attention_mask"] = labels_batch.attention_mask[:, :-1]
        batch["lang_decoder_input_ids"] = lang_labels_batch.input_ids
        batch["lang_labels"] = lang_labels

        batch["uid"] = uid_lst
        # print(batch)
        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, tgt_lang="arb", skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, tgt_lang="arb",skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
   # bleu = sacrebleu.corpus_bleu(pred_str, [label_str]).score
    #return {"bleu": bleu}
    return {"wer": wer}