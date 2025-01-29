import os
from datasets import load_dataset, ClassLabel, Features, Value, Dataset, Audio, concatenate_datasets, load_from_disk, \
    interleave_datasets
from pydub import AudioSegment
from tqdm import tqdm
import re
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    AutoProcessor, AutoTokenizer, SeamlessM4TForSpeechToText, EarlyStoppingCallback, SeamlessM4Tv2ForSpeechToText
import sys
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, TimeMask
import torchaudio.transforms as T
import copy
import numpy as np
from random import shuffle
from concurrent.futures import ThreadPoolExecutor
import torch

# from trainer_mem import MemSeq2SeqTrainer
from trainers.trainer_mem import MemSeq2SeqTrainer

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# TODO: change this
sys.path.append('/home/eugan/repos/yapay-net/src/hug/trainer/')


def normalize_text(utterance, language):
    arabic_filter = re.compile(r'[OUM]+/*|\u061F|\?|\!|\.')
    english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.')  #|\.
    cyrillic_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.')
    japanese_filter = re.compile(
        r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:|\.|\u3002|\u300C|\u300D|\uFF08|\uFF09|\uFF0C|\uFF1F|\uFF01|\uFF1A|\uFF1B')

    #english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\,|\"|\:')#|\.
    if language == "ar":
        return re.subn(arabic_filter, '', utterance)[0].lower()
    elif language == "en" or language == "de" or language == "es" or language == "tr":
        return re.subn(english_filter, '', utterance)[0].lower()
    elif language == "uk":
        return re.subn(cyrillic_filter, '', utterance)[0].lower()
    elif language == "zh" or language == "ja":
        return re.subn(japanese_filter, '', utterance)[0].lower()
    else:
        raise ValueError(f'Text normalization for {language} is not supported')


# CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞", "؟",
# "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "{", "}", "=", "`", "_", "+", "<",
# ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。", "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（",
# "）", "［", "］", "【", "】", "‥", "〽", "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−",
# "^", "ʻ", "ˆ","]","[","-", "#"]

CHARS_TO_IGNORE = [
    "¿", "¡", ";", "；", ":", '"', "%", "�", "ʿ", "·", "჻", "~", "՞", "؟", "।", "॥", "«", "»", "„", "“", "”",
    "「", "」", "‘", "’", "《", "》", "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›",
    "©", "®", "—", "→", "﹂", "﹁", "‧", "～", "﹏", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
    "『", "』", "〝", "〟", "⟨", "⟩", "〜", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ", "]", "[", "-", "#"
]

chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters(text):
    if chars_to_ignore_re is not None:
        return re.sub(chars_to_ignore_re, "", text).lower()
    else:
        return text.lower()


def file_exists(path):
    return os.path.exists(path)


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
    "def": "def",
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
            #print(UNK)
            return mapper["en"]
            #return mapper["<unk>"]

    tmp = list()
    for word in text.split():
        tmp.append(detect_language_word(word, mapper))
    return tmp


def load_asr_dataset(file_path, language):
    csw = True
    if type(language) is not list:
        language = [language]
        csw = False

    # Read the content of the STM filea
    with open(file_path, 'r', encoding='utf-8') as stm_file:
        lines = stm_file.readlines()

    # Process lines to create a list of dictionaries
    data = []
    skipped = 0
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) < 6: skipped += 1; continue
        try:
            uid, wavpath, start, end, duration, transcript = parts
        except Exception as e:
            uid, wavpath, start, end, duration = parts[:5]
            transcript = " ".join(parts[5:])

        if "/" not in wavpath:
            skipped += 1;
            continue
        # if not os.path.exists(wavpath): skipped +=1; continue

        duration = float(duration)  # ms
        if duration < 500 or duration > 20000: skipped += 1; continue

        transcript = remove_special_characters(transcript)
        if transcript.strip() == "":
            skipped += 1
            continue

        transcript = " ".join(transcript.split())

        language_str = ",".join(language)

        if csw:
            #print(line)
            language_ids = detect_language(transcript, mapper_orig)
            lang_mix_id = -100
        else:
            #print(ENES)
            lang_mix_id = mapper_orig[language[0]]
            language_ids = [mapper_orig[language[0]] for _ in transcript.split()]

        #if len(parts) >= 3:  # Assuming at least 5 columns in STM file
        #transcript = "<|startoftranscript|><|{}|><|transcribe|><|notimestamps|> {}<|endoftext|>".format(language[0],transcript)
        transcript = "<|startoftranscript|><|{}|><|transcribe|><|notimestamps|> {}<|endoftext|>".format(language[0],
                                                                                                        transcript)

        data.append({
            #'audio': audio.raw_data,
            #'audio_path': parts[0],
            'uid': uid,
            'audio': wavpath,
            'transcript': transcript,
            'start': start,
            'end': end,
            'duration': duration,
            'language': language_str,
            'language_ids': language_ids,
            'lang_mix_id': lang_mix_id,
        })
        #print(language_ids)

    # Create a Hugging Face Dataset
    dataset = Dataset.from_list(data)
    print(dataset)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    #print(dataset[0]["audio"])
    #print(type(dataset[0]["audio"]["array"]))
    #print(ASD)
    # for el in dataset:
    #     print(el)
    # print(AD)
    print("{}/{} skipped".format(skipped, len(lines)))
    return dataset


def find_index(x, ms, pseudo_csw_amount):
    check_range, counter = pseudo_csw_amount, 0
    while check_range < len(x) - 1:
        if x[check_range]["duration"] >= ms:
            break
        counter += 1
        check_range = pseudo_csw_amount * counter
    return min(check_range, len(x) - 1)


def do_csw_backup(train_list, pseudo_csw_amount, train=True):
    dummy_audio = None
    data_list = list()
    for x in train_list:
        x = x.sort("duration", reverse=False)
        if dummy_audio == None and x[0]["duration"] < 2000:
            dummy_path = x[0]["audio"]["path"]
        if train:
            ten_sec_idx = find_index(x, 10 * 1000, pseudo_csw_amount)
            if ten_sec_idx >= len(x) - 1 or int(pseudo_csw_amount) >= ten_sec_idx:
                tmp = x
            else:
                tmp_ten = x.train_test_split(test_size=int(ten_sec_idx), shuffle=False)["test"]
                tmp = tmp_ten.train_test_split(test_size=int(pseudo_csw_amount), shuffle=True)["test"]
            tmp = tmp.remove_columns(["audio"])
        else:
            tmp = x.train_test_split(test_size=int(pseudo_csw_amount), shuffle=False)
            tmp = tmp.remove_columns(["audio"])["test"]
        data_list.append(tmp)

    csw_data_list = list()
    number_of_elements = pseudo_csw_amount * 8
    used_idx = {}
    used_entry = False
    duration_20 = 0
    duration_30 = 0
    pbar = tqdm(total=number_of_elements)
    while number_of_elements > 0:
        if random.randint(0, 1) == 0:
            number_of_switches = 2
        else:
            number_of_switches = 3

        random_dataset = data_list[random.randint(0, len(data_list) - 1)]
        random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
        duration = random_entry["duration"]
        while duration > 10 * 1000:
            random_dataset = data_list[random.randint(0, len(data_list) - 1)]
            random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
            duration = random_entry["duration"]
        uid_list = [random_entry["uid"]]

        transcript = random_entry["transcript"].strip().split(" ", 1)[-1].replace("<|endoftext|>", "")

        start = random_entry["start"]
        end = random_entry["end"]

        language = ["<|{}|>".format(random_entry["language"])]
        number_of_elements -= 1
        pbar.update(1)

        for i in range(number_of_switches - 1):
            random_dataset = data_list[random.randint(0, len(data_list) - 1)]
            random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
            new_duration = random_entry["duration"]
            while (duration + new_duration > 15000 and i == 0) or (duration + new_duration > 20000 and i == 1):
                #print("{}+{}={}".format(duration, new_duration, duration+new_duration))
                random_dataset = data_list[random.randint(0, len(data_list) - 1)]
                random_entry = random_dataset[random.randint(0, len(random_dataset) - 1)]
                new_duration = random_entry["duration"]

            duration += new_duration

            uid_list.append(random_entry["uid"])
            new_transcript = "{} {}".format(transcript, random_entry["transcript"].strip().split(" ", 1)[-1].replace(
                "<|endoftext|>", ""))
            transcript = new_transcript
            language.append("<|{}|>".format(random_entry["language"]))
            number_of_elements -= 1
            pbar.update(1)
            print(transcript)

        csw_transcript = "<|startoftranscript|>{}<|transcribe|><|notimestamps|> {}<|endoftext|>".format(
            "".join(language), transcript)

        entry = {
            "uid": "--".join(uid_list),
            "audio": dummy_path,
            "transcript": csw_transcript,
            "start": start,
            "end": end,
            "duration": duration,
            "language": "--".join(language),
        }
        csw_data_list.append(entry)
        if duration > 20000: duration_20 += 1
        if duration > 30000: duration_30 += 1

    csw_dataset = Dataset.from_list(csw_data_list)
    print(csw_dataset)
    csw_dataset = csw_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("EXcced: 20: {} 30: {}".format(duration_20, duration_30))
    pbar.close()
    return csw_dataset

#
# def get_data(debug=False):
#     datasets = {}
#
#     # Iterate over all directories in the current working directory
#     for dir_name in os.listdir('../../../../repos/tmp'):
#         print(f"Loading {dir_name} ...")
#         # Check if it's a directory and ends with "..data"
#         if os.path.isdir(dir_name) and dir_name.endswith(".data"):
#             # Define the two files we need to load
#             all_tr_file = os.path.join(dir_name, "all-tr.stm")
#             small_dev_file = os.path.join(dir_name, "small-dev.stm")
#
#             # Check if the files exist in the directory
#             if os.path.isfile(all_tr_file) and os.path.isfile(small_dev_file):
#                 # Load the datasets using the hypothetical load_asr_dataset function
#                 all_tr_dataset = load_asr_dataset(all_tr_file, "def").shuffle(seed=42)
#                 small_dev_dataset = load_asr_dataset(small_dev_file, "def")
#
#                 # Use directory name and file name to create keys for the dictionary
#                 datasets[f"{dir_name}_all-tr"] = all_tr_dataset
#                 datasets[f"{dir_name}_small-dev"] = small_dev_dataset
#
#     dev_list = [v for k, v in datasets.items() if "_small-dev" in k]
#     datasets = {k: v for k, v in datasets.items() if "_small-dev" not in k}
#     #    dev_list = list()
#     #    for k in datasets.keys():
#     #        if "_small-dev" in k:
#     #            dev_list.append(datasets[k])
#     #            del datasets[k]
#
#     #  datasets = {}
#
#     #  print("Loading ARZEN...")
#     #  arzen_train="/project/asr_systems/LT2022/codeswitching/data/ARX-ENG/arzen/ArzEn_SpeechCorpus_1.0/train-clip-ntags.stm"
#     #  arzen_dev = "/project/asr_systems/LT2022/codeswitching/data/ARX-ENG/arzen/ArzEn_SpeechCorpus_1.0/dev-clip-ntags.stm"
#     #  shuffle_arzen_train_dataset = load_asr_dataset(arzen_train, "ar").shuffle(seed=42)
#     #  arzen_train_dataset = load_asr_dataset(arzen_train, "ar")
#     #  arzen_dev_dataset = load_asr_dataset(arzen_dev, "ar")
#
#     print("Loading SEAME...")
#     seame_train = "/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/seame/train_clip.stm"
#     seame_train_dataset = load_asr_dataset(seame_train, "def")
#     tmp = seame_train_dataset.train_test_split(test_size=3000)
#     seame_train_dataset = tmp["train"]
#     seame_dev_dataset = tmp["test"]
#     shuffle_seame_train_dataset = seame_train_dataset.shuffle(seed=42)
#
#     dev_list.append(seame_dev_dataset)
#     #  print("Loadin ASCEND ...")
#     #  ascend_train="/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/ASCEND/train_notags.stm"
#     #  ascend_dev="/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/ASCEND/dev_notags.stm"
#     #  ascend_train_dataset = load_asr_dataset(ascend_train, "def").shuffle(seed=42)
#     #  ascend_dev_dataset = load_asr_dataset(ascend_dev, "def")
#
#     print("Loading Fisher...")
#     fisher_train = "/project/asr_systems/LT2022/codeswitching/data/ESP-ENG/fisher/fisher_train_cs_train-transcript.stm"
#     fisher_dev = "/project/asr_systems/LT2022/codeswitching/data/ESP-ENG/fisher/fisher_train_cs_dev-transcript.stm"
#     shuffle_fisher_train_dataset = load_asr_dataset(fisher_train, "es").shuffle(seed=42)
#     fisher_train_dataset = load_asr_dataset(fisher_train, "es")
#     fisher_dev_dataset = load_asr_dataset(fisher_dev, "es")
#
#     fisher_mono_train = "/project/asr_systems/LT2022/codeswitching/data/ESP-ENG/fisher/fisher_train_mono-transcript.stm"
#     shuffle_fisher_mono_train_dataset = load_asr_dataset(fisher_mono_train, "es").shuffle(seed=42)
#
#     fisher_all_train_dataset = concatenate_datasets([shuffle_fisher_train_dataset, shuffle_fisher_mono_train_dataset])
#
#     #  print("Loading TALCS ...")
#     #  talcs_train = "/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/TALCS_corpus/train_set/train_time.stm"
#     #  talcs_dev = "/project/asr_systems/LT2022/codeswitching/data/CMN-ENG/TALCS_corpus/dev_set/dev_time.stm"
#     #  talcs_train_dataset = load_asr_dataset(talcs_train, "def").shuffle(seed=42)
#     #  talcs_dev_dataset = load_asr_dataset(talcs_dev, "def")
#
#     # csw_train_dataset = concatenate_datasets([shuffle_arzen_train_dataset, shuffle_seame_train_dataset, ascend_train_dataset, shuffle_fisher_train_dataset])
#     # csw_dev_dataset = concatenate_datasets([arzen_dev_dataset, seame_dev_dataset, ascend_dev_dataset, fisher_dev_dataset])
#
#     datasets["csw_train_dataset"] = fisher_all_train_dataset  #csw_train_dataset
#     datasets["shuffle_seame_train_dataset"] = shuffle_seame_train_dataset
#     #datasets["talcs_train_dataset"] = talcs_train_dataset
#
#     # dev_list.append(csw_dev_dataset)
#     #dev_list.append(talcs_dev_dataset)
#     dev_list.append(fisher_dev_dataset)
#     all_dev_dataset = concatenate_datasets(dev_list)
#
#     print(datasets)
#     print("===" * 20)
#     print(dev_list)
#
#     print("=====")
#     print(f"Dev data: {all_dev_dataset}")
#
#     return datasets, all_dev_dataset, fisher_dev_dataset


def get_data_seame(debug=False):
    datasets = {}

    dev_list = list()

    print("Loading SEAME...")
    seame_train = "seame_train_clip.stm"
    seame_train_dataset = load_asr_dataset(seame_train, "def")
    tmp = seame_train_dataset.train_test_split(test_size=3000)
    seame_train_dataset = tmp["train"]
    seame_dev_dataset = tmp["test"]
    shuffle_seame_train_dataset = seame_train_dataset.shuffle(seed=181195)

    datasets["shuffle_seame_train_dataset"] = shuffle_seame_train_dataset

    dev_list.append(seame_dev_dataset)

    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_fisher(debug=False):
    datasets = {}

    dev_list = []

    print("Loading Fisher...")
    fisher_train = "fisher_train_cs_train-transcript.stm"
    fisher_dev = "fisher_train_cs_dev-transcript.stm"
    shuffle_fisher_train_dataset = load_asr_dataset(fisher_train, "es").shuffle(seed=42)
    fisher_train_dataset = load_asr_dataset(fisher_train, "es")
    fisher_dev_dataset = load_asr_dataset(fisher_dev, "es")

    fisher_mono_train = "fisher_train_mono-transcript.stm"
    shuffle_fisher_mono_train_dataset = load_asr_dataset(fisher_mono_train, "es").shuffle(seed=42)

    fisher_all_train_dataset = concatenate_datasets([shuffle_fisher_train_dataset, shuffle_fisher_mono_train_dataset])

    datasets["csw_train_dataset"] = fisher_all_train_dataset  #csw_train_dataset

    dev_list.append(fisher_dev_dataset)
    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset


def get_data_arzen(debug=False):
    datasets = {}

    dev_list = []

    print("Loading ARZEN...")
    arzen_train="arzen-train-clip-ntags.stm"
    arzen_dev = "arzen-dev-clip-ntags.stm"
    shuffle_arzen_train_dataset = load_asr_dataset(arzen_train, "ar").shuffle(seed=42)
    arzen_train_dataset = load_asr_dataset(arzen_train, "ar")
    arzen_dev_dataset = load_asr_dataset(arzen_dev, "ar")

    datasets["csw_train_dataset"] = shuffle_arzen_train_dataset
    dev_list.append(arzen_dev_dataset)

    all_dev_dataset = concatenate_datasets(dev_list)

    print(datasets)
    print("===" * 20)
    print(dev_list)

    print("=====")
    print(f"Dev data: {all_dev_dataset}")

    return datasets, all_dev_dataset