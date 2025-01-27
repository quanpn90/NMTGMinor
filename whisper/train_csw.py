import os
from datasets import load_dataset, ClassLabel, Features, Value, Dataset, Audio, concatenate_datasets, \
    interleave_datasets
from pydub import AudioSegment
from tqdm import tqdm
import re
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    AutoProcessor, AutoTokenizer, SeamlessM4TForSpeechToText, EarlyStoppingCallback, SeamlessM4Tv2ForSpeechToText
from transformers import Seq2SeqTrainingArguments, SpeechEncoderDecoderConfig, AutoFeatureExtractor, WhisperConfig
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
import sys

sys.path.append('/home/eugan/repos/yapay-net/src/hug/trainer/')
# from trainer_mem import MemSeq2SeqTrainer
# from trainer_shuffle import MemSeq2SeqTrainer
from trainers.trainer_shuffle import MemSeq2SeqTrainer

# from wavlm.modeling_wavlm import
# from whispermt import WhisperForConditionalGenerationMTOnlyNext as WhisperForConditionalGenerationMT
# from whispermt import WhisperForConditionalGeneration2 as WhisperForConditionalGenerationMT
from whispermt import WhisperConfig2
from wavlm.configuration_wavlm import WavLMConfig
from mbart.configuration_mbart import MBartConfig
from wavlm.modeling_wavlm import WavLMModel, WavLMForSequenceClassification
from mbart.modeling_mbart import MBartForCausalLM, MultiLangMBartForCausalLM
from speech_encoder_decoder.modeling_speech_encoder_decoder import MultiDecoderSpeechEncoderDecoderModel
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, TimeMask
import torchaudio.transforms as T
import copy
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from decimal import Decimal, getcontext
from transformers import AdamW
from transformers import get_inverse_sqrt_schedule
from torch.nn.utils.rnn import pad_sequence
from torch import nn

sys.path.append('/project/relater/di/students/enes/ASR/RELATER/DE.EN.AR.UA.ES.ZH.TR.JA/MP3/ZH.EN/seame')
from prepare_data import get_data

sys.path.append('/home/eugan/repos/yapay-net/src/hug/net/')


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    # processor: Any
    feature_extractor: Any
    text_processor: Any
    model_config: Any
    uid_mapper: Any
    dataset: Any
    do_augment: bool
    audio_augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5, leave_length_unchanged=False),
        TimeMask(min_band_part=0.0, max_band_part=0.1, p=0.5),
    ])
    spec_time_masking = T.TimeMasking(time_mask_param=30)
    spec_freq_masking = T.FrequencyMasking(freq_mask_param=30)

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

        # print(bpe_languages_batch, flush=True)
        padded_tensor = self.text_processor(bpe_languages_batch, return_tensors="pt", add_special_tokens=False,
                                            padding=True)
        # print(padded_tensor, flush=True)
        # print("words_batch {} ".format(words_batch))
        # print("decoder_input_ids:  {} shape {}".format(decoder_input_ids, decoder_input_ids.shape))
        # padded_tensor = pad_sequence(bpe_languages_batch, batch_first=True, padding_value=-100)
        # print("bpe_languages_batch: {} ".format(bpe_languages_batch))
        # print(padded_tensor.input_ids, flush=True);
        # print(type(padded_tensor))
        return padded_tensor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        sr = 16000
        audio_lst = list()
        label_lst = list()
        num_languages_lst = list()
        language_ids_lst = list()
        lang_mix_id_lst = list()
        max_length = [250, 500, 750, 1000]
        max_duration = [5 * sr, 10 * sr, 15 * sr, 20 * sr]
        longest_audio = 0

        for el in features:
            uid = el["uid"]
            # print(el["transcript"])
            if len(uid.split("--")) > 1:
                uids = uid.split('--')
                num_languages_lst.append(len(uids))
                indexes = [self.uid_mapper[uid] for uid in uids if uid in self.uid_mapper]
                elements = [self.dataset[idx] for idx in indexes]
                audio = np.concatenate([self.dataset[idx]["audio"]["array"] for idx in indexes])

            else:
                num_languages_lst.append(1)
                audio = el["audio"]["array"]

            if len(audio) < 1:
                # print(el["audio"], flush=True)
                continue

            if not el.get("start", None) == None:
                if int(float(el["start"])) >= 0 and int(float(el["end"])) > 0:
                    start = int(el["start"])
                    end = int(el["end"])
                    audio = audio[start * 16: end * 16]

            new_audio = self.audio_augment(audio, sample_rate=16000) if self.do_augment else audio

            if len(new_audio) <= 20 * sr: audio = new_audio
            if len(audio) > longest_audio: longest_audio = len(audio)

            audio_lst.append(audio)
            label_lst.append(el["transcript"])
        # language_ids_lst.append(el["language_ids"])
        # lang_mix_id_lst.append(el["lang_mix_id"])

        selected_max_length = None
        for duration, length in zip(max_duration, max_length):
            if longest_audio <= duration:
                selected_max_length = length
                break

        try:
            batch = self.feature_extractor(audio_lst, sampling_rate=sr, return_tensors="pt")  # , padding=False)
        # print(batch)
        # print(batch.input_features.shape)
        # batch = self.processor.feature_extractor.pad(
        #                 batch,
        #                 padding="max_length",
        #                 max_length=selected_max_length,
        #                 return_tensors="pt",
        #                 )
        # print(batch)
        # print(batch.input_features.shape)
        except Exception as e:
            print("======")
            print(e)
            for el in features:
                print(el)
                audio = el["audio"]["array"]
                print(len(audio), flush=True)
            print(ASD)

        if self.do_augment:
            input_features = batch.input_features
            input_features = self.spec_time_masking(input_features)
            input_features = self.spec_freq_masking(input_features)
            batch.input_features = input_features
        # print("==")
        # print(batch)
        #     #  print(batch["input_features"].shape)
        # labels_batch = self.processor.tokenizer(label_lst, src_lang=language, tgt_lang=language, return_tensors="pt", padding=True)
        labels_batch = self.text_processor(label_lst, return_tensors="pt", add_special_tokens=False, padding=True)
        # print(label_lst)
        # print(labels_batch)

        # print("&&&&")
        # input_features = [{"input_features": feature["input_features"]} for feature in features]
        # batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        # label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        # labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # labels = self.randomize_positions(labels, num_languages_lst)
        # print(labels)
        # lang_labels_batch = self.get_language_for_token(label_lst, language_ids_lst, labels_batch.input_ids[:, 1:])
        # lang_labels = lang_labels_batch["input_ids"].masked_fill(lang_labels_batch.attention_mask.ne(1), -100)
        # print(lang_labels, flush=True)
        #  lang_labels[:, :2] = -100
        # print(lang_labels, flush=True)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        # print(self.processor.tokenizer.bos_token_id)
        # print(self.model_config.decoder_start_token_id)
        # print(self.model_config.pad_token_id)
        # if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
        # if (labels[:, 0] == self.model_config.decoder_start_token_id).all().cpu().item():
        #    labels = labels[:, 1:]
        # decoder_input_ids = shift_tokens_right(labels, self.model_config.pad_token_id, self.model_config.decoder_start_token_id)
        batch["decoder_input_ids"] = labels_batch.input_ids[:, :-1]
        batch["decoder_attention_mask"] = labels_batch.attention_mask[:, :-1]
        batch["labels"] = labels[:, 1:]
        # batch["lang_decoder_input_ids"] = lang_labels
        # batch["lang_labels"] = lang_labels
        # batch["only_lid_task"] = torch.Tensor([True])
        # batch["decoder_lang_mix_id"] = language_ids_lst

        return batch


all_tr_dataset, all_dev_dataset, concat_tr_dataset = get_data(debug=False)
print("Training data: {}".format(all_tr_dataset))
print("DEV data: {}".format(all_dev_dataset))
training_uid_mapper = {key: idx for idx, key in enumerate(concat_tr_dataset["uid"])}
dev_uid_mapper = {key: idx for idx, key in enumerate(all_dev_dataset["uid"])}


def count_parameters(model: nn.Module):
    total_params = 0
    frozen_params = 0

    for param in model.parameters():
        num_params = param.numel()  # Total number of elements (parameters)
        total_params += num_params
        if not param.requires_grad:
            frozen_params += num_params
    print(f"Total parameters: {total_params}")
    print(f"Frozen parameters: {frozen_params}")
    print(f"Trainable parameters: {total_params - frozen_params}")

    return total_params, frozen_params


model_name = "openai/whisper-small"
task = "transcribe"

processor = AutoProcessor.from_pretrained(model_name)

# model_name = "./model/checkpoint-200"
whisper_config = WhisperConfig.from_pretrained(model_name)
# print(whisper_config)
# model = WhisperForConditionalGenerationMT(whisper_config)#.from_pretrained(model_name)#, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
# model.load_pretrained(model_name)

# config = WhisperConfig2.from_pretrained("openai/whisper-small", has_pre_proj_out=True)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")  # , config=config)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
print("pad_token_id: {}".format(model.config.pad_token_id))

print(model)
count_parameters(model)

lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                         bias="none")  # , modules_to_save=["pre_proj_out"])
model.add_adapter(lora_config)

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# print(DAS)
# Example usage with a simple model
count_parameters(model)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor=processor.feature_extractor,
                                                     text_processor=processor.tokenizer, model_config=model.config,
                                                     uid_mapper=training_uid_mapper, dataset=concat_tr_dataset,
                                                     do_augment=False)
eval_data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor=processor.feature_extractor,
                                                          text_processor=processor.tokenizer, model_config=model.config,
                                                          uid_mapper=dev_uid_mapper, dataset=all_dev_dataset,
                                                          do_augment=False)

learning_rate = 1e-3
warmup_steps = 500

optimizer = AdamW(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=0.0005
)

lr_scheduler = get_inverse_sqrt_schedule(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
)

# model.freeze_encoder()

training_args = Seq2SeqTrainingArguments(
    output_dir="./model",  # change to a repo name of your choice
    # logging_dir="/export/data1/data/eugan/ASR/model/DE.EN.AR.UA.ES.ZH.TR.JA/whisper.v3/log",
    per_device_train_batch_size=15,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,  # 1e-3,#5e-5,
    warmup_steps=warmup_steps,
    # max_steps=180000,
    ddp_find_unused_parameters=True,
    num_train_epochs=100,
    gradient_checkpointing=False,
    fp16=True,
    # group_by_length=True,
    length_column_name="duration",
    # optim="adafactor",
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=1,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    eval_accumulation_steps=100,
    dataloader_num_workers=4,
    per_device_eval_batch_size=20,
    dataloader_persistent_workers=False,
    label_smoothing_factor=0,  # 0.1,
    #   dataloader_prefetch_factor=2,
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    # metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    # push_to_hub=True,
)
from transformers import Seq2SeqTrainer

print("all_tr_dataset: {}".format(all_tr_dataset))
print(type(all_tr_dataset))
getcontext().prec = 50
probabilities = list()
for _ in range(len(all_tr_dataset)):
    probability = Decimal(1) / Decimal(len(all_tr_dataset))
    probabilities.append(probability)
train_dataset = interleave_datasets(list(all_tr_dataset.values()), probabilities, seed=42)
print("TTTTT: {}".format(train_dataset))

# trainer = Seq2SeqTrainer(
trainer = MemSeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset_dict=all_tr_dataset,
    train_dataset=train_dataset,
    eval_dataset=all_dev_dataset,
    data_collator=data_collator,
    eval_data_collator=eval_data_collator,
    optimizers=(optimizer, lr_scheduler),
    # compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=5)]
)
# with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
trainer.train(resume_from_checkpoint=False)

# if _model == model_name:
#    print("SAME")
#    trainer.train(resume_from_checkpoint=False)
# else:
#    print("NOT SAME")
#    trainer.train(resume_from_checkpoint=False)
