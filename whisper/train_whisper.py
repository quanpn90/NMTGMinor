import argparse
import torch
from datasets import load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorWithPadding
from transformers import Seq2SeqTrainer
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback

# memory efficient Whisper
from memory_efficient_whisper import (MemoryEfficientWhisper,
                                      MemoryEfficientWhisperEncoderLayer,
                                      MemoryEfficientLayerNorm,
                                      create_whisper_model)

# Step 1: Parsing arguments

parser = argparse.ArgumentParser(description='create_dataset_whisper')
parser.add_argument('-dataset', required=True,
                    help="Path to the dataset in huggingface format")
parser.add_argument('-checkpoint', type=str, default="none",
                    help='Path to the checkpoint. Use the default (model card) if none is provided')
parser.add_argument('-save_steps', type=int, default=0,
                    help='Number of update steps per checkpoint ')
parser.add_argument('-logging_steps', type=int, default=0,
                    help='Number of update steps per logging (and evaluation) ')
parser.add_argument('-num_epoch', type=int, default=2,
                    help='Number of epoches in training ')
parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Peak learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-weight_decay', type=float, default=0.001,
                    help="""Weight Decay for AdamW""")
parser.add_argument('-output', required=False, default='./whisper_finetune',
                    help='Default folder to save the checkpoints')
parser.add_argument('-batch_size', type=int, default=8,
                    help='Number of update steps per checkpoint ')
parser.add_argument('-bsz_accumulate', type=int, default=1,
                    help='Number of update steps per logging (and evaluation) ')
parser.add_argument('-text_normalizer', type=str, default="none",
                    help='Text Normalizer: if not none select languages such as "english" or "german".')
parser.add_argument('-spec_augment', action='store_true',
                    help="Use spec augmentation")

args = parser.parse_args()

# Step 2: Loading the generated dataset

# Load dataset (replace "path/to/save/dataset" with your actual path)
dataset = load_from_disk(args.dataset)

dataset = dataset.cast_column("audio_path", Audio())

# Step 3: Load model and processor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model_name = "openai/whisper-large-v3-turbo"  # Choose the base model size
checkpoint_path = args.checkpoint
if checkpoint_path == "none":
    checkpoint_path = model_name


processor = WhisperProcessor.from_pretrained(model_name)
model = create_whisper_model(checkpoint_path, torch_dtype, attn_implementation="flash_attention_2")


# Adjust model settings for fine-tuning
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if args.spec_augment:
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05

# Step 4: Prepare the dataset for training
# DO THIS ON THE FLY

# def preprocess_function(batch):
#     # Extract audio features and tokenize the transcription
#     audio = batch["audio_path"]["array"]
#     input_features = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
#     labels = processor.tokenizer(batch["transcription"]).input_ids
#     return {"input_features": input_features, "labels": labels}
#
# # Apply preprocessing
# dataset = dataset.map(preprocess_function, remove_columns=["audio_path", "transcription"])


# Step 5: Define the training arguments:

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output,  # Directory to save model checkpoints
    per_device_train_batch_size=args.batch_size,  # Adjust based on your GPU memory
    gradient_accumulation_steps=args.bsz_accumulate,  # Effective batch size multiplier
    learning_rate=1e-5,
    num_train_epochs=args.num_epoch,
    evaluation_strategy="steps",
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    save_total_limit=5,  # Only keep the last two checkpoints
    logging_dir="./logs",
    predict_with_generate=True,
    bf16=True,  # Use mixed precision training (if supported)
    remove_unused_columns=False,  # Prevent Trainer from removing necessary columns
    dataloader_num_workers=16  # Number of workers for data loading
)

if args.text_normalizer != "none":
    print("[INFO] Using the %s text normalizer..." % args.text_normalizer)
    processor.language = args.text_normalizer

# data_collator = DataCollatorForSeq2Seq(processor, model=model, return_tensors="pt")
class WhisperDataCollator(DataCollatorWithPadding):
    def __init__(self, processor, normalize="none"):
        # Initialize the parent class with the tokenizer for padding
        super().__init__(tokenizer=processor.tokenizer)
        self.processor = processor
        self.normalize = normalize != "none"

    def __call__(self, samples):
        # Extract audio arrays and transcriptions
        audio_arrays = [sample["audio_path"]["array"] for sample in samples]
        sampling_rate = 16000
        if self.normalize:
            transcriptions = list()
            for sample in samples:
                t = sample["transcription"]
                normalized_t = self.processor.tokenizer.normalize(t)
                transcriptions.append(normalized_t)
        else:
            transcriptions = [sample["transcription"] for sample in samples]

        # Process audio features
        input_features = self.processor.feature_extractor(
            audio_arrays, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features

        # Tokenize the transcriptions and pad them
        labels = self.processor.tokenizer(
            transcriptions, padding=True, truncation=False, return_tensors="pt"
        ).input_ids

        input_features = input_features.to(torch_dtype)

        return {"input_features": input_features, "labels": labels}


# Define the custom data collator
data_collator = WhisperDataCollator(processor, normalize=args.text_normalizer)


def inverse_sqrt_scheduler(optimizer, num_warmup_steps=1000):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return (num_warmup_steps ** 0.5) / (current_step ** 0.5)

    return LambdaLR(optimizer, lr_lambda)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        # Define the Adam optimizer with learning rate max = 0.0005
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer):
        # print(optimizer, flush=True)
        self.lr_scheduler = inverse_sqrt_scheduler(optimizer, num_warmup_steps=1000)
        return self.lr_scheduler

    def train(self, *args, **kwargs):
        # TODO: add option to Compile the model before training
        # self.model = torch.compile(self.model)
        return super().train(*args, **kwargs)


# Define the scheduler




class ConsoleLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Print the loss and learning rate
            loss = logs.get("loss", None)
            learning_rate = logs.get("learning_rate", None)
            print(f"Step: {state.global_step}")
            if loss is not None:
                print(f"  Loss: {loss:.4f}")
            if learning_rate is not None:
                print(f"  Learning Rate: {learning_rate:.6f}")


# Step 7: Train the model
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    # processing_class
    callbacks=[ConsoleLoggingCallback()]
)

# Start training
trainer.train()
