import argparse
import torch
from datasets import load_from_disk
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
                                      MemoryEfficientLayerNorm)

# Step 1: Parsing arguments

parser = argparse.ArgumentParser(description='create_dataset_whisper')
parser.add_argument('-dataset', required=True,
                    help="Path to the dataset in huggingface format")
parser.add_argument('-save_steps', type=int, default=0,
                    help='Number of update steps per checkpoint ')
parser.add_argument('-logging_steps', type=int, default=0,
                    help='Number of update steps per logging (and evaluation) ')

args = parser.parse_args()

# Step 2: Loading the generated dataset

# Load dataset (replace "path/to/save/dataset" with your actual path)
dataset = load_from_disk(args.dataset)

# Step 3: Load model and processor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model_name = "openai/whisper-large-v3-turbo"  # Choose the base model size
processor = WhisperProcessor.from_pretrained(model_name)
model = MemoryEfficientWhisper.from_pretrained(model_name,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch_dtype,
                                          attn_implementation="flash_attention_2",
                                          )


# Function to replace and transfer weights
def replace_layer_with_weights(model, config):
    for i in range(len(model.model.encoder.layers)):
        old_layer = model.model.encoder.layers[i]
        new_layer = MemoryEfficientWhisperEncoderLayer(config)

        # Copy weights from the old layer to the new one
        new_layer.load_state_dict(old_layer.state_dict())

        dtype = next(old_layer.parameters()).dtype
        new_layer.to(dtype)

        # Replace the layer in the encoder
        model.model.encoder.layers[i] = new_layer

def replace_layernorm_with_memory_efficient(model):
    for name, module in model.named_children():
        # Check if the current module is LayerNorm
        if isinstance(module, torch.nn.LayerNorm):

            custom_layer = MemoryEfficientLayerNorm(module.normalized_shape, module.eps)

            # Copy weights and biases
            custom_layer.weight.data.copy_(module.weight.data)
            custom_layer.bias.data.copy_(module.bias.data)

            # convert to the right type
            custom_layer.to(module.weight.dtype)
            # Replace with MemoryEfficientLayerNorm
            setattr(model, name, custom_layer)
        else:
            # Recursively apply to submodules
            replace_layernorm_with_memory_efficient(module)

# replace the layer first
replace_layer_with_weights(model, model.config)
replace_layernorm_with_memory_efficient(model)

# Adjust model settings for fine-tuning
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

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
    output_dir="./whisper-finetuned-flash",  # Directory to save model checkpoints
    per_device_train_batch_size=16,  # Adjust based on your GPU memory
    gradient_accumulation_steps=2,  # Effective batch size multiplier
    learning_rate=1e-5,
    num_train_epochs=3,
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


# Step 6: Set Up the Data Collator

# data_collator = DataCollatorForSeq2Seq(processor, model=model, return_tensors="pt")
class WhisperDataCollator(DataCollatorWithPadding):
    def __init__(self, processor):
        # Initialize the parent class with the tokenizer for padding
        super().__init__(tokenizer=processor.tokenizer)
        self.processor = processor

    def __call__(self, samples):
        # Extract audio arrays and transcriptions
        audio_arrays = [sample["audio_path"]["array"] for sample in samples]
        sampling_rate = 16000
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
data_collator = WhisperDataCollator(processor)


def inverse_sqrt_scheduler(optimizer, num_warmup_steps=1000):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return (num_warmup_steps ** 0.5) / (current_step ** 0.5)

    return LambdaLR(optimizer, lr_lambda)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        # Define the Adam optimizer with learning rate max = 0.0005
        self.optimizer = AdamW(self.model.parameters(), lr=5e-4)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer):
        # print(optimizer, flush=True)
        self.lr_scheduler = inverse_sqrt_scheduler(optimizer, num_warmup_steps=1000)
        return self.lr_scheduler

    def train(self, *args, **kwargs):
        # Compile the model before training
        self.model = torch.compile(self.model)
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
