import argparse
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration


from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorWithPadding
from transformers import Seq2SeqTrainer
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Step 1: Parsing arguments

parser = argparse.ArgumentParser(description='create_dataset_whisper')
parser.add_argument('-dataset', required=True,
                    help="Path to the dataset in huggingface format")

args = parser.parse_args()

# Step 2: Loading the generated dataset

# Load dataset (replace "path/to/save/dataset" with your actual path)
dataset = load_from_disk(args.dataset)


# Step 3: Load model and processor

# Load model and processor
model_name = "openai/whisper-large-v3-turbo"  # Choose the base model size
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Adjust model settings for fine-tuning
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Step 4: Prepare the dataset for training

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
    output_dir="./whisper-finetuned",  # Directory to save model checkpoints
    per_device_train_batch_size=4,     # Adjust based on your GPU memory
    gradient_accumulation_steps=8,     # Effective batch size multiplier
    learning_rate=1e-5,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_steps=500,
    save_total_limit=2,                # Only keep the last two checkpoints
    logging_dir="./logs",
    predict_with_generate=True,
    fp16=True,                         # Use mixed precision training (if supported)
    remove_unused_columns=False  # Prevent Trainer from removing necessary columns
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
            transcriptions, padding=True, truncation=True, return_tensors="pt"
        ).input_ids

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

# Define the scheduler



from transformers import TrainerCallback

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
    callbacks=[ConsoleLoggingCallback()]
)

# Start training
trainer.train()


