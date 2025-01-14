import os
import sys
import warnings
import argparse
import torch
from datasets import load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorWithPadding
from transformers import Seq2SeqTrainer
# from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback, TrainerState, TrainerControl

# Get the local rank from the environment variables
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

if local_rank != 0:
    # Suppress stdout and stderr for non-zero ranks
    sys.stdout = open(os.devnull, "w")
    # sys.stderr = open(os.devnull, "w")
    warnings.filterwarnings("ignore")  # Ignore all warnings

import warnings

warnings.filterwarnings("ignore", message="Positional args are being deprecated, use kwargs instead.")
warnings.filterwarnings("ignore",
                        message="Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.")

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
parser.add_argument('-max_steps', type=int, default=-1,
                    help='Max number of training steps (taking over num_epoch)')

parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Peak learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-weight_decay', type=float, default=0.001,
                    help="""Weight Decay for AdamW""")
parser.add_argument('-warm_up_steps', type=int, default=2000,
                    help='Number of warm up steps for learning rate')

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
parser.add_argument('-fsdp', action='store_true',
                    help="Use Fully Sharded Distributed Training")
parser.add_argument('-teacher_distillation', type=float, default=0
                    , help='Use the original Whisper model as a teacher')

parser.add_argument('-filter_length', action='store_true',
                    help="Use spec augmentation")
parser.add_argument('-check_audio_path', action='store_true',
                    help="Use spec augmentation")

parser.add_argument('-ema', action='store_true',
                    help="Use exponential moving average during training")
parser.add_argument('-ema_rate', type=float, default=0.9
                    , help='Dropout value of the model')

parser.add_argument('-streaming', action='store_true',
                    help="Use exponential moving average during training")
parser.add_argument('-freeze_embedding', action='store_true',
                    help="Use exponential moving average during training")

parser.add_argument('-optim', type=str, default="adam",
                    help='Optimizer: ["adam", "rmsprop", "sgd".')

parser.add_argument('-lr_scheduler', type=str, default="inv_sqrt",
                    help='LR scheduler: ["inv_sqrt", "cosine".')
parser.add_argument('-dropout', type=float, default=0.0
                    , help='Dropout value of the model')

args = parser.parse_args()

# Step 2: Loading the generated dataset
model_name = "openai/whisper-large-v3-turbo"  # Choose the base model size
processor = WhisperProcessor.from_pretrained(model_name)

dataset_names = args.dataset.split("|")

datasets = list()
for dataset_name in dataset_names:
    dataset = load_from_disk(dataset_name)

    if args.filter_length:
        max_length = 448
        print("[INFO] Filtering dataset %s the sequences longer than 448 tokens..." % dataset_name)


        def filter_long_examples(example):
            # Ensure both input_ids and labels are within the max length
            # return len(example["input_ids"]) <= max_length and len(example["labels"]) <= max_length

            x = processor.tokenizer(example["transcription"]).input_ids
            return len(x) <= max_length


        # Apply the filter to your dataset
        dataset = dataset.filter(filter_long_examples, num_proc=64)

    if args.check_audio_path:

        def filter_audio_path(example):
            # Ensure both input_ids and labels are within the max length
            # return len(example["input_ids"]) <= max_length and len(example["labels"]) <= max_length

            # x = processor.tokenizer(example["transcription"]).input_ids
            path = example["audio_path"]

            return os.path.exists(path)


        # Apply the filter to your dataset
        dataset = dataset.filter(filter_audio_path, num_proc=64)

    dataset = dataset.cast_column("audio_path", Audio())

    datasets.append(dataset)

# Step 3: Load model and processor


device = device if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
# Load model and processor
checkpoint_path = args.checkpoint
if checkpoint_path == "none":
    checkpoint_path = model_name

# passing a device_map requires the low_cpu_mem_usage
model = create_whisper_model(checkpoint_path, torch_dtype, attn_implementation="flash_attention_2",
                             low_cpu_mem_usage=True,
                             device_map={"": device})

if args.freeze_embedding:
    model.proj_out.weight.requires_grad = False
    if model.proj_out.bias is not None:
        model.proj_out.bias.requires_grad = False
    model.model.decoder.embed_tokens.weight.requires_grad = False

if args.dropout > 0:
    model.config.dropout = args.dropout


# Adjust model settings for fine-tuning
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if args.spec_augment:
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05

# Create optimizer

optimized_params = [p for p in model.parameters() if p.requires_grad]
total_params = sum([p.numel() for p in model.parameters()])
total_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

print(f"[INFO] Total params: {total_params}")
print(f"[INFO] Total trainable params: {total_trainable_params}")

optimizer_grouped_parameters = [
    {
        "params": optimized_params,
        "weight_decay": args.weight_decay,
    }
]

if args.optim in ['adam', 'adamw']:

    optim_class = torch.optim.AdamW

elif args.optim == 'sgd':

    optim_class = torch.optim.SGD

else:
    raise NotImplementedError

if args.optim in ['adam', 'adamw']:

    betas = (0.9, 0.999)

    if args.ema:
        raise NotImplementedError("EMA is not implemented for Adam optimizers yet.")
    else:
        optimizer = optim_class(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=betas)



elif args.optim == 'sgd':

    optim_class = torch.optim.SGD

    if args.ema:
        class EMASGD(torch.optim.SGD):

            def __init__(self, params, ema_rate, **kwargs):
                super().__init__(params, **kwargs)
                self.ema_rate = ema_rate
                self.counter = 0
                # self.shadow_weights = {id(p): p.clone().detach() for group in self.param_groups for p in group['params']}
                self.shadow_weights = dict()

                for group in self.param_groups:
                    for p in group['params']:
                        self.shadow_weights[id(p)] = p.data.new_zeros(p.size())
                        self.shadow_weights[id(p)].add_(p.data)

            def step(self, closure=None):
                """
                Performs a single optimization step and updates EMA weights.
                """
                # Perform the base SGD step
                loss = super().step(closure)
                self.counter += 1

                # alpha = (1 / self.counter) * self.ema_rate
                # alpha = max(0.01, min(alpha, 0.5))
                alpha = 1 - self.ema_rate

                total = 0
                # Update EMA weights
                with torch.no_grad():
                    for group in self.param_groups:
                        for p in group['params']:
                            if p.requires_grad:
                                param_id = id(p)
                                if param_id in self.shadow_weights:

                                    if p.device != self.shadow_weights[param_id].device:
                                        self.shadow_weights[param_id] = self.shadow_weights[param_id].to(p.device)
                                    # print("[DEBUGGING] alpha:", alpha)
                                    # print("[DEBUGGING] BEFORE")

                                    # w_sum = p.data.sum().item()
                                    # sw_sum = self.shadow_weights[param_id].sum().item()
                                    #
                                    # if w_sum != sw_sum:
                                    #     print(f"p.data sum: {w_sum}, "
                                    #           f"shadow_weights sum: {sw_sum}")
                                    p.data.mul_(alpha).add_(self.shadow_weights[param_id] * (1 - alpha))
                                    # w_sum = p.data.sum().item()
                                    # sw_sum = self.shadow_weights[param_id].sum().item()
                                    #
                                    # if w_sum != sw_sum:
                                    #     # print("[DEBUGGING] AFTER")
                                    #     print(f"p.data sum: {w_sum}, "
                                    #           f"shadow_weights sum: {sw_sum}")
                                    #
                                    #     total += p.data.numel()
                                        # print("---")

                                    self.shadow_weights[param_id].copy_(p.data)

                # print("Total number of EMA-updated params:", total)

                return loss


        optimizer = EMASGD(optimized_params, ema_rate=args.ema_rate, weight_decay=args.weight_decay,
                           lr=args.learning_rate)
    else:
        optimizer = optim_class(optimizer_grouped_parameters,
                                lr=args.learning_rate)





if args.lr_scheduler in ['inv_sqrt', 'noam']:
    def inverse_sqrt_scheduler(_optimizer, num_warmup_steps=1000):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return (num_warmup_steps ** 0.5) / (current_step ** 0.5)

        return LambdaLR(_optimizer, lr_lambda)


    lr_scheduler = inverse_sqrt_scheduler(optimizer, num_warmup_steps=args.warm_up_steps)

elif args.lr_scheduler == 'cosine':
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warm_up_steps)

else:
    raise NotImplementedError

if args.teacher_distillation > 0:
    print("[INFO] Using the Whisper model as a teacher")
    # actually student and teacher can be the same xD
    teacher = create_whisper_model(model_name, torch_dtype, attn_implementation="flash_attention_2",
                                   low_cpu_mem_usage=True,
                                   device_map={"": device})

    # freeze the parameters for the teacher
    for param in teacher.parameters():
        param.requires_grad = False
    model.teacher = teacher
    model.teacher_distillation = args.teacher_distillation
else:
    teacher = None

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
# TODO: check layerdrop values to determine ddp_find_unused_parameters


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


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    # def create_optimizer(self):
    #     # Define the Adam optimizer with learning rate max = 0.0005
    #     self.optimizer = optimizer
    #
    # def create_scheduler(self, num_training_steps: int, optimizer):
    #     # print(optimizer, flush=True)
    #     self.lr_scheduler = inverse_sqrt_scheduler(optimizer, num_warmup_steps=1000)
    #     return self.lr_scheduler

    def save_model(self, output_dir=None, _internal_call=True):
        # Temporarily set the submodule to None

        if args.teacher_distillation > 0:

            _teacher = self.model.teacher
            self.model.teacher = None

            # Call the original save_model method
            super().save_model(output_dir, _internal_call)

            # Restore the submodule after saving
            self.model.teacher = _teacher
        else:
            super().save_model(output_dir, _internal_call)

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
            # distill_loss = loss.get("distilled_loss", None)
            learning_rate = logs.get("learning_rate", None)
            print(f"Step: {state.global_step}")
            if loss is not None:
                print(f"  Loss: {loss:.4f}")
            # if distill_loss is not None:
            #     print(f"  DT Loss: {distill_loss:.4f}")
            if learning_rate is not None:
                print(f"  lr: {learning_rate:.6f}")


callbacks = [ConsoleLoggingCallback()]
update_count = 0
ema_rate = args.ema_rate

# if args.ema:
    # class EMACallback(TrainerCallback):
    #     def __init__(self, ema_rate=1.0):
    #         """
    #         Exponential averaging callback for weight updates.
    #
    #         Args:
    #             ema_rate
    #         """
    #         # self.alpha_values = alpha_values  # Coefficients for exponential averaging
    #         # self.ema_parameters = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
    #         self.storage = {name: param.clone().detach() for name, param in model.named_parameters()}
    #         self.ema_rate = ema_rate
    #
    #
    #     def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None,
    #                     optimizer=None, **kwargs):
    #         """
    #         Perform exponential averaging of weights after optimizer step.
    #         """
    #         # if self.alpha_values is None:
    #         #     # Default alpha=0.9 for all parameter groups if not provided
    #         #     self.alpha_values = [0.9] * len(optimizer.param_groups)
    #
    #         global update_count
    #
    #         update_count += 1
    #         num_update = update_count
    #
    #         if num_update == 0:  # Prevent division by zero on the first step
    #             return
    #
    #         # alpha = max(1 / num_update * args.ema_rate, 0.5)
    #         # alpha = min(alpha, 0.99)
    #         # alpha = 1 / num_update * self.ema_rate
    #         #
    #         # alpha = max(0.01, min(alpha, 0.5))
    #         alpah
    #
    #         for name, param in model.named_parameters():
    #
    #             if param.requires_grad:
    #                 with torch.no_grad():
    #                     self.storage[name] = self.storage[name].to(param.device)
    #                     param.data.mul_(alpha).add_(self.storage[name], alpha=1.0 - alpha)
    #                     print(param.sum().item(), self.storage[name].sum().item())
    #                     # Update the storage with the current weight
    #                     self.storage[name].copy_(param.data)

    # print("[INFO] Training the model with Exponential Moving Average SGD...")
    # callbacks.append(EMACallback(ema_rate=ema_rate))


# what happens if we have a list of dataset?
# dataset_list =

# # Step 7: Train the model
# trainer = CustomSeq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
#     data_collator=data_collator,
#     processing_class=processor.feature_extractor,
#     # processing_class
#     callbacks=callbacks
# )
#
# # Start training
# trainer.train()


def continual_learning_single_trainer(model, datasets,
                                      dataset_names, callbacks):
    """
    Train a model sequentially on multiple datasets using a single Trainer instance.

    Args:
        model: Pre-trained model to be fine-tuned.
        datasets: List of datasets for sequential training.
        training_args: TrainingArguments for the Hugging Face Trainer.

    Returns:
        The model after continual learning.
    """
    for dataset, dataset_name in zip(datasets, dataset_names):
        output_dir = os.path.join(args.output, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,  # Directory to save model checkpoints
            per_device_train_batch_size=args.batch_size,  # Adjust based on your GPU memory
            gradient_accumulation_steps=args.bsz_accumulate,  # Effective batch size multiplier
            learning_rate=1e-5,
            max_steps=args.max_steps,
            num_train_epochs=args.num_epoch,
            eval_strategy="steps",
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            save_total_limit=5,  # Only keep the last two checkpoints
            logging_dir="./logs",
            predict_with_generate=True,
            bf16=True,  # Use mixed precision training (if supported)
            remove_unused_columns=False,  # Prevent Trainer from removing necessary columns
            dataloader_num_workers=16,  # Number of workers for data loading
            ddp_find_unused_parameters=False,
            # Enables FSDP with full sharding and auto-wrapping
            fsdp="full_shard auto_wrap" if args.fsdp else "",
            # Specify layers to wrap
            fsdp_transformer_layer_cls_to_wrap="MemoryEfficientWhisperEncoderLayer,WhisperDecoderLayer" if args.fsdp else None
        )

        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            processing_class=processor.feature_extractor,
            optimizers=(optimizer, lr_scheduler),
            # processing_class
            callbacks=callbacks
        )

        # initial_metrics = trainer.evaluate()
        # print(f"Initial Validation Metrics: {initial_metrics}")

        print("[INFO] Start training on dataset %s" % dataset_name)
        trainer.train()


continual_learning_single_trainer(model, datasets, dataset_names, callbacks)
