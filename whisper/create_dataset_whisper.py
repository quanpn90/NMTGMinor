from datasets import load_dataset, concatenate_datasets, Audio
from datasets import DatasetDict
import argparse
from transformers import AutoProcessor

parser = argparse.ArgumentParser(description='create_dataset_whisper')
parser.add_argument('-train_json', required=True,
                    help="Path to the training data in json format")

parser.add_argument('-valid_json', required=True,
                    help="Path to the validation data in json format")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-preprocessor', default="none",
                    help="if not None: load the preprocessor specified to preprocess the data.")

args = parser.parse_args()

train_data = load_dataset("json", data_files=args.train_json)["train"]
val_data = load_dataset("json", data_files=args.valid_json)["train"]

# Combine datasets into a DatasetDict
dataset = DatasetDict({
    "train": train_data,
    "validation": val_data
})

dataset = dataset.cast_column("audio_path", Audio())


print(dataset)
print(dataset["train"][0])  # First training sample
print(dataset["validation"][0])  # First validation sample

# Save the DatasetDict to a directory
dataset.save_to_disk(args.save_data)

if args.preprocessor != "none":

    print("Processing data ....")
    model_name = args.preprocessor  # Choose the base model size
    processor = AutoProcessor.from_pretrained(model_name)

    def preprocess_function(batch):
        # Extract audio features and tokenize the transcription
        audio = batch["audio_path"]["array"]
        sampling_rate = 16000  # Get the sampling rate
        input_features = processor.feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
        labels = processor.tokenizer(batch["transcription"]).input_ids
        return {"input_features": input_features, "labels": labels}

    # Apply preprocessing
    dataset = dataset.map(preprocess_function, remove_columns=["audio_path", "transcription"])

    dataset.save_to_disk(args.save_data + ".preprocessed")