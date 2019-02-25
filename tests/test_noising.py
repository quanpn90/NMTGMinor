
import argparse

from nmtg.data import Dictionary
from nmtg.data.noisy_text import NoisyTextDataset
from nmtg.data.text_lookup_dataset import TextLookupDataset
from nmtg.tasks.denoising_text_task import DenoisingTextTask

parser = argparse.ArgumentParser()
DenoisingTextTask.add_options(parser)
args = parser.parse_args()

task = DenoisingTextTask.setup_task(args)
dictionary = Dictionary.infer_from_text(task.tgt_dataset)

noisy_text = NoisyTextDataset(TextLookupDataset(task.src_dataset, dictionary, True,
                                                args.lower, False, False, False),
                              args.word_shuffle, args.noise_word_dropout, args.word_blank, args.bpe_symbol)

for i in range(len(noisy_text)):
    print(task.tgt_dataset[i])
    print(dictionary.string(noisy_text[i]))
    input()
