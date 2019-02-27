import argparse
import sys

from nmtg import custom_logging, convert, options
from nmtg.trainers.nmt_trainer import NMTTrainer

# Example for how to use NMTGMinor from a program and not from the command line

parser = argparse.ArgumentParser()
options.add_general_options(parser)
NMTTrainer.add_eval_options(parser)
# either add log options or set up your own logging. NMTGMinor uses logging.getLogger(__name__) to build its loggers
custom_logging.add_log_options(parser)

# To choose custom parameter values, either modify the args object after parsing or use parser.set_defaults
parser.set_defaults(beam_size=2, alpha=0.5)

args = parser.parse_args([])  # parse the empty list so we don't use command line arguments

# Use this to let NMTGMinor configure logging
logger = custom_logging.setup_logging_from_args(args, 'test')

trainer = NMTTrainer(args)

model = trainer.load_checkpoint(convert.load_checkpoint("your_checkpoint_filename.pt"))

for output in trainer.online_translate(model, sys.stdin):
    print(output)
