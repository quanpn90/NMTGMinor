#!/usr/bin/env python

# from fairseq.checkpoint_utils import load_model_ensemble_and_task, load_checkpoint_to_cpu

from __future__ import division

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import sys
import h5py as h5
import numpy as np
from onmt.inference.fast_translator import FastTranslator
from onmt.inference.stream_translator import StreamTranslator
from torch.cuda.amp import autocast


parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-lm', required=False,
                    help='Path to language model .pt file. Used for cold fusion')
parser.add_argument('-vocab_list', default="",
                    help='A Vocabulary list (1 word per line). Only are these words generated during translation.')
parser.add_argument('-autoencoder', required=False,
                    help='Path to autoencoder .pt file')
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-attributes', default="",
                    help='Attributes for the decoder. Split them by | ')
parser.add_argument('-ensemble_weight', default="",
                    help='Weight for ensembles. Default as uniform. Split them by | and they will be normalized later')
parser.add_argument('-sub_ensemble_weight', default="",
                    help='Weight for ensembles. Default as uniform. Split them by | and they will be normalized later')

parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")
parser.add_argument('-concat', type=str, default="1",
                    help="Concate sequential audio features to decrease sequence length")
parser.add_argument('-asr_format', default="h5", required=False,
                    help="Format of asr data h5 or scp")
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img|audio].")
parser.add_argument('-previous_context', type=int, default=0,
                    help="Number of previous sentence for context")
parser.add_argument('-max_memory_size', type=int, default=512,
                    help="Number of memory states stored in the buffer for XL models")

parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-scp_output', default='output.scp',
                    help="""Path to output the feature paths""")

parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size (in audio samples)')

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-fp16', action='store_true',
                    help='To use floating point 16 in decoding')


def _is_oversized(batch, new_sent_size, batch_size):
    """
    Function to see if adding new sentence will make the current batch
    :param batch:
    :param new_sent_size:
    :param batch_size_words:
    :return:
    """

    # Always return False if empty
    if len(batch) == 0:
        return False

    current_max_length = max([sent.size(0) for sent in batch])

    # Because adding a new sentence will potential enlarge the area of the rectangle, we need to check
    if max(current_max_length, new_sent_size) * (len(batch) + 1) > batch_size:
        return True

    return False


def write_codes(utts, codes, padding_mask, out_scp, opt):
    # cache_wav = ''

    codes = codes.cpu()
    bsz, seq_len, groups = codes.size()

    if padding_mask is not None:
        lengths = (1 - padding_mask.long()).sum(dim=1).long()
    else:
        lengths = torch.LongTensor(bsz).fill_(seq_len)

    assert len(utts) == bsz

    for i in range(bsz):
        code_ = codes[i, 0:lengths[i], :]
        code_ = code_.prod(dim=-1, keepdim=False)
        code_ = code_.tolist()

        code_ = " ".join([str(c) for c in code_])
        seg_name = utts[i]
        # print(seg_name)
        # print(code_)

        out_scp.write(code_ + "\n")
        # dic = {seg_name: feature_}

        # from onmt.data.kaldiio.io import write_ark_file
        # write_ark_file(out_ark, out_scp, dic)


def build_data(src_sents):

    from onmt.data.wav_dataset import WavDataset
    src_data = src_sents
    data_type = 'wav'

    tgt_data = None

    src_lang_data = [torch.Tensor([0])]
    tgt_lang_data = None

    return onmt.Dataset(src_data, tgt_data,
                        src_langs=src_lang_data, tgt_langs=tgt_lang_data,
                        batch_size_words=sys.maxsize,
                        max_src_len=sys.maxsize,
                        data_type=data_type,
                        batch_size_sents=sys.maxsize,
                        src_align_right=False,
                        past_src_data=None)


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    print("Loading Wav2vec 2.0 model ...")
    from onmt.models.speech_recognizer.wav2vec2 import FairseqWav2VecQuantizer
    model = FairseqWav2VecQuantizer(opt.model)
    print("Done")

    if opt.cuda:
        model = model.cuda()
    model.eval()

    scp_out = open(opt.scp_output, 'w')

    audio_data = open(opt.src)
    from onmt.utils import safe_readaudio

    i = 0

    n_models = len(opt.model.split("|"))
    src_batch = list()
    src_utts = list()

    while True:
        try:
            line = next(audio_data).strip().split()
            utt = line[0]

            if len(line) == 2:
                wav_path = line[1]
                start = 0
                end = 0
            else:
                wav_path, start, end = line[1], float(line[2]), float(line[3])
            line = safe_readaudio(wav_path, start=start, end=end, sample_rate=16000)

        except StopIteration:
            break

        src_length = line.size(0)

        """
        Read features output from wav2vec model and write into scp/ark file just like Kaldi w/ logmel features
        """

        if _is_oversized(src_batch, src_length, opt.batch_size):
            # If adding a new sentence will make the batch oversized
            # Then do translation now, and then free the list
            print("Batch sizes :", len(src_batch))
            dataset = build_data(src_batch)
            batch = dataset.get_batch(0)
            batch.cuda()

            with torch.no_grad():
                with autocast(enabled=opt.fp16):
                    codes, padding_mask = model(batch)
            write_codes(src_utts, codes, padding_mask, scp_out, opt)

            src_batch = []
            src_utts = []
        src_batch.append(line)
        src_utts.append(utt)

    # catch the last batch
    if len(src_batch) != 0:
        print("Batch sizes :", len(src_batch), )
        dataset = build_data(src_batch)
        batch = dataset.get_batch(0)
        batch.cuda()
        with autocast(enabled=opt.fp16):
            codes, padding_mask = model(batch)
        write_codes(src_utts, codes, padding_mask, scp_out, opt)

        src_batch = []
        src_utts = []

    ark_out.close()
    scp_out.close()
