#!/usr/bin/env python
import sys
import tempfile
from itertools import islice
from time import time
from multiprocessing import Pool

from translate import main as translate_main


def hasopt(opt):
    return ('-' + opt) in sys.argv


def popopt(opt):
    # TODO handle different option formats, e.g. --opt or -opt=val
    idx = sys.argv.index('-' + opt)
    sys.argv.pop(idx)
    return sys.argv.pop(idx)


def distribute_to_tempfiles(srcfile, n):
    tmpfiles = [tempfile.NamedTemporaryFile('w', encoding='utf8') for _ in range(n)]

    with open(srcfile, 'r', encoding='utf8') as f:
        nlines = len(list(f))
        f.seek(0)
        # round up
        linesperpart = int((nlines + n - 1) / n)
        for tf in tmpfiles:
            for line in islice(f, linesperpart):
                tf.write(line)
            tf.flush()

    return tmpfiles


def run_part(args):
    infile, goldfile, outfile, gpu = args
    start = time()
    sys.argv += ['-gpu', gpu, '-src', infile, '-output', outfile]
    if goldfile:
        sys.argv += ['-tgt', goldfile]
    translate_main()
    print('GPU {} done after {:.1f}s'.format(gpu, time() - start))


srcfile = popopt('src')
outfile = popopt('output')
gpu_list = popopt('gpus').split(',')

# (1) distribute input lines to N tempfiles
inparts = distribute_to_tempfiles(srcfile, len(gpu_list))
if hasopt('tgt'):
    goldfile = popopt('tgt')
    goldparts = distribute_to_tempfiles(goldfile, len(gpu_list))
else:
    goldparts = [None for _ in range(len(gpu_list))]

# (2) run N processes translating one tempfile each
outparts = [tempfile.NamedTemporaryFile('r', encoding='utf8') for _ in gpu_list]
filenames = lambda tmpfiles: [tf.name if tf else None for tf in tmpfiles]
with Pool(len(gpu_list)) as p:
    p.map(run_part, zip(filenames(inparts), filenames(goldparts), filenames(outparts), gpu_list))

# (3) concatenate tempfiles into one output file
with open(outfile, 'w', encoding='utf8') as f:
    for outp in outparts:
        f.write(outp.read())
