#!/usr/bin/env python
import sys
import tempfile
from itertools import islice
from time import time
from multiprocessing import Pool

from translate import main as translate_main

onmt = sys.path[0]

def popopt(opt):
    # TODO handle different option formats, e.g. --opt or -opt=val
    idx = sys.argv.index('-' + opt)
    sys.argv.pop(idx)
    return sys.argv.pop(idx)

def run_part(args):
    infile, outfile, gpu = args
    start = time()
    sys.argv += ['-gpu', str(gpu), '-src', infile, '-output', outfile]
    translate_main()
    print('GPU {} done after {:.1f}s'.format(gpu, time() - start))


srcfile = popopt('src')
outfile = popopt('output')
gpu_list = popopt('gpus')
gpu_list = [int(gpu) for gpu in gpu_list.split(',')]
inparts  = [tempfile.NamedTemporaryFile('w', encoding='utf8') for _ in gpu_list]
outparts = [tempfile.NamedTemporaryFile('r', encoding='utf8') for _ in gpu_list]
n_gpus = len(gpu_list)


# (1) distribute input lines to `n_gpus` tempfiles
with open(srcfile, 'r', encoding='utf8') as f:
    nlines = len(list(f))
    f.seek(0)
    # round up
    linesperpart = int((nlines + n_gpus - 1) / n_gpus)
    for inp in inparts:
        for line in islice(f, linesperpart):
            inp.write(line)
        inp.flush()

# (2) run `n_gpus` processes translating one tempfile each
with Pool(len(gpu_list)) as p:
    innames = [tf.name for tf in inparts]
    outnames = [tf.name for tf in outparts]
    p.map(run_part, zip(innames, outnames, gpu_list))

# (3) concatenate tempfiles into one output file
with open(outfile, 'w', encoding='utf8') as f:
    for outp in outparts:
        f.write(outp.read())
