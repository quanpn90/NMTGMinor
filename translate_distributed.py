#!/usr/bin/env python
import sys
import os
import tempfile
from itertools import islice
from time import time
from multiprocessing import Pool

from translate import main as translate_main
from onmt.utils import safe_readline


def find_offsets(filename, num_chunks):
    """
    :param filename: string
    :param num_chunks: int
    :return: a list of offsets (positions to start and stop reading)
    """
    with open(filename, 'r', encoding='utf-8') as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def hasopt(opt):
    return ('-' + opt) in sys.argv


def popopt(opt):
    # TODO handle different option formats, e.g. --opt or -opt=val
    idx = sys.argv.index('-' + opt)
    sys.argv.pop(idx)
    return sys.argv.pop(idx)


def distribute_to_tempfiles(srcfile, n):
    tmpfiles = [tempfile.NamedTemporaryFile('w', encoding='utf8') for _ in range(n)]

    offsets = find_offsets(srcfile, n)
    lines_per_tf = list()

    all_lines = len(open(srcfile).readlines())

    for i, tf in enumerate(tmpfiles):

        n_lines = 0
        start, end = offsets[i], offsets[i + 1]

        with open(srcfile, 'r', encoding='utf8') as f:
            f.seek(start)
            line = safe_readline(f)

            while line:
                if 0 < end < f.tell():
                    break

                tf.write(line)
                n_lines += 1

                line = f.readline()

        tf.flush()

        lines_per_tf.append(n_lines)

    print("Lines per tmp files to be translated: ", lines_per_tf)
    assert (sum(lines_per_tf) == all_lines)

    #     nlines = len(list(f))
    #     f.seek(0)
    #     # round up
    #     linesperpart = int((nlines + n - 1) / n)
    #     for tf in tmpfiles:
    #         for line in islice(f, linesperpart):
    #             tf.write(line)
    #         tf.flush()

    return tmpfiles, lines_per_tf


def distribute_to_tempfiles_withlist(srcfile, n, line_per_tf):
    tmpfiles = [tempfile.NamedTemporaryFile('w', encoding='utf8') for _ in range(n)]
    assert len(line_per_tf) == n

    with open(srcfile) as f:
        for i, tf in enumerate(tmpfiles):
            nlines = line_per_tf[i]
            for _ in range(nlines):
                line = f.readline()
                tf.write(line)
            tf.flush()

    return tmpfiles


def run_part(args):
    infile, goldfile, subsrcfile, outfile, gpu = args
    start = time()
    sys.argv += ['-gpu', gpu, '-src', infile, '-output', outfile]
    if goldfile:
        sys.argv += ['-tgt', goldfile]
    if subsrcfile:
        sys.argv += ['-sub_src', subsrcfile]

    translate_main()
    print('GPU {} done after {:.1f}s'.format(gpu, time() - start))


srcfile = popopt('src')
outfile = popopt('output')
gpu_list = popopt('gpus').split(',')

# (1) distribute input lines to N tempfiles
inparts, lines_per_file = distribute_to_tempfiles(srcfile, len(gpu_list))
if hasopt('tgt'):
    goldfile = popopt('tgt')
    goldparts = distribute_to_tempfiles_withlist(goldfile, len(gpu_list), lines_per_file)
else:
    goldparts = [None for _ in range(len(gpu_list))]

if hasopt('sub_src'):
    sub_src_file = popopt('sub_src')
    sub_src_parts = distribute_to_tempfiles_withlist(sub_src_file, len(gpu_list), lines_per_file)
else:
    sub_src_parts = [None for _ in range(len(gpu_list))]

# (2) run N processes translating one tempfile each
outparts = [tempfile.NamedTemporaryFile('r', encoding='utf8') for _ in gpu_list]
filenames = lambda tmpfiles: [tf.name if tf else None for tf in tmpfiles]
with Pool(len(gpu_list)) as p:
    p.map(run_part, zip(filenames(inparts),
                        filenames(goldparts),
                        filenames(sub_src_parts),
                        filenames(outparts),
                        gpu_list))

# (3) concatenate tempfiles into one output file
with open(outfile, 'w', encoding='utf8') as f:
    for outp in outparts:
        f.write(outp.read())
