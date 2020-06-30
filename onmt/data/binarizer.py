# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
import os
from onmt.utils import safe_readline
# from multiprocessing import Pool
import torch.multiprocessing as mp
import onmt
import numpy as np


class Binarizer:

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def binarize_file_single_thread(filename, tokenizer, vocab, worker_id=0, bos_word=None, eos_word=None,
                                    offset=0, end=-1, data_type='int64', verbose=False):
        """
        This function should read in the lines, convert sentences to tensors
        And then finalize into a dataset?
        """

        result = dict()
        unk_word = onmt.constants.UNK_WORD

        data = list()
        sizes = list()

        count = 0

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)

            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)

            while line:
                if end > 0 and f.tell() > end:
                    break

                tokenized_sent = tokenizer.tokenize(line)

                binarized_line = vocab.convertToIdx(tokenized_sent, unk_word,
                                                    bos_word=bos_word, eos_word=eos_word, type=data_type)

                # move to shared_memory to transfer between threads
                # conversion to numpy is necessary because torch.Tensor seems to be not serializable by the mprocess
                data += [binarized_line.numpy()]
                sizes += [len(tokenized_sent)]

                line = f.readline()

                count += 1
                if count % 100000 == 0:
                    if verbose:
                        print("[INFO] Thread %d processed %d lines." % (worker_id, count))

        if verbose:
            print("[INFO] Thread %d Done." % worker_id)
        result['data'] = data
        result['sizes'] = sizes
        result['id'] = worker_id
        result['total'] = len(sizes)

        return result

    @staticmethod
    def binarize_file(filename, vocab, tokenizer, bos_word=None, eos_word=None,
                      data_type='int64', num_workers=1, verbose=False):

        result = dict()

        for i in range(num_workers):
            result[i] = dict()

        final_result = dict()

        def merge_result(bin_result):
            result[bin_result['id']]['data'] = bin_result['data']
            result[bin_result['id']]['sizes'] = bin_result['sizes']

        offsets = Binarizer.find_offsets(filename, num_workers)

        if num_workers > 1:
            pool = mp.Pool(processes=num_workers)
            mp_results = []

            for worker_id in range(num_workers):
                mp_results.append(pool.apply_async(
                    Binarizer.binarize_file_single_thread,
                    args=(filename, tokenizer, vocab, worker_id, bos_word, eos_word,
                          offsets[worker_id], offsets[worker_id + 1], data_type, verbose),
                    ))

            pool.close()
            pool.join()

            for r in mp_results:
                merge_result(r.get())

        else:
            sp_result = Binarizer.binarize_file_single_thread(filename, tokenizer, vocab, 0, bos_word, eos_word,
                                                              offsets[0], offsets[1], data_type)
            merge_result(sp_result)

        final_result['data'] = list()
        final_result['sizes'] = list()

        # put the data into the list according the worker indices
        for idx in range(num_workers):
            final_result['data'] += result[idx]['data']
            final_result['sizes'] += result[idx]['sizes']

        return final_result
