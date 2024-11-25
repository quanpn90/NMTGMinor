# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
import os
from onmt.utils import safe_readline, safe_readaudio
# from multiprocessing import Pool
import torch.multiprocessing as mp
import torch
import onmt
import numpy as np
from .audio_utils import ArkLoader, safe_readaudio, wav_to_fmel
# from .whisper_audio import log_mel_spectrogram
import torchaudio


class SpeechBinarizer:


    @staticmethod
    def binarize_file_single_thread(filename, ark_loader, offset=0, end=-1, worker_id=0,
                                    input_format='scp', output_format='raw',
                                    prev_context=0, concat=4, stride=1, fp16=False, sample_rate=16000,
                                    verbose=False, num_mel_bin=0):

        audio_processor = None

        # if output_format is scp, we only read the length for sorting
        if output_format == 'scp':
            raise NotImplementedError
            # assert input_format in ['kaldi', 'scp']
        elif output_format == 'wav':
            input_format = 'wav'
        elif 'whisper' in output_format:

            input_format = output_format
            from .whisper_audio import WhisperAudioProcessor
            audio_processor = WhisperAudioProcessor(output_format)
            # TODO: get the whisper
            pass
        else:
            print("[ERROR] Unknown output format: {}".format(output_format))
            raise NotImplementedError

        # placeholder to store the data
        result = dict()
        data = list()
        lengths = list()
        index = 0

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)

            line = safe_readline(f)

            while line:
                if 0 < end < f.tell():
                    break

                parts = line.split()

                key = parts[0]

                # this special case is for the "preceeding"
                if key == 'NULL':
                    feature_vector = torch.zeros(0, 0)
                    lengths.append(feature_vector.size(0))
                    line = f.readline()
                    continue

                if input_format in ['scp', 'kaldi']:
                    # # an scp file has the format: uttid path:mem
                    # path = parts[1]
                    # # read numpy array from the ark here
                    # feature_vector = ark_loader.load_mat(path)
                    #
                    # if stride == 1:
                    #     feature_vector = torch.from_numpy(feature_vector)
                    # else:
                    #     feature_vector = torch.from_numpy(feature_vector[0::stride])
                    #
                    # if concat > 1:
                    #     add = (concat - feature_vector.size()[0] % concat) % concat
                    #     z = torch.FloatTensor(add, feature_vector.size()[1]).zero_()
                    #     feature_vector = torch.cat((feature_vector, z), 0)
                    #     feature_vector = feature_vector.reshape((int(feature_vector.size()[0] / concat),
                    #                                              feature_vector.size()[1] * concat))
                    #
                    # if prev_context > 0:
                    #     print("Multiple ASR context isn't supported at the moment   ")
                    #     raise NotImplementedError
                    #
                    # if fp16 and output_format not in ['scp', 'scpmem']:
                    #     feature_vector = feature_vector.half()
                    #
                    # if output_format not in ['scp', 'scpmem']:
                    #     data.append(feature_vector.numpy())  # convert to numpy for serialization
                    # else:
                    #     data.append(path)

                    raise NotImplementedError

                elif input_format == 'wav':

                    # an wav input file should have format uttid wav_file start end
                    # in which the start and end (by second) can be 0 0

                    if len(parts) >= 4:
                        wavpath, start_time, end_time = parts[1], float(parts[2]), float(parts[3])
                    else:
                        wavpath = parts[1]
                        start_time = 0
                        end_time = -1

                    if wavpath.endswith("wav"):
                        # if using wav we can use a cache loader to avoid loading big wavfiles too many times
                        feature_vector = ark_loader.load_wav(wavpath, start_time, end_time, sample_rate=sample_rate)
                    else:
                        feature_vector = safe_readaudio(wavpath, start_time, end_time, sample_rate=sample_rate)

                    # if we also extract the fmel then we can also do it here
                    if num_mel_bin > 0:
                        feature_vector = wav_to_fmel(feature_vector, num_mel_bin=num_mel_bin)

                    # store a tuple of data and information to load the wav again during training
                    data.append((wavpath, start_time, end_time, sample_rate))

                elif input_format == "whisper":
                    if len(parts) >= 4:
                        wavpath, start_time, end_time = parts[1], float(parts[2]), float(parts[3])
                    else:
                        wavpath = parts[1]
                        start_time = 0
                        end_time = -1


                    # # an wav input file should have format uttid wav_file start end
                    # # in which the start and end (by second) can be 0 0
                    #
                    # if len(parts) >= 4:
                    #     wavpath, start_time, end_time = parts[1], float(parts[2]), float(parts[3])
                    # else:
                    #     wavpath = parts[1]
                    #     start_time = 0
                    #     end_time = -1

                length = feature_vector.size(0)
                lengths.append(length)

                line = f.readline()

                if (index + 1) % 100000 == 0:
                    print("[INFO] Thread %d Processed %d audio utterances." % (worker_id, index + 1))

                index = index + 1

        result['data'] = data
        result['sizes'] = lengths
        result['id'] = worker_id
        result['total'] = len(lengths)

        return result

    @staticmethod
    def binarize_file(filename, input_format='scp', output_format='raw', num_mel_bin=0,
                      prev_context=0, concat=4, stride=1, fp16=False, num_workers=1, verbose=False):

        result = dict()

        for i in range(num_workers):
            result[i] = dict()

        final_result = dict()

        def merge_result(bin_result):
            result[bin_result['id']]['data'] = bin_result['data']
            result[bin_result['id']]['sizes'] = bin_result['sizes']

        offsets = Binarizer.find_offsets(filename, num_workers)

        ark_loaders = dict()
        for i in range(num_workers):
            if input_format in ['scp', 'kaldi']:
                ark_loaders[i] = ArkLoader()
            elif input_format in ['wav']:
                from .audio_utils import WavLoader
                ark_loaders[i] = WavLoader()
            else:
                ark_loaders[i] = None

        if num_workers > 1:

            pool = mp.Pool(processes=num_workers)
            mp_results = []

            for worker_id in range(num_workers):
                mp_results.append(pool.apply_async(
                    SpeechBinarizer.binarize_file_single_thread,
                    args=(filename, ark_loaders[worker_id], offsets[worker_id], offsets[worker_id + 1], worker_id,
                          input_format, output_format, prev_context, concat, stride, fp16, 16000, verbose, num_mel_bin),
                ))

            pool.close()
            pool.join()

            for r in mp_results:
                merge_result(r.get())

        else:
            sp_result = SpeechBinarizer.binarize_file_single_thread(filename, ark_loaders[0], offsets[0], offsets[1], 0,
                                                                    input_format='scp', output_format=output_format,
                                                                    prev_context=prev_context, concat=concat,
                                                                    stride=stride, fp16=fp16, verbose=verbose, num_mel_bin=num_mel_bin)
            merge_result(sp_result)

        final_result['data'] = list()
        final_result['sizes'] = list()

        # put the data into the list according the worker indices
        for idx in range(num_workers):

            for j in range(len(result[idx]['data'])):
                x = result[idx]['data'][j]

                # if we store the numpy array, then convert to torch
                # otherwise, x is the scp path to the matrix
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                final_result['data'].append(x)
            final_result['sizes'] += result[idx]['sizes']

        # remember to close the workers when its done
        for i in range(num_workers):
            if ark_loaders[i] is not None:
                ark_loaders[i].close()

        return final_result


class Binarizer:

    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def binarize_file_single_thread(filename, tokenizer, vocab, worker_id=0, bos_word=None, eos_word=None,
                                    offset=0, end=-1, data_type='int64', verbose=False,
                                    external_tokenizer=[None, None], lang=None, target=False):
        """
        This function should read in the lines, convert sentences to tensors
        And then finalize into a dataset?
        """

        result = dict()
        unk_word = onmt.constants.UNK_WORD
        data = list()
        sizes = list()

        count = 0
        ext_tokenizer, external_tokenizer_name = external_tokenizer

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)

            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            n_bad_sentences = 0

            while line:
                if 0 < end < f.tell():
                    break

                if ext_tokenizer is None:
                    tokenized_sent = tokenizer.tokenize(line)

                    binarized_line = vocab.convertToIdx(tokenized_sent, unk_word,
                                                        bos_word=bos_word, eos_word=eos_word, type=data_type)

                    # move to shared_memory to transfer between threads
                    # conversion to numpy is necessary because torch.Tensor is not serializable by the mprocess
                    data += [binarized_line.numpy()]
                    sizes += [len(tokenized_sent)]

                else:
                    tensor = ext_tokenizer(line.strip())['input_ids']
                    # print(tensor)
                    # assert that the mbart50 tokenizer uses the correct language ID
                    if "mbart-large-50" in external_tokenizer_name.lower():
                        assert tensor[0] == vocab.convertToIdx([lang], None)[0], "The first token must be language ID"
                        pad_id = vocab.convertToIdx(["<pad>"], None)[0]
                        assert pad_id not in tensor, "Pad is not supposed to appear in the tensors."

                    elif "mbart50pre" in external_tokenizer_name.lower():
                        assert tensor[0] == vocab.convertToIdx([lang], None)[0], "The first token must be language ID"
                        pad_id = vocab.convertToIdx(["<pad>"], None)[0]
                        assert pad_id not in tensor, "Pad is not supposed to appear in the tensors."

                        bos_id = vocab.convertToIdx(["<s>"], None)[0].item()

                        tensor = [bos_id] + tensor

                        if pad_id in tensor:
                            print("[WARNING] Pad is not supposed to appear in the tensors.")
                        # assert pad_id not in tensor, "Pad is not supposed to appear in the tensors."

                    elif "m2m" in external_tokenizer_name.lower():
                        lang_token = "__" + lang + "__"
                        assert tensor[0] == vocab.convertToIdx([lang_token], None)[0], \
                            "The first token must be language ID"
                        pad_id = vocab.convertToIdx(["<pad>"], None)[0]
                        assert pad_id not in tensor, "Pad is not supposed to appear in the tensors."
                    elif "deltalm" in external_tokenizer_name.lower():
                        if len(tensor) > 2:
                            if tensor[0] not in [0, 1, 2, 3]:
                                assert tensor[0] == vocab.convertToIdx([lang], None)[0], "The first token must be language ID"

                        pad_id = vocab.convertToIdx(["<pad>"], None)[0]
                        assert pad_id not in tensor, "Pad is not supposed to appear in the tensors."

                        if target and tensor[0] != tensor[-1]:
                            # for the target side and in the multilingual case it is <eos> <langid> X <eos>
                            tensor = [tensor[-1]] + tensor

                    elif "mbart50eu" in external_tokenizer_name.lower():
                        if len(tensor) > 2:
                            if tensor[0] not in [0, 1, 2, 3]:
                                _lang = _lang if lang != "eu" else "en_XX"
                                assert tensor[0] == vocab.convertToIdx([lang], None)[0], \
                                    "The first token must be language ID, expecting %d get %d. Current language: %s" \
                                    % (vocab.convertToIdx([lang], None)[0], tensor[0], ext_tokenizer.src_lang)

                        # pad_id = vocab.convertToIdx(["<pad>"], None)[0]
                        # assert pad_id not in tensor, "Pad is not supposed to appear in the tensors."

                    if len(tensor) <= 2:
                        n_bad_sentences += 1
                        # print("[Warning] empty sentence with %d tokens including <bos> <eos>" % len(tensor))
                    sizes += [len(tensor)]

                    _dtype = np.int32
                    if data_type == "int64":
                        _dtype = np.int64
                    elif data_type == "int16":
                        _dtype = np.int16

                    data += [np.asarray(tensor, dtype=_dtype)]

                line = f.readline()

                count += 1
                if count % 100000 == 0:
                    if verbose:
                        print("[INFO] Thread %d processed %d lines." % (worker_id, count))

        if verbose:
            if n_bad_sentences > 0:
                print("[Warning] %d empty sentence including <bos> <eos>" % n_bad_sentences)
            print("[INFO] Thread %d Done." % worker_id)
        result['data'] = data
        result['sizes'] = sizes
        result['id'] = worker_id
        result['total'] = len(sizes)

        return result

    @staticmethod
    def binarize_file(filename, vocab, tokenizer, bos_word=None, eos_word=None,
                      data_type='int64', num_workers=1, verbose=False, external_tokenizer="",
                      lang=None, lang_list=[], target=False):

        if "mbart50cluster" in external_tokenizer.lower():

            print("Loading a mbart50 tokenizer yet extended for housing clusters")
            from pretrain_module.tokenization_mbart50_clusters import MBart50ClusterTokenizer
            vocab_file = "sentencepiece/sentencepiece.bpe.model"

            tokenizer = MBart50ClusterTokenizer(vocab_file=vocab_file,
                                                src_lang=None,
                                                tgt_lang=None,
                                                tokenizer_file=None,
                                                eos_token="</s>",
                                                sep_token="</s>",
                                                cls_token="<s>",
                                                unk_token="<unk>",
                                                pad_token="<pad>",
                                                mask_token="<mask>", )

            ext_tokenizer = tokenizer
            src_lang = "<s>" if lang is None else lang
            ext_tokenizer.src_lang = src_lang

        elif "mbart-large-50" in external_tokenizer.lower():

            print("[INFO] Using the external %s tokenizer..." % external_tokenizer)

            from transformers import MBart50TokenizerFast
            try:  # check if this tokenizer is saved locally or not
                print("Looking for pre-downloaded tokenizer ...")
                ext_tokenizer = torch.load("mbart-large-50.tokenizer.pt")
                ext_tokenizer.src_lang = lang
                if ext_tokenizer.src_lang != lang:
                    raise RuntimeError("The language %s does not exist in mBART50." % lang)
            except FileNotFoundError as e:
                print("Expected error: ", e, "Downloading tokenizer ...")
                ext_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
                ext_tokenizer.src_lang = lang
                # ext_tokenizer.src_lang = lang
                if ext_tokenizer.src_lang != lang:
                    raise RuntimeError("The language %s does not exist in mBART50." % lang)
                torch.save(ext_tokenizer, "mbart-large-50.tokenizer.pt")
        elif "m2m100" in external_tokenizer.lower():

            print("[INFO] Using the external %s tokenizer..." % external_tokenizer)

            from transformers import M2M100Tokenizer
            ext_tokenizer = M2M100Tokenizer.from_pretrained(external_tokenizer, src_lang=lang)
            ext_tokenizer.src_lang = lang
            if ext_tokenizer.src_lang != lang:
                raise RuntimeError("The language %s does not exist in M2M100." % lang)

        elif "mbart50pre" in external_tokenizer.lower():

            from transformers import MBart50TokenizerFast
            try:  # check if this tokenizer is saved locally or not
                print("Looking for pre-downloaded tokenizer ...")
                ext_tokenizer = torch.load("mbart-large-50.tokenizer.pt")
                ext_tokenizer.src_lang = lang
                if ext_tokenizer.src_lang != lang:
                    raise RuntimeError("The language %s does not exist in mBART50." % lang)
            except FileNotFoundError as e:
                print("Expected error: ", e, "Downloading tokenizer ...")
                ext_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
                ext_tokenizer.src_lang = lang
                # ext_tokenizer.src_lang = lang
                if ext_tokenizer.src_lang != lang:
                    raise RuntimeError("The language %s does not exist in mBART50." % lang)
                torch.save(ext_tokenizer, "mbart-large-50.tokenizer.pt")


        elif "mbart50eu" in external_tokenizer.lower():

            print("[INFO] Using the MBART50EU tokenizer...")
            from transformers import MBart50TokenizerFast
            # from pretrain_module.tokenization_mbart50eu import MBART50TokenizerEU
            # src_lang = lang if lang != "eu" else "en_XX"
            src_lang = "<s>"
            ext_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
            ext_tokenizer.src_lang = src_lang

        elif "bart" in external_tokenizer.lower():

            print("[INFO] Using the external BART tokenizer...")

            from transformers import BartTokenizer
            ext_tokenizer = BartTokenizer.from_pretrained(external_tokenizer)

        elif "deltalm" in external_tokenizer.lower():

            print("[INFO] Using the DeltaLM tokenizer...")
            from pretrain_module.tokenization_deltalm import MultilingualDeltaLMTokenizer
            ext_tokenizer = MultilingualDeltaLMTokenizer.from_pretrained("facebook/mbart-large-50", lang_list=lang_list,
                                                                         src_lang=lang)

            # from pretrain_module.tokenization_deltalm import DeltaLMTokenizer
            # try:  # check if this tokenizer is saved locally or not
            #     ext_tokenizer = torch.load("deltalm.tokenizer.pt")
            #     ext_tokenizer.src_lang = lang
            # except FileNotFoundError:
            #     ext_tokenizer = DeltaLMTokenizer.from_pretrained("facebook/mbart-large-50", src_lang=lang)

        elif "nllb" in external_tokenizer.lower():

            from transformers import NllbTokenizer
            from pretrain_module.tokenization_deltalm import DeltaLMTokenizer
            try:  # check if this tokenizer is saved locally or not
                ext_tokenizer = torch.load("nllb.tokenizer.pt")
                ext_tokenizer.src_lang = lang
            except FileNotFoundError:
                ext_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=lang)
                torch.save(ext_tokenizer, "nllb.tokenizer.pt")

        elif external_tokenizer is None or len(external_tokenizer) == 0:
            ext_tokenizer = None
        else:
            raise NotImplementedError

        ext_tokenizer = [ext_tokenizer, external_tokenizer]

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
                          offsets[worker_id], offsets[worker_id + 1], data_type, verbose, ext_tokenizer, lang, target),
                ))

            pool.close()
            pool.join()

            for r in mp_results:
                merge_result(r.get())

        else:
            sp_result = Binarizer.binarize_file_single_thread(filename, tokenizer, vocab, 0, bos_word, eos_word,
                                                              offsets[0], offsets[1], data_type,
                                                              external_tokenizer=ext_tokenizer,
                                                              lang=lang, target=target)
            merge_result(sp_result)

        final_result['data'] = list()
        final_result['sizes'] = list()

        # put the data into the list according the worker indices
        for idx in range(num_workers):
            final_result['data'] += result[idx]['data']
            final_result['sizes'] += result[idx]['sizes']

        return final_result