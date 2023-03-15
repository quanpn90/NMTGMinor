import langid
import torch.multiprocessing as mp
import os
import sys


# this function is borrowed from Facebook
# avoid jumping into the middle of a character
def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


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


def langid_file_single_thread(filename, worker_id=0, offset=0, end=-1):
    # if output_format is scp, we only read the length for sorting

    result = dict()
    data = list()
    probs = list()
    index = 0

    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(offset)

        line = safe_readline(f)

        while line:
            if 0 < end < f.tell():
                break

            class_result = langid.classify(line)
            lang = class_result[0]
            prob = class_result[1]

            data.append(lang)
            probs.append(prob)

            line = f.readline()

            if (index + 1) % 100000 == 0:
                print("[INFO] Thread %d Processed %d lines." % (worker_id, index + 1))

            index = index + 1

    result['lang'] = data
    result['prob'] = probs
    result['id'] = worker_id

    return result

def lang_classify_file(filename, num_workers=1):

    result = dict()

    for i in range(num_workers):
        result[i] = dict()

    final_result = dict()

    def merge_result(bin_result):
        result[bin_result['id']]['lang'] = bin_result['lang']
        result[bin_result['id']]['prob'] = bin_result['prob']

    offsets = find_offsets(filename, num_workers)

    if num_workers > 1:

        pool = mp.Pool(processes=num_workers)
        mp_results = []

        for worker_id in range(num_workers):
            mp_results.append(pool.apply_async(
                langid_file_single_thread,
                args=(filename, worker_id, offsets[worker_id], offsets[worker_id + 1]),
            ))

        pool.close()
        pool.join()

        for r in mp_results:
            merge_result(r.get())

    else:
        sp_result = langid_file_single_thread(filename, 0, offsets[0], offsets[1])

        merge_result(sp_result)

    final_result['lang'] = list()
    final_result['prob'] = list()

    # put the data into the list according the worker indices
    for idx in range(num_workers):
        final_result['lang'] += result[idx]['lang']
        final_result['prob'] += result[idx]['prob']

    return final_result


if __name__ == "__main__":

    import sys

    input_file = sys.argv[1]
    num_workers = int(sys.argv[2])

    print("Start language idenfitication ....")
    final_result = lang_classify_file(input_file, num_workers=num_workers)
    print("Finished.")

    c = 0
    print("Writing the output ....")
    # output_file = open(input_file + ".langid", 'w')
    with open(input_file + ".langid", 'w') as w:

        for (lang, prob) in zip(final_result['lang'], final_result['prob']):

            w.write(lang + " " + str(prob) + "\n")
            c += 1

            if (c + 1) % 100000 == 0:
                print("[INFO] Written %d lines." % c)


