import os
import sys
import torch.multiprocessing as mp

def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins

class Normalizer:

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
    def normalize_file_single_thread(filename, normalizer, worker_id=0,
                                    offset=0, end=-1):
        result = dict()
        data = list()
        data_org = list()

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)

            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            count = 0

            while line:
                if 0 < end < f.tell():
                    break

                _line = line.strip()
                outline = normalizer(_line)
                data.append(outline)
                data_org.append(_line)

                line = f.readline()
                count += 1

        result['data'] = data
        result['data_org'] = data_org
        result['id'] = worker_id
        result['total'] = data

        return result

    @staticmethod
    def normalize_file(filename, normalizer, num_workers=1):

        result = dict()

        for i in range(num_workers):
            result[i] = dict()

        def merge_result(bin_result):
            result[bin_result['id']]['data'] = bin_result['data']
            result[bin_result['id']]['data_org'] = bin_result['data_org']

        offsets = Normalizer.find_offsets(filename, num_workers)

        if num_workers > 1:

            pool = mp.Pool(processes=num_workers)
            mp_results = []

            for worker_id in range(num_workers):
                mp_results.append(pool.apply_async(
                    Normalizer.normalize_file_single_thread,
                    args=(filename, normalizer, worker_id,
                          offsets[worker_id], offsets[worker_id + 1]),
                ))

            pool.close()
            pool.join()

            for r in mp_results:
                merge_result(r.get())

        else:
            sp_result = Normalizer.normalize_file_single_thread(filename, normalizer, 0,
                                                              offsets[0], offsets[1])

            merge_result(sp_result)

        final_result = list()
        org_data = list()

        # put the data into the list according the worker indices
        for idx in range(num_workers):
            final_result += result[idx]['data']
            org_data += result[idx]['data_org']

        return org_data, final_result

if __name__ == '__main__':

    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer

    input_file = sys.argv[1]
    lang = sys.argv[2]
    num_workers = int(sys.argv[3])
    cleaning = 0 if len(sys.argv) == 4 else int(sys.argv[4])

    if lang == "en":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    print("Normalizing file {}".format(input_file))

    # problem: still need to write data into RAM
    org_data, normalized_data = Normalizer.normalize_file(input_file, normalizer, num_workers=num_workers)

    if cleaning != 0:
        output_file = input_file + ".norm.clean"
        writer = open(output_file, 'w')
        org_writer = open(input_file + ".clean", "w")

        print("Done. Now cleaning and writing to {}".format(output_file))

        for org_line, line in zip(org_data, normalized_data):

            org_parts = org_line.split()
            parts = line.split()

            if len(parts) <= 0.7 * len(org_parts):
                continue

            org_writer.write(org_line + "\n")
            writer.write(line + "\n")

        org_writer.close()
    else:
        output_file = input_file + ".norm"
        writer = open(output_file, 'w')

        print("Done. Now writing to {}".format(output_file))

        for line in normalized_data:

            parts = line.split()

            writer.write(line + "\n")


    writer.close()
    print("Finished.")
