import numpy as np


def indexed_data_iterator(filename, offsets, decode_fn):
    with open(filename, 'rb') as f:
        for i in range(offsets):
            length = offsets[i+1] - offsets[i]
            yield decode_fn(f.read(length))


class IndexedData:
    def __init__(self, filename, offsets, decode_fn):
        self.filename = filename
        self.offsets = offsets
        self.data_file = open(self.filename, 'rb')
        self.decode_fn = decode_fn

    def __del__(self):
        self.data_file.close()

    def __getitem__(self, index):
        self.data_file.seek(self.offsets[index])
        length = self.offsets[index + 1] - self.offsets[index]
        return self.decode_fn(self.data_file.read(length))

    def __len__(self):
        return len(self.offsets) - 1

    def __iter__(self):
        return indexed_data_iterator(self.filename, self.offsets, self.decode_fn)

    @staticmethod
    def save(samples, f, offset_filename, encode_fn):
        offsets = [0]
        for sample in samples:
            f.write(encode_fn(sample))
            offsets.append(f.tell())
        np.save(offset_filename, offsets)
