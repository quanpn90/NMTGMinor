from nmtg.data.dataset import Dataset
from nmtg.data.text_lookup_dataset import TextLookupDataset


class ParallelDataset(Dataset):
    def __init__(self, src_data: TextLookupDataset, tgt_data: TextLookupDataset = None):
        self.src_data = src_data  # technically a duplicate, but it's fine
        self.tgt_data = tgt_data

    def __getitem__(self, index):
        source = self.src_data[index]
        res = {'id': index, 'src_indices': source, 'src_size': len(source)}

        if self.tgt_data is not None:
            target = self.tgt_data[index]
            target_input = target[:-1]
            target_output = target[1:]
            res['tgt_input'] = target_input
            res['tgt_output'] = target_output
            res['tgt_size'] = len(target_output)

        return res

    def __len__(self):
        return len(self.src_data)

    def collate_samples(self, samples):
        src_batch = self.src_data.collate_samples([x['src_indices'] for x in samples])
        res = {'src_indices': src_batch['indices'], 'src_lengths': src_batch['lengths']}

        if self.tgt_data is not None:
            target_input = self.tgt_data.collate_samples([x['tgt_input'] for x in samples])
            target_output = self.tgt_data.collate_samples([x['tgt_output'] for x in samples])
            res['tgt_input'] = target_input
            res['tgt_output'] = target_output
            res['tgt_size'] = target_output['size']
            res['tgt_lengths'] = target_output['lengths']  # TODO: unsure if needed

        return res
