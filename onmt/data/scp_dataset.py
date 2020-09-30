import torch
from kaldiio import load_mat
from functools import lru_cache
import numpy as np
from .audio_utils import _parse_arkpath, ArkLoader


class SCPIndexDataset(torch.utils.data.Dataset):
    """
    This dataset simply stores a list of paths to ark matrices
    The __get__ function uses load_mat from kaldiio to read the ark matrices for retrieval
    """

    def __init__(self, scp_path_list, concat=4):
        """
        :param scp_path_list: list of path to the ark matrices
        """
        self.scp_path_list = scp_path_list
        self._sizes = len(self.scp_path_list)
        self._dtype = torch.float32
        self.concat = concat
        self.reader = ArkLoader()

    @property
    def dtype(self):
        # I'm not sure when this function is called
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    def __len__(self):
        return self._sizes

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        scp_path = self.scp_path_list[i]
        mat = self.reader.load_mat(scp_path)

        feature_vector = torch.from_numpy(mat)
        concat = self.concat

        if concat > 1:
            add = (concat - feature_vector.size()[0] % concat) % concat
            z = torch.FloatTensor(add, feature_vector.size()[1]).zero_()
            feature_vector = torch.cat((feature_vector, z), 0)
            feature_vector = feature_vector.reshape((int(feature_vector.size()[0] / concat),
                                                     feature_vector.size()[1] * concat))

        return feature_vector

    @property
    def sizes(self):
        return self._index.sizes