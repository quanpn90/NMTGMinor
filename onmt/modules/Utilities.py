import torch
import torch.nn as nn

class AttributeEmbeddings(nn.Module):

    def __init__(self, atb_dicts, atb_size):

        self.n_attributes = len(atb_dicts)
        self.atb_sizes = atb_size
        super().__init__()

        self.atb_embeddings = nn.ModuleDict()

        for i in atb_dicts:

            self.atb_embeddings[str(i)] = nn.Embedding(atb_dicts[i].size(), atb_size)

    def forward(self, atbs):
        """
        Input: atbs is a dictionary of features
        """

        embeddings = []

        for i in atbs:

            embedding = self.atb_embeddings[str(i)](atbs[i])

            embeddings.append(embedding)

        # Concatenation of the features
        embedding = torch.cat(embeddings, dim=-1)

        return embedding

    def size(self):

        return len(self.atb_embeddings)