import torch
import torch.nn as nn
import torch.nn.functional as F

class 3DBottle(nn.Module):
    
        def __init__(self, m):
    
            super(3DBottle, self).__init__()
            
            self.module = m() # we initialize the module using lambda 
            
        def forward(self, input):
            
            # input should be a 3D variable
            # B x L x H
            B = input.size(0)
            L = input.size(1)
            
            resized2D = input.view(B * L, -1)
            
            output = self.module(resized2D)
            
            resizedOut = output.contiguous().view(B, L, -1)
            
            return resizedOut


class AttributeEmbeddings(nn.Module):

    def __init__(self, atb_dicts, n_attributes, atb_size):

        self.n_attributes = n_attributes
        self.atb_sizes = atb_size
        super().__init__(self)

        self.atb_embeddings = nn.ModuleDict()

        for i in atb_dicts:

            self.atb_embeddings[i] = nn.Embeddings(len(atb_dicts[i]), atb_size)

    def forward(self, atbs):
    """
    Input: atbs is a dictionary of features
    """

    embeddings = {}

    for i in atbs:

        embedding = self.atb_embeddings[i](atbs[i])

        embeddings.append(embedding)

    # Concatenation of the features
    embedding = torch.cat(embeddings, dim=-1)

    return embedding