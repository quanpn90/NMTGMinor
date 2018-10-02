import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, nmt_model,opt):
        super(Autoencoder, self).__init__()

        self.nmt = nmt_model
        self.representation = opt.representation
        if(opt.representation == "EncoderHiddenState"):
            self.inputSize = nmt_model.encoder.model_size
        else:
            raise NotImplementedError("Waring!"+opt.represenation+" not implemented for auto encoder")

        self.hiddenSize = opt.auto_encoder_hidden_size

        layers = []
        if(opt.auto_encoder_drop_out > 0):
            layers.append(nn.Dropout(opt.auto_encoder_drop_out))
        layers.append(nn.Linear(self.inputSize, self.hiddenSize))
        layers.append(nn.Sigmoid())
        if(opt.auto_encoder_drop_out > 0):
            layers.append(nn.Dropout(opt.auto_encoder_drop_out))
        layers.append(nn.Linear(self.hiddenSize, self.inputSize))

        self.model = nn.Sequential(*layers)


    def forward(self,input):

        src = input[0].transpose(0,1)

        if(self.representation == "EncoderHiddenState"):
            context, src_mask = self.nmt.encoder(src,grow=False)
            flattened_context = context.contiguous().view(-1, context.size(-1))
            flattened_mask = src_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
            non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
            clean_context = flattened_context.index_select(0, non_pad_indices)
        else:
            raise NotImplementedError("Waring!"+opt.represenation+" not implemented for auto encoder")

        return clean_context,self.model(clean_context)