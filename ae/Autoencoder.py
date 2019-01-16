import torch
import torch.nn as nn
import onmt


from ae.VariationalLayer import VariationalLayer

class Autoencoder(nn.Module):

    def __init__(self, nmt_model,opt):
        super(Autoencoder, self).__init__()

        self.param_init = opt.param_init
        
        self.nmt = nmt_model
        self.representation = opt.representation
        if(opt.auto_encoder_type is None):
            self.model_type = "Baseline"
        else:
            self.model_type = opt.auto_encoder_type
        if(opt.representation == "EncoderHiddenState"):
            self.inputSize = nmt_model.encoder.model_size
        elif (opt.representation == "DecoderHiddenState"):
            self.inputSize = nmt_model.decoder.model_size
        else:
            raise NotImplementedError("Waring!"+opt.represenation+" not implemented for auto encoder")

        self.hiddenSize = opt.auto_encoder_hidden_size

        layers = []
        if(opt.auto_encoder_drop_out > 0):
            layers.append(nn.Dropout(opt.auto_encoder_drop_out))
        if(self.model_type == "Baseline"):
            layers.append(nn.Linear(self.inputSize, self.hiddenSize))
            layers.append(nn.Sigmoid())
        elif(self.model_type == "Variational"):
            self.variational_layer = VariationalLayer(self.inputSize,self.hiddenSize)
            layers.append(self.variational_layer)
        else:
            raise NotImplementedError("Waring!" + self.model_type + " not implemented for auto encoder")
        if(opt.auto_encoder_drop_out > 0):
            layers.append(nn.Dropout(opt.auto_encoder_drop_out))
        layers.append(nn.Linear(self.hiddenSize, self.inputSize))

        self.model = nn.Sequential(*layers)


    def forward(self,input):

        src = input[0].transpose(0,1)
        tgt = input[1][:-1].transpose(0, 1)

        if(self.representation == "EncoderHiddenState"):
            with torch.no_grad():
                context, src_mask = self.nmt.encoder(src,grow=False)
                flattened_context = context.contiguous().view(-1, context.size(-1))
                flattened_mask = src_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
                non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
                clean_context = flattened_context.index_select(0, non_pad_indices)
        elif(self.representation == "DecoderHiddenState"):
            with torch.no_grad():
                context, src_mask = self.nmt.encoder(src, grow=False)
                output, coverage = self.nmt.decoder(tgt, context, src, grow=False)
                tgt_mask = tgt.data.eq(onmt.Constants.PAD).unsqueeze(1)
                flattened_output = output.contiguous().view(-1, output.size(-1))
                flattened_mask = tgt_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
                non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
                clean_context = flattened_output.index_select(0, non_pad_indices)

        else:
            raise NotImplementedError("Waring!"+opt.represenation+" not implemented for auto encoder")
        
        clean_context.require_grad=False
        return clean_context,self.model(clean_context)


    def init_model_parameters(self):
        for p in self.parameters():
            p.data.uniform_(-self.param_init, self.param_init)
            
