import torch
import torch.nn as nn
import onmt
import torch.nn.functional as F


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
        elif (opt.representation == "EncoderDecoderHiddenState"):
            self.inputSize = nmt_model.encoder.model_size
        elif (opt.representation == "Probabilities"):
            if(type(nmt_model.generator) is nn.ModuleList):
                self.inputSize = nmt_model.generator[0].output_size
            else:
                self.inputSize = nmt_model.generator.output_size
        else:
            raise NotImplementedError("Waring!"+opt.represenation+" not implemented for auto encoder")

        self.outputSize = self.inputSize
        if (opt.representation == "EncoderDecoderHiddenState"):
            self.outputSize = self.inputSize = nmt_model.decoder.model_size
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

#        if(opt.auto_encoder_drop_out > 0):
#            layers.append(nn.Dropout(opt.auto_encoder_drop_out,inplace=True))


        layers.append(nn.Linear(self.hiddenSize, self.outputSize))

        self.model = nn.Sequential(*layers)

        self.layers = layers
        print("Autoencoder:",self.model)

    def forward(self,batch):

        src = batch.get('source')
        tgt = batch.get('target_input')

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)


        if(self.representation == "EncoderHiddenState"):
            with torch.no_grad():
                encoder_output = self.nmt.encoder(src)
                context = encoder_output['context']
                src_mask = encoder_output['src_mask']

                flattened_context = context.contiguous().view(-1, context.size(-1))
                flattened_mask = src_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
                non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
                clean_context = flattened_context.index_select(0, non_pad_indices)
                clean_output = clean_context
        elif(self.representation == "DecoderHiddenState"):
            with torch.no_grad():

                encoder_output = self.nmt.encoder(src)
                context = encoder_output['context']

                decoder_output = self.nmt.decoder(tgt, context, src)
                output = decoder_output['hidden']

                tgt_mask = tgt.data.eq(onmt.Constants.PAD).unsqueeze(1)
                tgt_mask2 = tgt.data.eq(onmt.Constants.EOS).unsqueeze(1)
                tgt_mask = tgt_mask + tgt_mask2
                flattened_output = output.contiguous().view(-1, output.size(-1))
                flattened_mask = tgt_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
                non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
                clean_context = flattened_output.index_select(0, non_pad_indices)
                clean_output = clean_context
        elif (self.representation == "EncoderDecoderHiddenState"):
            with torch.no_grad():
                encoder_output = self.nmt.encoder(src)
                context = encoder_output['context']

                decoder_output = self.nmt.decoder(tgt, context, src)
                output = decoder_output['hidden']

                clean_output = output.clone()
                # predict sum of target for all inputs
                #tgt_mask = tgt.data.eq(onmt.Constants.PAD).unsqueeze(1)
                #tgt_mask2 = tgt.data.eq(onmt.Constants.EOS).unsqueeze(1)
                #tgt_mask = (tgt_mask + tgt_mask2).squeeze(1).transpose(0,1)
                #output.masked_fill_(tgt_mask.unsqueeze(2).expand(-1,-1,output.size(2)),0)
                #output = output.sum(0).unsqueeze(0).expand(context.size(0),-1,-1)

                #select encoder states
                src_mask = encoder_output['src_mask']

                #flattened_context = context.contiguous().view(-1, context.size(-1))
                #flattened_mask = src_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
                ##non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
                #clean_context = flattened_context.index_select(0, non_pad_indices)
                #clean_output = output.contiguous().view(-1, output.size(-1)).index_select(0,non_pad_indices)
                clean_context = context.contiguous().view(-1, context.size(-1)).clone()

        elif(self.representation == "Probabilities"):
            with torch.no_grad():

                encoder_output = self.nmt.encoder(src)
                context = encoder_output['context']

                decoder_output = self.nmt.decoder(tgt, context, src)
                output = decoder_output['hidden']


                tgt_mask = tgt.data.eq(onmt.Constants.PAD).unsqueeze(1)
                tgt_mask2 = tgt.data.eq(onmt.Constants.EOS).unsqueeze(1)
                tgt_mask = tgt_mask + tgt_mask2
                flattened_output = output.contiguous().view(-1, output.size(-1))
                flattened_mask = tgt_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
                non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
                clean_context = flattened_output.index_select(0, non_pad_indices)
                if (type(self.nmt.generator) is nn.ModuleList):
                    clean_context = self.nmt.generator[0](clean_context)
                else:
                    clean_context = self.nmt.generator(clean_context)
                clean_output = clean_context


        else:
            raise NotImplementedError("Waring!"+opt.represenation+" not implemented for auto encoder")
        
        # clean_context.require_grad=False
        clean_context.detach_()
        clean_output.detach_()
        
        #result = self.model(clean_context)

        result = clean_context

        for i in range(len(self.layers)):
            result = self.layers[i](result)

        if (self.representation == "Probabilities"):
            result = F.log_softmax(result, dim=-1)

        if (self.representation == "EncoderDecoderHiddenState"):
            result = result.view(src_mask.size(-1),src_mask.size(0),-1)

            expand_result = result.unsqueeze(1).expand(-1,clean_output.size(0),-1,-1)
            clean_output = clean_output.unsqueeze(0).expand(result.size(0),-1, -1, -1)
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            sim = cos(expand_result,clean_output)

            tgt_mask = tgt.data.eq(onmt.Constants.PAD).unsqueeze(1)
            tgt_mask2 = tgt.data.eq(onmt.Constants.EOS).unsqueeze(1)
            tgt_mask = (tgt_mask + tgt_mask2).squeeze(1).transpose(0,1).unsqueeze(0).expand(result.size(0),-1,-1)
            src_mask_align = src_mask.transpose(0,2).expand(-1,tgt_mask.size(1),-1)
            mask = torch.max(src_mask_align,tgt_mask)

            sim.masked_fill_(mask,-float('inf'))
            alignment = F.softmax(sim,dim=0).masked_fill_(mask,0)

            clean_output = (alignment.unsqueeze(-1).expand(-1,-1,-1,result.size(-1)) * clean_output).sum(1)

            flattened_result = result.contiguous().view(-1, result.size(-1))
            flattened_output = clean_output.contiguous().view(-1, clean_output.size(-1))
            flattened_mask = src_mask.squeeze(1).transpose(0,1).contiguous().view(-1)
            non_pad_indices = torch.nonzero(1-flattened_mask).squeeze(1)
            result = flattened_result.index_select(0, non_pad_indices)
            clean_output = flattened_output.index_select(0, non_pad_indices)
        return clean_output,result

    def calcAlignment(self, batch):

        src = batch.get('source')
        tgt = batch.get('target_input')

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        if (self.representation == "EncoderDecoderHiddenState"):
            with torch.no_grad():
                encoder_output = self.nmt.encoder(src)
                context = encoder_output['context']
                decoder_output = self.nmt.decoder(tgt, context, src)
                output = decoder_output['hidden']

                flat_context = context.contiguous().view(-1, context.size(-1))

        else:
            raise NotImplementedError("Waring!" + opt.represenation + " not implemented for auto encoder")

        # clean_context.require_grad=False
        flat_context = flat_context.detach()
        output = output.detach()

        # result = self.model(clean_context)

        result = flat_context

        for i in range(len(self.layers)):
            result = self.layers[i](result)

        if (self.representation == "Probabilities"):
            result = F.log_softmax(result, dim=-1)

        return output, result.view(context.size(0),context.size(1),-1)

    def autocode(self,input):

        result = input.view(-1,input.size(2))
        for i in range(len(self.layers)):
            result = self.layers[i](result)
        return result.view(input.size())


    def init_model_parameters(self):
        for p in self.parameters():
            p.data.uniform_(-self.param_init, self.param_init)
            

    def parameters(self):
        param = []
        for (n,p) in self.named_parameters():
            if('nmt' not in n):
                param.append(p)
        return param
        

    def load_state_dict(self, state_dict, strict=True):
        
        def condition(param_name):
            
            if 'positional_encoder' in param_name:
                return False
            if 'time_transformer' in param_name and self.nmt.encoder.time == 'positional_encoding':
                return False
            if param_name == 'nmt.decoder.mask':
                return False
#            if 'nmt' in param_name:
#                return False
            
            return True
        
        if("nmt.generator.linear.weight" in state_dict and type(self.nmt.generator) is nn.ModuleList):
            self.nmt.generator = self.nmt.generator[0]

        filtered = {k: v for k, v in state_dict.items() if condition(k)}
        
        model_dict = self.state_dict()
        for k,v in model_dict.items():
            if k not in filtered:
                filtered[k] = v
        super().load_state_dict(filtered)   

        if(type(self.nmt.generator) is not nn.ModuleList):
            self.nmt.generator = nn.ModuleList([self.nmt.generator])
