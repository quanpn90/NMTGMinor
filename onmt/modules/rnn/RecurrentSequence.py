import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np

from onmt.modules.rnn import StackedLSTM, mLSTMCell

class RecurrentSequential(nn.Module):
    """
    Based on implementation of StackedLSTM from openNMT-py
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/StackedRNN.py
    Args:
        embed: instance of torch.nn.Embedding or something with an equivalent __call__ function
        cell: string specifying recurrent cell type ['gru', 'mlstm', 'lstm', 'rnn']. Default: `rnn`
        n_layers: how many of these cells to stack
        in_size: The dimension of the input to the recurrent module (output dimension of embedder)
        rnn_size: The number of features in the hidden states of the lstm cells
        out_size: dimension of linear transformation layer on the output of the stacked rnn cells.
            If <=0 then no output layer is applied. Default: -1
        dropout: probability of dropout layer (applied after rnn, but before output layer). Default: 0
        fused: use fused LSTM kernels if applicable
    Inputs: *inputs, **kwargs
        - **input** (batch, input_size): tensor containing input features
        - **h_0** (num_layers, batch, hidden_size): tensor containing the initial hidden
          state for each layer of each element in the batch.
        - **c_0** (num_layers, batch, hidden_size): tensor containing the initial cell state
          for each layer of each element in the batch.
    Outputs: (h_1, c_1), output
        - **h_1** (num_layers, batch, hidden_size): tensor containing the next hidden state
          for each layer of each element in the batch
        - **c_1** (num_layers, batch, hidden_size): tensor containing the next cell state
          for each layer of each element in the batch
        - **output** (batch, output_size): tensor containing output of stacked rnn. 
            If `output_size==-1` then this is equivalent to `h_1`
    Examples:
        >>> rnn = nn.StackedLSTM(mLSTMCell, 1, 10, 20, 15, 0)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx, cx = hiddens = rnn.state0(3)
        >>> hx.size() # (1,3,20)
        >>> cx.size() # (1,3,20)
        >>> output = []
        >>> for i in range(6):
        ...     hiddens, out = rnn(input[i], hiddens)
        ...     output.append(out)
    """
    def __init__(self, cell, n_layers, in_size, rnn_size, dropout=0.2, fused=True):
        super(RecurrentSequential, self).__init__()
        cell = cell.lower()
        
        # default as mLSTM
        rnn_cell = mLSTMCell
        
        rnn = StackedLSTM(mLSTMCell, n_layers, in_size, rnn_size, dropout=dropout)
        self.add_module('rnn', rnn)
    
    
    def forward(self, input, hidden=None, seq_masks=None, output=None):
        
        """
        Inputs:
            x List: a list of tensor shaped [batch, input_dim] or [1, batch, input_dim]
            hidden: possibly a tuple of tensors of [layers,batch,hidden_dim] shape. Initial hidden state
            masks (optional): [time,batch] shaped tensor of 0s,1s specifying whether to persist state
                on a given timestep or reset it. Used to reset state mid sequence.
            
               
        Returns: out, hidden
            out: stacked tensor of all outputs
            hidden: either last hidden state or stacked tensor of outputs based on value of return_sequence.
                If hidden is tuple of (cell,hidden) state then (cells,hiddens) stacked tensors are returned.
        
        
        """
        outputs = []            
        
        
        for t, input_t in enumerate(input):
            input_t = input_t.squeeze(0)
            if hidden is None:
                hidden = self.rnn.state0(input_t)
                
            # recurrent computation
            # print(hidden[0])
            _output, _hidden = self.rnn(input_t, hidden)
            # 
            cell = _hidden[0] # layers x batch x hidden_dim
            if seq_masks:
                
                dont_mask = seq_masks[t].type_as(cell.data).unsqueeze(2) # 1 x batch_size x 1
                hidden_dont_mask = dont_mask.expand_as(cell)
                
                output_dont_mask = dont_mask.squeeze(0).expand_as(_output)
                
                if output is not None:
                    output = _output *  output_dont_mask + output * ( 1 - output_dont_mask)
                else:
                    output = _output
                    
                hidden = ( _hidden[0]*hidden_dont_mask + hidden[0]*(1 - hidden_dont_mask), 
                           _hidden[1]*hidden_dont_mask + hidden[1]*(1 - hidden_dont_mask) )
                           
                
            else:
                hidden = _hidden
                output = _output
               
            
            outputs += [output]
        
        outputs = torch.stack(outputs) # time * batch * hidden
        
        return outputs, hidden 