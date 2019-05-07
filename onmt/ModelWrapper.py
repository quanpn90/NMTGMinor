from collections import defaultdict

import numpy as np

from torch.nn.parallel import DataParallel, DistributedDataParallel

class ModelWrapper(object):
    """
    Arguments:
        module (nn.Module): module we want to provide wrapper functionality for
    Attributes:
        using_cuda: whether module is on gpu or cpu
        loss_fn: loss function to use to optimize module. Module should perform loss computation
            within call to __forward__
        loss: current loss value from the module's optimization
        agg_func: function to aggregate loss values. Used when module is DataParallel, to aggregate
            the loss values from each of the replicas
        gn: gradient norm of all module parameters resulting from optimizing loss. calculated by
            sum([gn(p) for p in module.parameters()])
        is_training: whether module is in training mode or not.
        init_state: initial state to use for the module if it is stateful
        distributed: whether module is using DistributedDataParallel
        
    """
def __init__(self, model, opt, loss_function=None):
    super(ModelWrapper, self).__init__()
    self.model = model
    self.using_cuda = False
    self._module = self.module
    self.data_parallel = False
    self.distributed = False
    
    self.loss_function = loss_function

def cuda(self):
    
    self.model = self.model.cuda()
    
    if self.loss_function is not None:
        self.loss_function.cuda()
    
        

def toMultiGPU(self):
    
    pass
    

def save_states(self):
    
    # return something for loading
    pass
    
def load_states(self, states):
    
    pass
