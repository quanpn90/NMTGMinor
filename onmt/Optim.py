import torch.optim as optim
from torch.nn.utils import clip_grad_norm

def normalize_gradients(parameters, denom):
    
    """ early return if no need to normalize """
    if denom == 1:
        return
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    denom = float(denom)
    
    for p in parameters:
        p.grad.data.div_(denom)


class NoamOptim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-9,
                                    weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        print(self.optimizer)
        
    # def __init__(self, lr, max_grad_norm=0,
                 # model_size=512, warmup_steps=4096):
    def __init__(self, opt):
        self.lr = opt.learning_rate
        self.model_size = opt.model_size
        self.max_grad_norm = opt.max_grad_norm
        self.init_lr = self.model_size**(-0.5) * self.lr
        self.lr = self.init_lr
        self._step = 0 
        self.warmup_steps=opt.warmup_steps
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.weight_decay = opt.weight_decay
        self.amsgrad = opt.amsgrad 
            
    def step(self, grad_denom=None):
        
        "Normalize gradients by batch size"
        self.normalize_grad(denom=grad_denom)
        
        "Compute gradients norm."
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
            
            
        "Automatically scale learning rate over learning period"
        self.updateLearningRate()
        self.optimizer.step()
        
        """Reset the denom for normalization"""
    def normalize_grad(self, denom=None):
        
        if denom is None:
            denom = 1
        
        #~ if self.num_accumulated_samples == 0:
            #~ denom = 1
        #~ else:
            #~ denom = self.num_accumulated_samples
        normalize_gradients(self.params, denom)
    
    def updateLearningRate(self):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """
        
        self._step += 1
        if self._step <= self.warmup_steps:
            self.lr = self.init_lr*self._step*self.warmup_steps**(-1.5)
        else:
            self.lr = self.init_lr*self._step**(-0.5)
        
        self.optimizer.param_groups[0]['lr'] = self.lr
        

    
    def setLearningRate(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        self.lr = lr
        
    def getLearningRate(self):
        return self.lr
        
    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['_step'] = self._step
        return state_dict
        
    def load_state_dict(self, state_dict):
        
        self._step = state_dict['_step']
        
        state_dict.pop('_step', None)
        self.optimizer.load_state_dict(state_dict)
      
    def zero_grad(self):
        self.optimizer.zero_grad()
      
class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm,
                 lr_decay=1, start_decay_at=None):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0

    def step(self):
        "Compute gradients norm."
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()
        self._step += 1

    def updateLearningRate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        self.optimizer.param_groups[0]['lr'] = self.lr
    
    def setLearningRate(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        self.lr = lr
