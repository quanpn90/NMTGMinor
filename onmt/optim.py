import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch, math


def normalize_gradients(parameters, denom):
    
    """ early return if no need to normalize """
    if denom == 1:
        return
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    denom = float(denom)
    
    for p in parameters:
        p.grad.data.div_(denom)


def detech_nan(parameters):
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    for p in parameters:        
        if (torch.equal(p.grad.data, p.grad.data)):
            continue
        else:
            return True
            
    return False


def clip_grad_norm(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    if max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    return total_norm


class Optim(object):

    def set_parameters(self, params):
    
        params_ = filter(lambda p: p.requires_grad, params)
        self.params = list(params_)  # careful: params may be a generator
        # self.optimizer = Adam(self.params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-9,
                                    #~ weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.0)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-9,
                                        weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        elif self.method == 'fused_adam':
            import apex
            if self.amsgrad:
                print("Note: AMSGRAD is not compatible with Fused Adam")
            self.optimizer = apex.optimizers.FusedAdam(self.params, lr=self.lr,
                                                       betas=(self.beta1, self.beta2), eps=1e-9,
                                                       weight_decay=self.weight_decay, amsgrad=False)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
        print(self.optimizer)

    def __init__(self, opt):
        self.lr = opt.learning_rate
        self.model_size = opt.model_size
        self.max_grad_norm = opt.max_grad_norm
        self.update_method = opt.update_method
        self.method = opt.optim
        
        if 'noam' in self.update_method:
            self.init_lr = self.model_size**(-0.5) * self.lr
        else:
            self.init_lr = self.lr
        self.lr = self.init_lr
        self._step = 0 
        if self.update_method == 'noam2':
            self._step = opt.warmup_steps
        if self.update_method == 'cosine':
            self.min_lr = 0.00001
        self.warmup_steps=opt.warmup_steps
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.weight_decay = opt.weight_decay
        self.amsgrad = opt.amsgrad
        self.max_steps = opt.max_steps

    def step(self, grad_denom=None):

        "Normalize gradients by batch size"
        self.normalize_grad(denom=grad_denom)
        
        "Compute gradients norm."
        grad_norm = clip_grad_norm(self.params, self.max_grad_norm).item()

        "Automatically scale learning rate over learning period"
        self._step += 1
        if 'noam' in self.update_method or 'cosine' in self.update_method:
            self.updateLearningRate()
        self.optimizer.step()
        
        return grad_norm
        
    """Reset the denom for normalization"""
    def normalize_grad(self, denom=None):
        
        if denom is None:
            denom = 1
        
        normalize_gradients(self.params, denom)
    
    def updateLearningRate(self):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.update_method in ['noam', 'noam2']:
            if self._step <= self.warmup_steps:
                self.lr = self.init_lr*self._step*self.warmup_steps**(-1.5)
            else:
                self.lr = self.init_lr*self._step**(-0.5)

        elif self.update_method in ['cosine']:
            self.lr = self.min_lr + (self.init_lr - self.min_lr) * \
                      (1 + math.cos(math.pi * self._step / self.max_steps)) / 2

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