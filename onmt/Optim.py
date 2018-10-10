import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch, math

#~ from torch.nn.utils import clip_grad_norm

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

class Adam(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class Optim(object):

    def set_parameters(self, params):
    
        params_ = filter(lambda p: p.requires_grad, params)
        self.params = list(params_)  # careful: params may be a generator
        #~ self.optimizer = Adam(self.params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-9,
                                    #~ weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.0)
        elif self.method == 'adam':
            self.optimizer = Adam(self.params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-9,
                                        weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
        # ~ print(self.optimizer)
        
    
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
        self.warmup_steps=opt.warmup_steps
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.weight_decay = opt.weight_decay
        self.amsgrad = opt.amsgrad 
        
            
    def step(self, grad_denom=None):
        
        #~ if detech_nan(self.params):
            #~ self.zero_grad()
            #~ return 0
        
        "Normalize gradients by batch size"
        self.normalize_grad(denom=grad_denom)
        
        "Compute gradients norm."
        grad_norm = clip_grad_norm(self.params, self.max_grad_norm).item()
        #~ print(grad_norm)
            
        "Automatically scale learning rate over learning period"
        self._step += 1
        if 'noam' in self.update_method:
            self.updateLearningRate()
        
        #~ print("UDPATE PARAMETERS")
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
        self.updateLearningRate()
      
    def zero_grad(self):
        self.optimizer.zero_grad()
      
# This version of Adam keeps an fp32 copy of the parameters and 
# does all of the parameter updates in fp32, while still doing the
# forwards and backwards passes using fp16 (i.e. fp16 copies of the 
# parameters and fp16 activations).
#
# Note that this calls .float().cuda() on the params such that it 
# moves them to gpu 0--if you're using a different GPU or want to 
# do multi-GPU you may need to deal with this.
class Adam16(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        eps = 1e-4
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
            # for p in group['params']:
        
        self.fp32_param_groups = [p.data.float().cuda() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group,fp32_group in zip(self.param_groups,self.fp32_param_groups):
            for p,fp32_p in zip(group['params'],fp32_group['params']):
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss
