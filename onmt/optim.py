import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch, math


class FusedAdam(torch.optim.Optimizer):

    """Implements Adam algorithm. Currently GPU-only.
       Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square.
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params,
                 lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, eps_inside_sqrt=False,
                 weight_decay=0., max_grad_norm=0., amsgrad=False):
        global fused_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        if amsgrad:
            raise RuntimeError('AMSGrad variant not supported.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    def step(self, closure=None, grads=None, output_params=None,
             scale=1., grad_norms=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients.
                (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None]*len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            output_params_group = [None]*len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0]) != list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None]*len(self.param_groups)

        for group, grads_this_group, output_params_this_group, \
            grad_norm in zip(self.param_groups, grads_group,
                             output_params_group, grad_norms):
            if grads_this_group is None:
                grads_this_group = [None]*len(group['params'])
            if output_params_this_group is None:
                output_params_this_group = [None]*len(group['params'])

            # compute combined scale factor for this group
            combined_scale = scale
            if group['max_grad_norm'] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad, output_param in zip(group['params'],
                                             grads_this_group,
                                             output_params_this_group):
                # note: p.grad should not ever be set for correct operation of
                # mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse \
                                       gradients, please consider \
                                       SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                out_p = torch.tensor([], dtype=torch.float) if output_param \
                    is None else output_param
                fused_adam_cuda.adam(p.data,
                                     out_p,
                                     exp_avg,
                                     exp_avg_sq,
                                     grad,
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     combined_scale,
                                     state['step'],
                                     self.eps_mode,
                                     bias_correction,
                                     group['weight_decay'])
        return loss


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
        if torch.equal(p.grad.data, p.grad.data):
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
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.0)
        elif self.method in ['adam', 'fused_adam']:

            fast_adam = True
            try:
                import apex
                if self.amsgrad:
                    print("Note: AMSGRAD is not compatible with Fused Adam")
                self.optimizer = apex.optimizers.FusedAdam(self.params, lr=self.lr,
                                                           betas=(self.beta1, self.beta2), eps=1e-9,
                                                           weight_decay=self.weight_decay, amsgrad=False,
                                                           set_grad_none=False)
            except RuntimeError:
                fast_adam = False

            if not fast_adam:
                self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-9,
                                            weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        elif self.method in ['novograd']:
            try:
                import apex
                if self.amsgrad:
                    print("Note: AMSGRAD is not compatible with Fused Novograd")
                self.optimizer = apex.optimizers.FusedNovoGrad(self.params, lr=self.lr,
                                                               betas=(self.beta1, self.beta2), eps=1e-9,
                                                               weight_decay=self.weight_decay, amsgrad=False,
                                                               set_grad_none=False)
            except RuntimeError as e:
                raise e
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

        # print(self.optimizer)

    def __init__(self, opt):
        self.optimizer = None
        self.params = None
        self.lr = opt.learning_rate
        self.model_size = opt.model_size
        self.max_grad_norm = opt.max_grad_norm
        self.update_method = opt.update_method
        self.method = opt.optim

        if 'noam' in self.update_method:
            self.init_lr = self.model_size ** (-0.5) * self.lr
        elif 'cosine' in self.update_method:
            print("* Using Cosine learning rate schedule")
            self.scheduler = None
            self.eta_min = 0.0
            self.max_step = opt.max_step if hasattr(opt, 'max_step') else 33333
            self.init_lr = self.lr
        else:
            self.init_lr = self.lr
        self.lr = self.init_lr
        self._step = 0
        self._first_step = 0
        if self.update_method == 'noam2':
            self._step = opt.warmup_steps
        if self.update_method == 'cosine':
            self.min_lr = 0.00
        self.warmup_steps = opt.warmup_steps
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.weight_decay = opt.weight_decay
        self.amsgrad = opt.amsgrad
        self.max_steps = opt.max_steps

    def step(self, grad_denom=None, warmup=False):

        "Normalize gradients by batch size"
        self.normalize_grad(denom=grad_denom)

        "Compute gradients norm."
        # grad_norm = clip_grad_norm(self.params, self.max_grad_norm).item()

        overflow = False
        # if self.optimizer._amp_stash.already_patched:
        #     overflow = True
        "Automatically scale learning rate over learning period if not overflow"
        if not overflow:
            self._step += 1
            if 'noam' in self.update_method or 'cosine' in self.update_method:
                self.updateLearningRate()

        self.optimizer.step()

        # return grad_norm

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
                self.lr = self.init_lr * self._step * self.warmup_steps ** (-1.5)
            else:
                self.lr = self.init_lr * self._step ** (-0.5)

            self.optimizer.param_groups[0]['lr'] = self.lr

        elif self.update_method in ['cosine']:
            # if self.scheduler is None:
            #     self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_step,
            #                                                           eta_min=self.eta_min)
            #
            # self.scheduler.step(self._step)
            self.lr = self.min_lr + 0.5 * (self.init_lr - self.min_lr) * \
                (1 + math.cos((self._step / self.max_step) * math.pi))
            self.optimizer.param_groups[0]['lr'] = self.lr
            # self.lr = self.optimizer.param_groups[0]['lr']
            # self.lr = self.min_lr + (self.init_lr - self.min_lr) * \
            #           (1 + math.cos(math.pi * self._step / self.max_steps)) / 2
        elif self.update_method in ['regular', 'basic']:

            " :) "
            self.lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = self.lr

    def setLearningRate(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        self.lr = lr

    def getLearningRate(self):
        return self.lr

    def reset(self):
        self._step = self._first_step
        for group in self.optimizer.param_groups:
            if 'step' in group:
                group['step'] = self._first_step

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['_step'] = self._step
        return state_dict

    def load_state_dict(self, state_dict):
        self._step = state_dict['_step']
        self._first_step = self._step
        print("* Loading from step %d " % self._step)

        state_dict.pop('_step', None)
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()
