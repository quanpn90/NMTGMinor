import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer


class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)
    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(
            self,
            params,
            lr=None,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=True,
            relative_step=True,
            warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        # this should override the rel_step_sz
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
                .rsqrt_()
                .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


def normalize_gradients(parameters, denom):
    """ early return if no need to normalize """
    if denom == 1:
        return

    with torch.no_grad():

        parameters = list(filter(lambda p: p.grad is not None, parameters))

        denom = float(denom)

        for p in parameters:
            p.grad.data.div_(denom)


def detech_nan_inf(parameters):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        if torch.isinf(p.grad.data).any() or torch.isnan(p.grad.data).any():
            return True
        else:
            continue

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
    with torch.no_grad():
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
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-9,
                                        weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        elif self.method == 'adafactor':

            relative_step = False if self.lr > 0 else True
            self.optimizer = Adafactor(self.params, lr=self.lr if self.lr > 0 else None,
                                       eps=(1e-30, 1e-3), beta1=None,
                                       weight_decay=self.weight_decay,
                                       relative_step=relative_step,
                                       scale_parameter=False if self.lr > 0 else True,
                                       warmup_init=relative_step)
        elif self.method in ['fused_adam']:

            fast_adam = True
            try:
                import fused_optim
                if self.amsgrad:
                    print("Note: AMSGRAD is not compatible with Fused Adam")
                from onmt.modules.optimized.fused_adam import FusedAdam
                self.optimizer = FusedAdam(self.params, lr=self.lr,
                                           betas=(self.beta1, self.beta2), eps=1e-9,
                                           weight_decay=self.weight_decay, amsgrad=False,
                                           set_grad_none=False)
            except (RuntimeError, ModuleNotFoundError):
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

    def __init__(self, opt):
        self.optimizer = None
        self.params = None
        self.lr = opt.learning_rate
        self.model_size = opt.model_size
        self.max_grad_norm = opt.max_grad_norm
        self.update_method = opt.update_method
        self.method = opt.optim

        if self.lr > 0:
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

    def step(self, scaler=None, grad_denom=None, warmup=False):

        "Normalize gradients by batch size"
        self.normalize_grad(denom=grad_denom)

        "Compute gradients norm."
        # grad_norm = clip_grad_norm(self.params, self.max_grad_norm).item()

        overflow = False
        if hasattr(self.optimizer, "_amp_stash.already_patched") and self.optimizer._amp_stash.already_patched:
            overflow = True

        # if gradients have NaN/inf: return (which will be zeroed afterwards)
        # only do that if the scaler is None, i.e no mechanism to detect inf/nan implicitly
        # for apex amp, only skip if overflow is not detected
        # if detech_nan_inf(self.params):
        #     if scaler is None and not overflow:
        #         return

        "Automatically scale learning rate over learning period if not overflow"
        if not overflow:
            self._step += 1
            if 'noam' in self.update_method or 'cosine' in self.update_method:
                self.updateLearningRate()

        if scaler is not None:
            scaler.step(self.optimizer)
        else:
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
        if self.lr < 0:
            return

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

        if self.optimizer.param_groups[0]['lr'] is None:
            return self.lr
        else:
            return self.optimizer.param_groups[0]['lr']

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
        state_dict['step'] = self._step
        self._first_step = self._step
        print("* Loading from step %d " % self._step)

        state_dict.pop('_step', None)
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_starting_step(self, step):
        self._step = step
        self._first_step = step