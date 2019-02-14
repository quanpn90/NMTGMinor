from torch.optim import Adam

from nmtg.optim import Optimizer, register_optimizer


@register_optimizer('adam')
class AdamOptimizer(Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super().__init__(params)
        self._optimizer = Adam(params, weight_decay=weight_decay,
                               betas=betas, eps=eps, amsgrad=amsgrad)

    @staticmethod
    def add_options(parser):
        parser.add_argument('-beta1', type=float, default=0.9,
                            help='beta_1 value for adam')
        parser.add_argument('-beta2', type=float, default=0.98,
                            help='beta_2 value for adam')
        parser.add_argument('-weight_decay', type=float, default=0.0,
                            help='weight decay (L2 penalty)')
        parser.add_argument('-amsgrad', action='store_true',
                            help='Using AMSGRad for adam')

    @classmethod
    def build_optimizer(cls, args, params):
        return cls(params, (args.beta1, args.beta2), weight_decay=args.weight_decay,
                   amsgrad=args.amsgrad)
