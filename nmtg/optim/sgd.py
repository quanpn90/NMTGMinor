from torch.optim import SGD

from nmtg.optim import Optimizer, register_optimizer


@register_optimizer('sgd')
class SGDOptimizer(Optimizer):

    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = SGD(params, lr=self.get_lr(), weight_decay=args.weight_decay, momentum=0.0)

    @staticmethod
    def add_args(parser):
        parser.add_argument('-weight_decay', type=float, default=0.0,
                            help='weight decay (L2 penalty)')