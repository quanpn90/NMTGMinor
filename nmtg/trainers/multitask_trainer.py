from torch import Tensor
from typing import Sequence

from nmtg.optim.lr_scheduler.lr_scheduler import MultiScheduler
from nmtg.optim.optimizer import MultiOptimizer
from nmtg.trainers import Trainer


class MultitaskTrainer(Trainer):
    # This is more of a guide on how to write a Multitask trainer, do not actually inherit from this class

    def _build_optimizers(self, model):
        # Call self._build_optimizer with appropriate params
        # Return [(lr_scheduler1, optim1), (lr_scheduler2, optim2), ...]
        raise NotImplementedError

    def load_data(self, model_args=None):
        # Should call self._build_multi_optimizer instead of self._build_optimizer
        raise NotImplementedError

    def _get_loss(self, model, batch) -> (Sequence[Tensor], float):
        # Should return one loss per parameter group/optimizer
        raise NotImplementedError

    def _build_multi_optimizer(self, model):
        lr_schedulers, optimizers = zip(*self._build_optimizers(model))
        lr_scheduler = MultiScheduler(lr_schedulers)
        optimizer = MultiOptimizer(optimizers)
        return lr_scheduler, optimizer
