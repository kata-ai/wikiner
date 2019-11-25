"""
AllenNLP uses most
`PyTorch learning rate schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_,
with a thin wrapper to allow registering them and instantiating them ``from_params``.
The available learning rate schedulers from PyTorch are
* `"step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
* `"multi_step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
* `"exponential" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
* `"reduce_on_plateau" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_
In addition, AllenNLP also provides a Noam schedule and `cosine with restarts
<https://arxiv.org/abs/1608.03983>`_, which are registered as "noam" and "cosine", respectively.
"""

# During training using the AllenNLP `Trainer`, this is the API and calling
#sequence for `step` and `step_batch`.  Also note that `step` is called
#once in `torch.optim.lr_scheduler._LRScheduler.__init__`.
#
#   scheduler = ... # creates scheduler, calls self.step(last_epoch + 1) in __init__
#
#   batch_num_total = 0
#   for epoch in range(num_epochs):
#       for batch in batchs_in_epoch:
#           # compute loss, update parameters with current learning rates
#           # call step_batch AFTER updating parameters
#           batch_num_total += 1
#           scheduler.step_batch(batch_num_total)
#       # call step() at the END of each epoch
#       scheduler.step(validation_metrics, epoch)
from typing import List
import logging

import numpy as np
import torch.optim.lr_scheduler
# from overrides import overrides

# from allennlp.common.checks import ConfigurationError
# from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler


logger = logging.getLogger(__name__) # pylint: disable=invalid-name


@LearningRateScheduler.register('slanted_triangular')
class SlantedTriangular(torch.optim.lr_scheduler._LRScheduler): # pylint: disable=protected-access
    """
    Implements the Slanted Triangular Learning Rate schedule with optional gradual
    unfreezing. The schedule corresponds to first linearly increasing the learning
    rate and annealing the learning based on a fixed ratio.
    If we gradually unfreeze, then in the first epoch of training, only the top
    layer is trained; in the second epoch, the top two layers are trained, etc.
    During freezing, the learning rate is increased and annealed over one epoch.
    After freezing finished, the learning rate is increased and annealed over
    the remaining training iterations.
    Note that with this schedule, early stopping should typically be avoided.
    Parameters
    ----------
    num_epochs : ``int``, required.
        The total number of epochs for which the model should be trained.
    num_steps_per_epoch: ``int``, required.
        The number of steps (updates, batches) per training epoch.
    cut_frac: ``float``, optional (default = 0.1).
        The fraction of the steps to increase the learning rate.
    ratio: ``float``, optional (default = 32).
        The ratio of the smallest to the (largest) base learning rate.
    gradual_unfreezing: ``bool``, optional (default = False).
        Whether gradual unfreezing should be used.
    discriminative_fine_tuning: ``bool``, optional (default = False).
        Whether discriminative fine-tuning (different learning rates per layer)
        are used.
    decay_factor: ``float``, optional (default = 0.38).
        The decay factor by which the learning rate is reduced with
        discriminative fine-tuning when going a layer deeper.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 num_steps_per_epoch: int,
                 cut_frac: float = 0.1,
                 ratio: int = 32,
                 last_epoch: int = -1,
                 gradual_unfreezing: bool = False,
                 discriminative_fine_tuning: bool = False,
                 decay_factor: float = 0.38) -> None:
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.gradual_unfreezing = gradual_unfreezing
        self.freezing_current = self.gradual_unfreezing
        self.is_first_epoch = True
        # track the actual number of steps for each epoch
        self.batch_num_total_epoch_end: List[int] = []
        if self.gradual_unfreezing:
            assert not optimizer.param_groups[-1]["params"], \
                "The default group should be empty."
        if self.gradual_unfreezing or discriminative_fine_tuning:
            assert len(optimizer.param_groups) > 2, \
                "There should be at least 3 param_groups (2 + empty default group)" \
                " for gradual unfreezing / discriminative fine-tuning to make sense."
        super().__init__(optimizer, last_epoch=last_epoch)
        if discriminative_fine_tuning:
            # skip the last param_group if it is has no parameters
            exponent = 0
            for i in range(len(self.base_lrs)-1, -1, -1):
                param_group = optimizer.param_groups[i]
                if param_group['params']:
                    param_group['lr'] = self.base_lrs[i] * decay_factor ** exponent
                    self.base_lrs[i] = param_group['lr']
                    exponent += 1
        # set up for the first batch
        self.last_batch_num_total = -1
        self.step_batch(0)

    def step(self, epoch=None):
        if len(self.batch_num_total_epoch_end) == 0: # pylint: disable=len-as-condition
            self.batch_num_total_epoch_end.append(0)
        else:
            self.batch_num_total_epoch_end.append(self.last_batch_num_total)

        if self.gradual_unfreezing:
            # the method is called once when initialising before the
            # first epoch (epoch 0) and then always at the end of each
            # epoch; so the first time, with epoch id 0, we want to set
            # up for epoch #1; the second time, still with epoch id 0,
            # we want to set up for epoch #2, etc.
            num_layers_to_unfreeze = epoch + 1 if self.is_first_epoch else epoch + 2
            if self.is_first_epoch:
                self.is_first_epoch = False
            if num_layers_to_unfreeze >= len(self.optimizer.param_groups)-1:
                logger.info('Gradual unfreezing finished. Training all layers.')
                self.freezing_current = False
            else:
                logger.info(f'Gradual unfreezing. Training only the top {num_layers_to_unfreeze} layers.')
            for i, param_group in enumerate(reversed(self.optimizer.param_groups)):
                for param in param_group["params"]:
                    # i = 0 is the default group; we care about i > 0
                    param.requires_grad = bool(i <= num_layers_to_unfreeze)

    def step_batch(self, batch_num_total=None):
        if batch_num_total is None:
            batch_num_total = self.last_batch_num_total + 1
        self.last_batch_num_total = batch_num_total
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = learning_rate

    def get_lr(self):
        # get the actual number of batches per epoch seen in training
        if len(self.batch_num_total_epoch_end) > 1:
            # have finished an epoch
            actual_num_steps_per_epoch = int(
                    self.batch_num_total_epoch_end[-1] /
                    (len(self.batch_num_total_epoch_end) - 1)
            )
        else:
            actual_num_steps_per_epoch = max(self.num_steps_per_epoch,
                                             self.last_batch_num_total)

        if self.freezing_current:
            # if we still freeze, we restrict the schedule to the current epoch
            num_steps = actual_num_steps_per_epoch
            step = min(self.last_batch_num_total - self.batch_num_total_epoch_end[-1],
                       num_steps)
        else:
            # otherwise we use the schedule for the rest of training
            if not self.gradual_unfreezing:
                frozen_steps = 0
            else:
                num_frozen_epochs = len(self.optimizer.param_groups) - 2
                frozen_steps = self.batch_num_total_epoch_end[num_frozen_epochs]
            num_steps = self.num_epochs * actual_num_steps_per_epoch - frozen_steps
            step = min(self.last_batch_num_total - frozen_steps,
                       num_steps)
        cut = int(num_steps * self.cut_frac)
        prop = step / cut if step < cut else 1 - (step - cut) / (num_steps - cut)
        return [lr * (1 + prop * (self.ratio - 1)) / self.ratio for lr in self.base_lrs]
