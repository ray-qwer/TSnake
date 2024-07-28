from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from collections import Counter
import torch
from bisect import bisect_right

def make_lr_scheduler(optimizer, config, iter_per_epoch):
    if 'warmup' in config.train.optimizer.keys():
        return WarmupMultiStepLR(optimizer, milestones=config.train.optimizer['milestones'], gamma=config.train.optimizer['gamma'],  
                                 warmup_iters=iter_per_epoch* config.train.optimizer["warmup"]["epoch"])
    elif config.train.scheduler_name == "linear":
        return LinearLR(optimizer, **config.train.scheduler_option)
    scheduler = MultiStepLR(optimizer, milestones=config.train.optimizer['milestones'],
                            gamma=config.train.optimizer['gamma'])
    return scheduler


def set_lr_scheduler(scheduler, config):
    scheduler.milestones = Counter(config.train.optimizer['milestones'])
    scheduler.gamma = config.train.optimizer['gamma']


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """multi-step learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be main.tex list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]