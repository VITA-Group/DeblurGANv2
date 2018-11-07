from torch.optim import lr_scheduler
import math

class WarmRestart(lr_scheduler.CosineAnnealingLR):
    """This class implements Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.

    Set the learning rate of each parameter group using a cosine annealing schedule, When last_epoch=-1, sets initial lr as lr.
    This can't support scheduler.step(epoch). please keep epoch=None.
    """

    def __init__(self, optimizer, T_max=30, T_mult=1, eta_min=0, last_epoch=-1):
        """implements SGDR

        Parameters:
        ----------
        T_max : int
            Maximum number of epochs.
        T_mult : int
            Multiplicative factor of T_max.
        eta_min : int
            Minimum learning rate. Default: 0.
        last_epoch : int
            The index of last epoch. Default: -1.
        """
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        print(self.last_epoch)
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for
                base_lr in self.base_lrs]


class LinearDecay(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, min_lr=0, num_decay=1):
        """implements Linear Decay

        Parameters:
        ----------
        min_lr : int
            Minimum learning rate
        num_decay : int
            Number of epochs for which to decay lr linearly

        """
        self.min_lr = min_lr
        self.num_decay = num_decay
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr - self.last_epoch * ((base_lr - self.min_lr) / self.num_decay) for base_lr in self.base_lrs]


