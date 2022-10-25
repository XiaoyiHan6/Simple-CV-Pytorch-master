import math
from torch import optim
from torch.optim.lr_scheduler import LambdaLR


class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class adjust_learning_rate(object):
    """Sets the learning rate to the initial LR decayed by 10 at every
    specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, init_lr, gamma):
        super(adjust_learning_rate, self).__init__()
        self.init_lr = init_lr
        self.gamma = gamma

    def forward(self, optimizer, step):
        lr = self.init_lr * (self.gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_scheduler(cfg, optimizer):
    scheduler = cfg['OPTIMIZE']['SCHEDULER']
    epoch = cfg['OPTIMIZE']['EPOCH']
    lr = cfg['OPTIMIZE']['LR']
    # 1.
    if scheduler == 'StepLR':
        step = epoch // 3
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)
    # 2.
    elif scheduler == 'MultiStepLR':
        if epoch == 30:
            milestones = [20, 25, 30]
        elif epoch == 60:
            milestones = [40, 55, 60]
        elif epoch == 100:
            milestones = [80, 95, 100]
        elif epoch == 130:
            milestones = [80, 110, 130]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # 3.
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # 4.
    elif scheduler == 'LinearLR':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / 3,
                                                end_factor=lr, total_iters=(epoch // 3 * 2))
    # 5.
    elif scheduler == 'ConstantLR':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0 / 3, total_iters=(epoch // 3 * 2))
    # 6.
    elif scheduler == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    # 7.
    elif scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=lr * 0.1)
    # 8.
    elif scheduler == 'WarmupCosineSchedule':
        if epoch < 100:
            warmup_steps = 1
        else:
            warmup_steps = 3
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps, epoch)
    # 9.
    elif scheduler == 'WarmupLinearSchedule':
        if epoch < 100:
            warmup_steps = 1
        else:
            warmup_steps = 3
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps, epoch)
    # 10.
    elif scheduler == 'adjust_learning_rate':
        scheduler = adjust_learning_rate(lr, 0.1)
    else:
        raise NotImplementedError

    return scheduler


if __name__ == '__main__':
    from utils.optimizer import get_optimizer
    from options.detection.SSD.train_options import cfg
    from torchvision.models.detection.ssd import ssd300_vgg16

    ssd = ssd300_vgg16()
    optimizer = get_optimizer(cfg, ssd)
    scheduler = get_scheduler(cfg, optimizer)
    step = 0
    for iter in range(cfg['OPTIMIZE']['MAX_ITER']):
        if iter in cfg['OPTIMIZE']['LR_STEP']:
            print("iter:", iter)
            step += 1
            scheduler.forward(optimizer, step)
