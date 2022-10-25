from torch import optim


def get_optimizer(cfg, module):
    optimizer = cfg['OPTIMIZE']['OPTIMIZER']
    lr = cfg['OPTIMIZE']['LR']
    # 1.
    if optimizer == 'sgd':
        optimizer = optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # 2.
    elif optimizer == 'asgd':
        optimizer = optim.ASGD(module.parameters(), lr=lr)
    # 3.
    elif optimizer == 'adam':
        optimizer = optim.Adam(module.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-4)
    # 4.
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(module.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-4)
    # 5.
    elif optimizer == 'radam':
        optimizer = optim.RAdam(module.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-4)
    else:
        raise NotImplementedError

    return optimizer
