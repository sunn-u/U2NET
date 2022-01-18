# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch.optim as optim

from ..data.types import DictConfigs


def build_optimizer(configs: DictConfigs, model):
    parameters = configs.solver.optimizer.parameters

    optimizer = optim.Adam(
        model.parameters(),
        lr=parameters.base_lr,
        betas=parameters.betas,
        eps=parameters.eps,
        weight_decay=parameters.weight_decay
    )
    return optimizer


def build_lr_scheduler(configs: DictConfigs, optimizer):
    parameters = configs.solver.scheduler.parameters

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: parameters.factor ** epoch
    )
    return scheduler
