# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch.optim as optim

from ..data.types import DictConfigs


def build_optimizer(configs: DictConfigs, model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=configs["SOLVER"]["OPTIMIZER"]["BASE_LR"],
        betas=configs["SOLVER"]["OPTIMIZER"]["BETAS"],
        eps=configs["SOLVER"]["OPTIMIZER"]["EPS"],
        weight_decay=configs["SOLVER"]["OPTIMIZER"]["WEIGHT_DECAY"]
    )
    return optimizer


def build_lr_scheduler(configs: DictConfigs, optimizer):
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: configs["SOLVER"]["SCHEDULER"]["FACTOR"] ** epoch
    )
    return scheduler
