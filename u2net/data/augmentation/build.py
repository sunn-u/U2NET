# Coding by SunWoo(tjsntjsn20@gmail.com)

import os

from u2net.data.augmentation.augmentation import AUGMENTATION_REGISTRY
from .augmentation import Augmentation
from ..types import DictConfigs


def build_augmentation(configs: DictConfigs):
    augmentation_configs = configs.user.data.dataloader.augmentation
    funcs = []
    for name, value in augmentation_configs.func.items():
        if value is not None:
            func = AUGMENTATION_REGISTRY.get(name)(value)
            funcs.append(func)

    return Augmentation(
        type=configs.data.datasets.information.type,
        save=augmentation_configs.save,
        save_dir=augmentation_configs.save_dir,
        multiple=augmentation_configs.multiple,
        funcs=funcs
    )
