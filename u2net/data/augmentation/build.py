# Coding by SunWoo(tjsntjsn20@gmail.com)

from u2net.data.augmentation.augmentation import AUGMENTATION_REGISTRY
from .augmentation import Augmentation


def build_augmentation(configs: dict):
    augmentation_configs = configs["DATALOADER"]["AUGMENTATION"]
    funcs = []
    for name, value in augmentation_configs["FUNC"].items():
        if value is not None:
            func = AUGMENTATION_REGISTRY.get(name)(value)
            funcs.append(func)

    return Augmentation(
        type=configs["DATASETS"]["INFORMATION"]["TYPE"],
        save=augmentation_configs["SAVE"],
        data_dir=configs["DATA_DIR"],
        funcs=funcs
    )
