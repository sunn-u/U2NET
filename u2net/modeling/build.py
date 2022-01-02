# Coding by SunWoo(tjsntjsn20@gmail.com)

from .rsu import MODEL_REGISTRY
from .loss import LOSS_REGISTRY
from ..data.types import DictConfigs


def build_model(configs: DictConfigs):
    model_name = configs["MODEL"]["META_ARCHITECTURE"]
    model = MODEL_REGISTRY.get(model_name)(configs["MODEL"]["ARCHITECTURE_DICT"])
    return model.to(configs["MODEL"]["DEVICE"])


def build_criterion(configs: DictConfigs):
    loss_name = configs["MODEL"]["LOSS"]
    criterion = LOSS_REGISTRY.get(loss_name)
    return criterion
