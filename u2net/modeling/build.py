# Coding by SunWoo(tjsntjsn20@gmail.com)

from .u2net import MODEL_REGISTRY
from .loss import LOSS_REGISTRY
from ..data.types import DictConfigs


def build_model(configs: DictConfigs):
    model_name = configs.model.meta_architecture.name
    model = MODEL_REGISTRY.get(model_name)(configs)
    return model.to(configs.user.model.device)


def build_criterion(configs: DictConfigs):
    loss_name = configs.model.loss.name
    criterion = LOSS_REGISTRY.get(loss_name)()
    return criterion
