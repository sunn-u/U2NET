# Coding by SunWoo(tjsntjsn20@gmail.com)

from .rsu import MODEL_REGISTRY
from .loss import LOSS_REGISTRY
from ..data.types import DictConfigs


def build_model(configs: DictConfigs):
    meta_architecture = configs.model.meta_architecture
    model_name = meta_architecture.name
    model = MODEL_REGISTRY.get(model_name)(meta_architecture.architecture_dict)
    return model.to(configs.user.model.device)


def build_criterion(configs: DictConfigs):
    loss_name = configs.model.loss.name
    criterion = LOSS_REGISTRY.get(loss_name)
    return criterion
