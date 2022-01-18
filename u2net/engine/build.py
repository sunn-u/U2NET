# Coding by SunWoo(tjsntjsn20@gmail.com)

from ..data.types import DictConfigs
from .trainer import TRAINER_REGISTRY


def build_trainer(configs: DictConfigs):
    trainer_name = configs.model.trainer.name
    trainer = TRAINER_REGISTRY.get(trainer_name)(configs)
    return trainer


def build_tester(configs: DictConfigs):
    trainer_name = configs.model.trainer.name
    testor = TRAINER_REGISTRY.get(trainer_name)(configs)
    return testor
