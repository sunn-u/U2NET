# Coding by SunWoo(tjsntjsn20@gmail.com)

from ..data.types import DictConfigs, Logging
from .trainer import TRAINER_REGISTRY


def build_trainer(configs: DictConfigs, logger: Logging):
    trainer_name = configs["TRAINER"]["NAME"]
    trainer = TRAINER_REGISTRY.get(trainer_name)(configs, logger)
    return trainer


def build_tester(configs: DictConfigs, logger: Logging):
    # todo : trainer 에 test 하는 부분만 빼서 사용할 수 있게 수정
    trainer_name = configs["TRAINER"]["NAME"]
    testor = TRAINER_REGISTRY.get(trainer_name)(configs, logger)
    return testor
