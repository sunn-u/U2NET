# Coding by SunWoo(tjsntjsn20@gmail.com)

from ..data.types import DictConfigs
from .evaluation import EVALUTATION_REGISTRY


def build_evaluation(configs: DictConfigs):
    evaluation_name = configs["MODEL"]["EVALUATION"]["NAME"]
    evaluation = EVALUTATION_REGISTRY.get(evaluation_name)(configs["MODEL"]["EVALUATION"]["SMOOTH"])
    return evaluation
