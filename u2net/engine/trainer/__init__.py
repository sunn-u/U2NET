# Coding by SunWoo(tjsntjsn20@gmail.com)

from fvcore.common.registry import Registry

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.__doc__ = " Registry for Trainer "

from .default import DefaultTrainer
# from .default import CocoTrainer
