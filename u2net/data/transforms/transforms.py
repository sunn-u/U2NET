# Coding by SunWoo(tjsntjsn20@gmail.com)

import random

from torchvision.transforms import functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int
from fvcore.common.registry import Registry

from ..types import PIL, Tensor

TRANSFORMS_REGISTRY = Registry("TRANSFORMS")
TRANSFORMS_REGISTRY.__doc__ = " Registry for Transforms "


@TRANSFORMS_REGISTRY.register()
class ToTensor(object):
    def __init__(self, value: bool):
        self.use = value

    def __call__(self, image: PIL):
        assert self.use == True
        return F.to_tensor(image)


@TRANSFORMS_REGISTRY.register()
class Resize(object):
    def __init__(self, value: dict):
        '''
        :param value:
            size: int
            mode: int
                - 0: nearest, 1: lanczos, 2: bilinear, 3: bicubic, 4:box , 5: hamming
        '''

        self.size = value["size"]
        self.mode = _interpolation_modes_from_int(value["mode"])

    def __call__(self, image: Tensor):
        return F.reisze(image, [self.size, self.size], interpolation=self.mode)


@TRANSFORMS_REGISTRY.register()
class RandomVerticalFlip(object):
    def __init__(self, value: bool):
        self.use = value

    def __call__(self, image: Tensor):
        assert self.use == True
        if random.choice([0, 1]):
            return F.vflip(image)
        return image


@TRANSFORMS_REGISTRY.register()
class CenterCrop:
    def __init__(self, value: int):
        self.size = value

    def __call__(self, image: Tensor):
        return F.center_crop(image, [self.size, self.size])
