# Coding by SunWoo(tjsntjsn20@gmail.com)

from u2net.data.transforms.transforms import TRANSFORMS_REGISTRY
from ..types import PIL, DictConfigs


def build_transforms(configs: DictConfigs):
    transforms_configs = configs.data.dataloader.transforms
    funcs = []
    for name, value in transforms_configs.items():
        if value is not None:
            func = TRANSFORMS_REGISTRY.get(name)(value)
            funcs.append(func)
    return Compose(funcs)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: PIL):
        for t in self.transforms:
            image = t(image)
        return image
