# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import random
import PIL.ImageOps as ImageOps
from fvcore.common.registry import Registry

from ..types import PIL
from ..utils import load_image, save_image

AUGMENTATION_REGISTRY = Registry("AUGMENTATION")
AUGMENTATION_REGISTRY.__doc__ = " Registry for Augmentation"


@AUGMENTATION_REGISTRY.register()
class HorizontalFlipping(object):
    def __init__(self, value: bool):
        self.use = value

    def __call__(self, image: PIL, target: PIL):
        assert self.use == True
        image = ImageOps.mirror(image)
        target = ImageOps.mirror(target)
        return  image, target


class Augmentation(object):
    def __init__(self, type: str, save: bool, save_dir: str, multiple: float, funcs=None):
        self.type = type
        self.save = save
        self.funcs = funcs
        self.multiple = multiple

        self.augmentation_img_file, self.augmentation_mask_file = self._get_augmentation_dir(save_dir)

    def __call__(self, dataset_list: list, multiple=2):
        '''
            save 를 안 하면 augmentation 을 사용하지 못 하게 되어있음
            >> datasets 과 형식을 맞추기 위해 dir 형태로 가져가기 때문

            # todo : (1) 기존에 생성한 augmentation 파일을 불러와서 사용할 수 있게 하거나
            #        (2) 저장하지 않고 바로 사용 사용할 수 있는 방향으로 수정
            #        (3) (차차차후) 증식이도 붙일 수 있게 되면 좋을듯

        '''

        using_list = self._get_using_list(dataset_list=dataset_list)
        augmentation_samples = []
        for idx, (img_dir, mask_dir) in enumerate(using_list):
            img = load_image(img_dir, type=self.type)
            mask = load_image(mask_dir, type='GRAY')

            augmentation_func = random.choice(self.funcs)
            aug_img, aug_mask = augmentation_func(image=img, target=mask)

            if self.save:
                aug_img_dir, aug_mask_dir = self._get_save_dir(img_dir, mask_dir, idx)
                save_image(img=aug_img, save_dir=aug_img_dir)
                save_image(img=aug_mask, save_dir=aug_mask_dir)
                augmentation_samples.append((aug_img_dir, aug_mask_dir))

        return augmentation_samples

    def _get_augmentation_dir(self, data_dir: str):
        augmentation_file = os.path.join(data_dir, 'augmentation')
        augmentation_img_file = os.path.join(augmentation_file, 'images')
        augmentation_mask_file = os.path.join(augmentation_file, 'masks')
        if self.save:
            os.makedirs(augmentation_img_file, exist_ok=True)
            os.makedirs(augmentation_mask_file, exist_ok=True)
        return augmentation_img_file, augmentation_mask_file

    def _get_save_dir(self, img_dir: str, mask_dir: str, idx: int):
        # idx for unique
        aug_img_dir = os.path.join(self.augmentation_img_file, f'{idx}_{os.path.basename(img_dir)}')
        aug_mask_dir = os.path.join(self.augmentation_mask_file, f'{idx}_{os.path.basename(mask_dir)}')
        return aug_img_dir, aug_mask_dir

    def _get_using_list(self, dataset_list: list) -> list:
        '''
        :param dataset_list: list ex.[(img_dir, mask_dir) ...]
        :param multiple: float >> 0.1 ~ 1.0
            >> 전체 데이터셋 중에 얼마나 증식에 사용할건지를 결정
        '''

        need_counts = int(len(dataset_list) * self.multiple)
        assert need_counts <= len(dataset_list)
        assert self.multiple > 0
        return random.sample(dataset_list, need_counts)
