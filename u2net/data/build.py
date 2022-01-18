# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
from torch.utils.data import DataLoader

from . import datasets as Data
from .augmentation import build_augmentation
from .transforms import build_transforms
from .utils import make_samples
from .types import DictConfigs


def build_loader(configs: DictConfigs, is_train: bool):
    dataset_name = configs.user.data.dataset.train if is_train \
        else configs.user.data.dataset.test

    # Read files.
    samples = make_samples(
        img_dir=dataset_name.image,
        mask_dir=dataset_name.label,
        img_ext=configs.data.datasets.information.img_ext
    )

    # Transforms & Augmentation.
    Transforms = build_transforms(configs=configs)
    if is_train and configs.user.data.dataloader.augmentation.use:
        Augmentation = build_augmentation(configs=configs)
        samples += Augmentation(dataset_list=samples)

    # Build datasets.
    datasets = build_dataset(
        samples=samples,
        factory_name=configs.data.datasets.name,
        type=configs.data.datasets.information.type,
        transforms=Transforms
    )

    # build loader.
    loader = DataLoader(
        datasets,
        num_workers=configs.data.dataloader.num_workers,
        batch_size=configs.user.training.ims_per_batch,
        drop_last=configs.data.dataloader.drop_last,
        shuffle=configs.data.dataloader.shuffle
    )

    return loader


def build_dataset(samples: list, factory_name: str, type: str, transforms=None):
    factory = getattr(Data, factory_name)
    datasets = factory(
        samples=samples,
        type=type,
        transform=transforms
    )
    return datasets
