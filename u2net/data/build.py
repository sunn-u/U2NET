# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
from torch.utils.data import DataLoader

from . import datasets as Data
from .augmentation import build_augmentation
from .transforms import build_transforms
from .utils import make_samples


def build_loader(configs: dict, is_train: bool):
    data_main_dir = configs["DATA_DIR"]
    dataset_name = configs["DATASETS"]["DIRECTORY"]["TRAIN"] \
        if is_train else configs["DATASETS"]["DIRECTORY"]["TEST"]

    # Read files.
    samples = make_samples(
        img_dir=os.path.join(data_main_dir, dataset_name[0]),
        mask_dir=os.path.join(data_main_dir, dataset_name[1]),
        img_ext=configs["DATASETS"]["INFORMATION"]["IMG_EXT"]
    )

    # Transforms & Augmentation.
    Transforms = build_transforms(configs=configs)
    if  is_train and configs["DATALOADER"]["AUGMENTATION"]["USE"]:
        Augmentation = build_augmentation(configs=configs)
        samples += Augmentation(dataset_list=samples)

    # Build datasets.
    datasets = build_dataset(
        samples=samples,
        factory_name=configs["DATASETS"]["CLASS"]["FACTORY_TRAIN"]
        if is_train else configs["DATASETS"]["CLASS"]["FACTORY_TEST"],
        type=configs["DATASETS"]["INFORMATION"]["TYPE"],
        transforms=Transforms
    )

    # build loader.
    loader = DataLoader(
        datasets,
        num_workers=configs["DATALOADER"]["NUM_WORKERS"],
        batch_size=configs["SOLVER"]["IMS_PER_BATCH"],
        drop_last=configs["DATALOADER"]["DROP_LAST"],
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
