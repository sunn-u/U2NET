from torch.utils.data import DataLoader
from torchvision import transforms
from sunn_models.datasets.loader.basic import BasicDataset

from sunn_models.datasets.utils import split_train_val
from sunn_models.datasets.augmentation import Augmentation

'''
1. resized to 320x320
2. randomly flipped vertically and cropped to 288x288
'''

def _train_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((320, 320)),
        transforms.RandomVerticalFlip(p=0.5),
        # todo : center crop 하면 왜 안 되지?
        # transforms.CenterCrop((288, 288))
        # transforms.CenterCrop(288)
    ])

def _val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((320, 320))
    ])

def build_train_dataset(cfg):
    train_samples, val_samples = split_train_val(
        img_dir=cfg.DATA.IMAGE_DIR,
        mask_dir=cfg.DATA.MASK_DIR,
        train_percent=cfg.DATA.TRAIN_PERCENT,
        shuffle=cfg.DATA.SHUFFLE
    )

    if cfg.DATA.AUGMENTATION:
        train_samples, val_samples = Augmentation(
            train_set=train_samples, val_set=val_samples, data_dir=cfg.DATA.IMAGE_DIR
        )
    train_dataset = BasicDataset(train_samples, transform=_train_transforms())
    val_dataset = BasicDataset(val_samples, transform=_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=cfg.DATA.DROP_LAST)
    val_loader = DataLoader(val_dataset, batch_size=1, drop_last=cfg.DATA.DROP_LAST)

    return train_loader, val_loader
