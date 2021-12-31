# Coding by SunWoo(tjsntjsn20@gmail.com)

import glob
import random
from PIL import Image

from .types import PIL


def load_image(directory: str, type: str) -> PIL:
    image = Image.open(directory)
    if type == 'RGB':
        return image.convert('RGB')
    if type == 'GRAY':
        return image.convert('L')


def save_image(img: PIL, save_dir: str):
    img.save(save_dir)


def get_all_items(directory: str, img_ext: list) -> list:
    file_list = []
    for ext in img_ext:
        file_list += glob.glob(f'{directory}/**/*.{ext}', recursive=True)
    return file_list


def make_samples(img_dir: str, mask_dir: str, img_ext: list) -> list:
    # Suppose that the image name and the mask name are the same.

    img_list = sorted(get_all_items(img_dir, img_ext))
    mask_list = sorted(get_all_items(mask_dir, img_ext))

    samples = []
    for img_dir, mask_dir in zip(img_list, mask_list):
        samples.append((img_dir, mask_dir))

    return samples


def split_train_val(samples: list, split_percent: float, shuffle: bool):
    split_idx = int(len(samples) * split_percent)

    if shuffle:
        random.shuffle(samples)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    return train_samples, val_samples
