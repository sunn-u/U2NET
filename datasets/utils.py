import glob
import random


def get_all_items(directory: str):
    file_list = []
    IMG_EXT = ['**/*.jpg', '**/*.png']
    for ext in IMG_EXT:
        file_list += glob.glob(f'{directory}/{ext}', recursive=True)

    return file_list


def make_dataset(img_dir: str, mask_dir: str):
    # Suppose that the image name and the mask name are the same.

    img_list = sorted(get_all_items(img_dir))
    mask_list = sorted(get_all_items(mask_dir))

    samples = []
    for img_dir, mask_dir in zip(img_list, mask_list):
        samples.append((img_dir, mask_dir))

    return samples


def split_train_val(img_dir: str, mask_dir: str, train_percent: float, shuffle: bool):
    samples = make_dataset(img_dir, mask_dir)
    split_idx = int(len(samples) * train_percent)

    if shuffle:
        random.shuffle(samples)

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    return train_samples, val_samples
