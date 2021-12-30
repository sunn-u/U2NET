# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import shutil
import cv2
import random

from datasets.utils import get_all_items



class Augmentation:
    def __init__(self, train_set, val_set, data_dir):
        self.save_dir = self._get_augmentation_dir(data_dir)
        self.train_samples = train_set
        self.val_samples = val_set
        self.data_count = len(train_set) + len(val_set)

    def _get_augmentation_dir(self, data_dir):
        path_tuple = os.path.split(data_dir)
        root_path = path_tuple[0]
        folder_name = f'{path_tuple[1]}_augmentation'
        save_dir = os.path.join(root_path, folder_name)

        return save_dir

    def augmentation_func(self):
        if os.path.exists(self.save_dir):
            if get_all_items(self.save_dir) == self.data_count:
                train_samples = self.matching_file_name(train=True)
                val_samples = self.matching_file_name(train=False)

                return train_samples, val_samples

        shutil.rmtree(self.save_dir)
        os.makedirs(os.path.join(self.save_dir, 'image'))
        os.makedirs(os.path.join(self.save_dir, 'mask'))

        train_samples = self.horizontal_flipping(train=True)
        val_samples = self.horizontal_flipping(train=False)

        return train_samples, val_samples

    def horizontal_flipping(self, train=True):
        samples = self.train_samples if train else self.val_samples

        for img_dir, mask_dir in samples:
            aug_img_dir = os.path.join(self.save_dir, 'image', os.path.basename(img_dir))
            aug_mask_dir = os.path.join(self.save_dir, 'mask', os.path.basename(mask_dir))

            cv2.imwrite(aug_img_dir, cv2.flip(cv2.imread(img_dir), 1))
            cv2.imwrite(aug_mask_dir, cv2.flip(cv2.imread(mask_dir), 1))

            samples.append((aug_img_dir, aug_mask_dir))
        random.shuffle(samples)

        return samples

    def matching_file_name(self, train=True):
        samples = self.train_samples if train else self.val_samples

        for img_dir, mask_dir in samples:
            aug_img_dir = os.path.join(self.save_dir, 'image', os.path.basename(img_dir))
            aug_mask_dir = os.path.join(self.save_dir, 'mask', os.path.basename(mask_dir))
            samples.append((aug_img_dir, aug_mask_dir))
        random.shuffle(samples)

        return samples
