# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import torch

from u2net.models.base import weight_init_xavier


class CheckPointer(object):
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def save(self, save_dir: str, epoch: int):
        save_file = os.path.join(save_dir, f'{str(epoch).zfill(7)}.pth')
        torch.save(self.model, save_file)
        self._tag_last_checkpoint(save_dir)

        self.logger(f"Saving Model to {save_file}")

    def load(self, save_dir: str, weight_dir=None):
        if weight_dir is not None:
            assert os.path.exists(weight_dir) == True
            self.logger(f"Loading Model from {weight_dir}.")
            return torch.load(self.model, weight_dir)

        if self._has_checkpoint(save_dir):
            checkpoint_file = self._get_checkpoint_file(save_dir)
            self.logger(f"Loading Checkpoint from {checkpoint_file}.")
            return torch.load(self.model, checkpoint_file)
        else:
            self.logger(f"No checkpoint found. Initializing model from scratch.")
            self._apply_init()
            return self.model

    def _tag_last_checkpoint(self, save_dir: str):
        save_file = os.path.join(save_dir, 'last_checkpoint.pth')
        torch.save(self.model, save_file)

    def _has_checkpoint(self, save_dir: str):
        return os.path.exists(os.path.join(save_dir, 'last_checkpoint.pth'))

    def _get_checkpoint_file(self, save_dir: str):
        return os.path.join(save_dir, 'last_checkpoint.pth')

    def _apply_init(self):
        self.model.apply(weight_init_xavier)
