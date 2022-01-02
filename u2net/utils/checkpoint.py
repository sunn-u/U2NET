# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import torch

from u2net.data.types import Logging
from u2net.modeling.base import weight_init_xavier


class CheckPointer(object):
    def __init__(self, save_dir: str, logger: Logging):
        self.save_dir = save_dir
        self.logger = logger

    def save(self, model, epoch: int, last: bool):
        save_file = os.path.join(self.save_dir, f'{str(epoch).zfill(7)}.pth')
        torch.save(model, save_file)
        if last:
            self._tag_final_checkpoint(model,)
        else:
            self._tag_last_checkpoint(model,)

        self.logger.debug(f"Saving Model to {save_file}")

    def load(self, model, weight_dir=None):
        if weight_dir is not None:
            assert os.path.exists(weight_dir) == True
            self.logger.debug(f"Loading Model from {weight_dir}.")
            return torch.load(model, weight_dir)

        if self._has_checkpoint():
            checkpoint_file = self._get_checkpoint_file()
            self.logger.debug(f"Loading Checkpoint from {checkpoint_file}.")
            return torch.load(model, checkpoint_file)
        else:
            self.logger.debug(f"No checkpoint found. Initializing model from scratch.")
            model = self._apply_init(model)
            return model

    def _tag_final_checkpoint(self, model):
        save_file = os.path.join(self.save_dir, 'final_model.pth')
        torch.save(model, save_file)

    def _tag_last_checkpoint(self, model):
        save_file = os.path.join(self.save_dir, 'last_checkpoint.pth')
        torch.save(model, save_file)

    def _has_checkpoint(self):
        return os.path.exists(os.path.join(self.save_dir, 'last_checkpoint.pth'))

    def _get_checkpoint_file(self):
        return os.path.join(self.save_dir, 'last_checkpoint.pth')

    def _apply_init(self, model):
        return model.apply(weight_init_xavier)
