# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import torch
import logging

from u2net.utils.logger import set_logger
from u2net.layers.layer import weight_init_xavier


class CheckPointer(object):
    def __init__(self, max_epoch: int, save_dir: str):
        self.max_epoch = max_epoch
        self.save_dir = save_dir
        self.best_score = -1

        self.logger = logging.getLogger(__name__)
        self.best_logger = set_logger(level=2, logger_name='best_model', save_dir=save_dir)

    def save(self, model, epoch: int, score: float):
        save_file = os.path.join(self.save_dir, f'{str(epoch).zfill(7)}.pth')
        torch.save(model, save_file)
        self._tag_last_checkpoint(model)

        if score > self.best_score:
            self.best_score = score
            self._tag_best_model(model)

        if epoch == self.max_epoch:
            self._tag_final_checkpoint(model)
        self.logger.info(f"Saving Model to {save_file}")

    def load(self, model, weight_dir=None):
        if weight_dir is not None:
            assert os.path.exists(weight_dir) == True
            self.logger.info(f"Loading Model from {weight_dir}.")
            return torch.load(weight_dir)

        if self._has_checkpoint():
            checkpoint_file = self._get_checkpoint_file()
            self.logger.info(f"Loading Checkpoint from {checkpoint_file}.")
            return torch.load(checkpoint_file)
        else:
            self.logger.info(f"No checkpoint found. Initializing model from scratch.")
            model = self._apply_init(model)
            return model

    def _tag_final_checkpoint(self, model):
        save_file = os.path.join(self.save_dir, 'final_model.pth')
        torch.save(model, save_file)

    def _tag_last_checkpoint(self, model):
        save_file = os.path.join(self.save_dir, 'last_checkpoint.pth')
        torch.save(model, save_file)

    def _tag_best_model(self, model):
        save_file = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(model, save_file)
        self.best_logger.info(f"{save_file}: {self.best_score}")

    def _has_checkpoint(self):
        return os.path.exists(os.path.join(self.save_dir, 'last_checkpoint.pth'))

    def _get_checkpoint_file(self):
        return os.path.join(self.save_dir, 'last_checkpoint.pth')

    def _apply_init(self, model):
        return model.apply(weight_init_xavier)
