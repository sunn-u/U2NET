# Coding by SunWoo(tjsntjsn20@gmail.com)

import logging
import weakref

from ..data.types import Tensor, List
from u2net.utils.events import EventStorage, get_event_storage


class HookBase(object):
    '''
        Base class for hooks.
    '''

    def before_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_train(self):
        pass


class TrainerBase(object):
    '''
        Base class for iterative trainer.
    '''

    def __init__(self):
        self.epoch: int = 0
        self.start_epoch: int = 0
        self.max_epoch: int = 0

        self._hooks: List[HookBase] = []
        self.storage: EventStorage

    def register_hooks(self, hooks):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            # for weak references.
            h.trainer = weakref.proxy(self)
        return self._hooks.extend(hooks)

    def train(self, start_epoch: int, max_epoch: int):
        self.epoch = self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        with EventStorage() as self.storage:
            try:
                self.before_train()
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.epoch += 1
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        self.storage.epoch += 1

    def after_train(self):
        for h in self._hooks:
            h.after_train()


class SimpleTrainer(TrainerBase):
    '''
        A simple trainer for the most common type of task
            - 1. Compute the loss with a data from the data_loader
            - 2. Compute the gradients with the above loss.
            - 3. Update the model with the optimizer.
    '''

    def __init__(self, configs, model, optimizer, criterion, loader):
        super().__init__()

        self.configs = configs
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loader = loader

    def run_step(self):
        self.model.train()
        losses = 0.
        for idx, (data, target) in enumerate(self.loader):
            data = data.to(self.configs.user.model.device)
            target = target.to(self.configs.user.model.device)

            pred_masks, fuse_mask = self.model(data)
            loss = self.criterion(gt_mask=target, pred_masks=pred_masks, fuse_mask=fuse_mask)

            self.optimizer.zero_grad()
            loss.backward()
            losses += loss
        losses /= len(self.loader)

        self._update_results(dict(loss=losses))

    def _update_results(self, loss_dict: dict):
        storage = get_event_storage()
        storage.put_scalars(
            loss={k: float(v.detach().cpu()) for k, v in loss_dict.items()}
        )
