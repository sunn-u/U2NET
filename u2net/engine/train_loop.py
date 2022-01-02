# Coding by SunWoo(tjsntjsn20@gmail.com)

from ..utils.checkpoint import CheckPointer
from ..data.types import Tensor


class TrainerBase(object):
    '''
        Base class for iterative trainer.
    '''

    def __init__(self):
        self.epoch: int = 0
        self.start_epoch: int = 0
        self.max_epoch: int = 0

    def train(self, start_epoch: int, max_epoch: int):
        self.epoch = self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_step()
            self.run_step()
            self.after_step()
        self.epoch += 1
        self.after_train()

    def before_train(self):
        pass

    def before_step(self):
        pass

    def run_step(self):
        pass

    def after_step(self):
        pass

    def after_train(self):
        pass


class SimpleTrainer(TrainerBase):
    '''
        A simple trainer for the most common type of task
            - 1. Compute the loss with a data from the data_loader
            - 2. Compute the gradients with the above loss.
            - 3. Update the model with the optimizer.
    '''

    def __init__(self, configs, logger, model, optimizer, criterion):
        super().__init__()

        self.configs = configs
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpointer = CheckPointer(save_dir=self.configs["OUTPUT_DIR"], logger=self.logger)
        self.checkpoint = -1

    def before_train(self):
        # Load model
        self.model = self.checkpointer.load(
            model=self.model,
            weight_dir=self.configs["MODEL"]["WEIGHT"]
        )

    def run_step(self, x: Tensor, target: Tensor):
        self.model.train()

        pred_masks, fuse_mask = self.model(x)
        loss = self.criterion(gt_mask=target, pred_masks=pred_masks, fuse_mask=fuse_mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.debug(f'[{self.epoch}/{self.max_epoch}] TRAIN LOSS :: {loss.item()}')

    def after_step(self, x: Tensor, target: Tensor, save: bool):
        if save:
            self.checkpointer.save(model=self.model, epoch=self.epoch, last=False)
        self.model.eval()

        pred_masks, fuse_mask = self.model(x)
        loss = self.criterion(gt_mask=target, pred_masks=pred_masks, fuse_mask=fuse_mask)
        return loss.item(), fuse_mask

    def after_train(self):
        # Save last model
        self.checkpointer.save(model=self.model, epoch=self.epoch, last=True)
