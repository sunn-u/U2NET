# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch
import copy

from . import TRAINER_REGISTRY
from u2net.engine.train_loop import TrainerBase, SimpleTrainer
from u2net.data.types import DictConfigs, Logging
from u2net.data import build_loader
from u2net.modeling import build_model, build_criterion
from u2net.solver import build_optimizer, build_lr_scheduler
from u2net.evaluation import build_evaluation


@TRAINER_REGISTRY.register()
class DefaultTrainer(TrainerBase):
    '''
        A trainer with default training logic.
    '''

    def __init__(self, configs: DictConfigs, logger: Logging):
        super().__init__()

        self.logger = logger
        self.evaluation = build_evaluation(configs)

        # Build for trainer
        model = self.build_model(configs)
        criterion = self.build_criterion(configs)
        optimizer = self.build_optimizer(configs, model)
        lr_scheduler = self.build_lr_scheduler(configs, optimizer)
        self._trainer = SimpleTrainer(
            configs=configs,
            logger=self.logger,
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )

        # Build data-loader
        self.train_loader = self.build_data_loader(configs, is_train=True)
        self.test_loader = self.build_data_loader(configs, is_train=False)

        # Settings
        self.iter = 0
        self.start_epoch = 0
        self.max_epoch = configs["SOLVER"]["EPOCHS"]
        self.device = configs["MODEL"]["DEVICE"]
        self.save_epochs = configs["SOLVER"]["CHECKPOINT_PERIOD"]

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        self._trainer.before_train()
        self.logger.debug('Train Start!')

    def before_step(self):
        super().before_step()
        self.epoch_data = copy.deepcopy(self.train_loader)

    def run_step(self):
        for iter, (img, mask) in enumerate(self.epoch_data):
            img = img.to(self.device)
            mask = mask.to(self.device)
            self._trainer.run_step(x=img, target=mask)
            self.iter += iter

    def after_step(self):
        save = False
        losses = 0.
        gt_list, fuse_list = [], []
        self.eval_data = copy.deepcopy(self.test_loader)
        if self.epoch % self.save_epochs == 0:
            for idx, (img, mask) in enumerate(self.eval_data):
                img = img.to(self.device)
                mask = mask.to(self.device)
                if idx == len(self.test_loader)-1:
                    save = True
                loss, fuse_mask = self._trainer.after_step(x=img, target=mask, save=save)
                gt_list.append(mask), fuse_list.append(fuse_mask)
                losses += loss

        eval_results = self.evaluation(
            gt_masks=torch.cat(gt_list),
            pred_masks=torch.cat(fuse_list)
        ).calculate_f1score()
        self.logger.debug(f'[{self.epoch}/{self.max_epoch}] VALIDATION LOSS :: {losses / len(self.test_loader)}')
        self.logger.debug(f'[{self.epoch}/{self.max_epoch}] VALIDATION F1 SCORE :: {eval_results}')

    def after_train(self):
        self._trainer.after_train()
        self.logger.debug('Train Finish!')

    @classmethod
    def build_model(cls, configs: DictConfigs):
        # It now calls :func: 'u2net.modeling.build_model'.
        return build_model(configs)

    @classmethod
    def build_criterion(cls, configs: DictConfigs):
        # It now calls :func: 'u2net.modeling.build_criterion'.
        return build_criterion(configs)

    @classmethod
    def build_optimizer(cls, configs: DictConfigs, model):
        # It now calls :func: 'u2net.solver.build_optimizer'.
        return build_optimizer(configs, model)

    @classmethod
    def build_lr_scheduler(cls, configs: DictConfigs, optimizer):
        # It now calls :func: 'u2net.solver.build_lr_scheduler'.
        return build_lr_scheduler(configs, optimizer)

    @classmethod
    def build_data_loader(cls, configs: DictConfigs, is_train: bool):
        # It now calls :func: 'u2net.data.build_loader'.
        return build_loader(configs, is_train=is_train)
