# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch
import logging

from . import TRAINER_REGISTRY
from .. import hooks
from u2net.engine.train_loop import TrainerBase, SimpleTrainer
from u2net.data.types import DictConfigs
from u2net.data import build_loader
from u2net.utils.checkpoint import CheckPointer
from u2net.modeling import build_model, build_criterion
from u2net.solver import build_optimizer, build_lr_scheduler
from u2net.evaluation import build_evaluation


class Testor(object):
    def __init__(self, configs: DictConfigs, loader, evaluator):
        super().__init__()
        self.configs = configs
        self.loader = loader
        self.evaluator = evaluator

    def __call__(self, model):
        model.eval()

        gt_list, fuse_list = [], []
        with torch.no_grad():
            for data, target in self.loader:
                data = data.to(self.configs.user.model.device)
                _, fuse_mask = model(data)
                gt_list.append(target.to(self.configs.user.model.device))
                fuse_list.append(fuse_mask)

        eval_results = self.evaluator(gt_masks=torch.cat(gt_list), pred_masks=torch.cat(fuse_list))
        return eval_results


@TRAINER_REGISTRY.register()
class DefaultTrainer(TrainerBase):
    '''
        A trainer with default training logic.
    '''

    def __init__(self, configs: DictConfigs):
        super().__init__()

        # Build for trainer
        model = self.build_model(configs)
        criterion = self.build_criterion(configs)
        optimizer = self.build_optimizer(configs, model)
        loader = self.build_data_loader(configs, is_train=True)

        self._trainer = SimpleTrainer(
            configs=configs,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loader=loader
        )

        # Settings
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = configs.user.training.epochs
        self.device = configs.user.model.device
        self.eval_checkpoint_period = configs.user.training.checkpoint_period

        self._checkpointer = CheckPointer(max_epoch=self.max_epoch, save_dir=configs.user.training.output_dir)
        self._testor = Testor(
            configs=configs,
            loader=self.build_data_loader(configs, is_train=False),
            evaluator=self.build_evaluation(configs)
        )
        self.logger = logging.getLogger(__name__)

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        hook_block = [
            hooks.EpochTimer(),
            # for after_step.
            hooks.EvalHook(self._testor, self.eval_checkpoint_period),
            hooks.PeriodicCheckpointer(self._checkpointer, self.eval_checkpoint_period),
            hooks.CommonWriter()
        ]
        return hook_block

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def run_step(self):
        self._trainer.epoch = self.epoch
        self._trainer.run_step()

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
    def build_data_loader(cls, configs: DictConfigs, is_train: bool):
        # It now calls :func: 'u2net.data.build_loader'.
        return build_loader(configs, is_train=is_train)

    @classmethod
    def build_evaluation(cls, configs: DictConfigs):
        # It now calls :func: 'u2net.evaluation.build_evaluation'.
        return build_evaluation(configs)
