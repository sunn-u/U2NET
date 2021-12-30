# Coding by SunWoo(tjsntjsn20@gmail.com)

import time
import os
import torch

from models.build import build_model
from datasets.build import build_train_dataset
from build.optimizer import build_optimizer
from build.loss import Loss
from utils.logger import set_logger
from evaluation.measures import Measures


class Trainer:
    def __init__(self, cfg):

        self.model = build_model(cfg).to(cfg.MODEL.DEVICE)
        self.optimizer = build_optimizer(cfg, self.model.parameters())
        self.criterion = Loss(loss_func=cfg.LOSS.FUNCTION)
        self.train_loader, self.val_loader = build_train_dataset(cfg)

        self.cfg = cfg
        self.logger = set_logger(cfg.DATA.OUTPUT_DIR)
        self.checkpoint = -1

        self.write_cfg()

    def write_cfg(self):
        start_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
        self.logger.debug(f'TRAINING START AT {start_time}')
        self.logger.debug(self.cfg)

    def save_model(self, epoch, eval_results):
        model_dir = os.path.join(self.cfg.DATA.OUTPUT_DIR, f'{str(epoch).zfill(5)}.pth')
        torch.save(self.model.state_dict(), model_dir)

        if eval_results >= self.checkpoint:
            best_model_dir = os.path.join(self.cfg.DATA.OUTPUT_DIR, f'best_model.pth')
            torch.save(self.model.state_dict(), best_model_dir)
            self.checkpoint = eval_results

    def train(self):
        for epoch in range(self.cfg.TRAIN.EPOCHS):

            self.model.train()
            losses = 0.

            for img, mask in self.train_loader:
                img = img.to(self.cfg.MODEL.DEVICE)
                gt_mask = mask.to(self.cfg.MODEL.DEVICE)

                pred_masks, fuse_mask = self.model(img)
                loss = self.criterion.loss_function(gt_mask=gt_mask, pred_masks=pred_masks, fuse_mask=fuse_mask)
                loss.backward()

                self.optimizer.step()
                losses += loss.item()

            losses /= len(self.train_loader)
            self.logger.debug(f'[{epoch}/{self.cfg.TRAIN.EPOCHS}] TRAIN LOSS :: {losses}')

            ###########################################
            self.model.eval()
            losses = 0.
            gt_list, fuse_list = [], []

            for img, mask in self.val_loader:
                img = img.to(self.cfg.MODEL.DEVICE)
                gt_mask = mask.to(self.cfg.MODEL.DEVICE)

                pred_masks, fuse_mask = self.model(img)
                loss = self.criterion.loss_function(gt_mask=gt_mask, pred_masks=pred_masks, fuse_mask=fuse_mask)
                losses += loss.item()
                gt_list.append(gt_mask)
                fuse_list.append(fuse_mask)

            # todo : Measure 을 초기화자에 적용할 수 있게 - build
            # todo : Measure 보완 + 로그 기록 더 꼼꼼하게
            eval_results = Measures(gt_masks=torch.cat(gt_list), pred_masks=torch.cat(fuse_list)).calculate_f1score()
            losses /= len(self.val_loader)
            self.logger.debug(f'[{epoch}/{self.cfg.TRAIN.EPOCHS}] VALIDATION LOSS :: {losses}')
            self.logger.debug(f'[{epoch}/{self.cfg.TRAIN.EPOCHS}] VALIDATION F1 SCORE :: {eval_results}')
            self.save_model(epoch, eval_results)
