# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch.nn as nn
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = " Registry for Loss "


@LOSS_REGISTRY.register()
class BinaryCrossEntropy(object):
    def __init__(self):
        self.criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

    def __call__(self, gt_mask, pred_masks, fuse_mask):
        loss = 0.
        for pred_mask in pred_masks:
            loss += self.criterion(pred_mask, gt_mask)
        loss += self.criterion(fuse_mask, gt_mask)
        return loss
