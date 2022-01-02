# Coding by SunWoo(tjsntjsn20@gmail.com)

import torch

from fvcore.common.registry import Registry

EVALUTATION_REGISTRY = Registry("EVALUATION")
EVALUTATION_REGISTRY.__doc__ = " Registry for Evaluation "

'''

1. precision-recall curves
2. maximal F-measure
3. Mean Absolute Error
4. weighted F-measure
5. structure measure
6. relaxed F-measure of boundary

'''

@EVALUTATION_REGISTRY.register()
class F1_SCORE(object):
    def __init__(self, smooth: float):
        self.smooth = smooth

    def __call__(self, gt_masks, pred_masks):
        batch, _, shape = gt_masks.shape
        thresholds = torch.linspace(0, 1 - self.smooth, shape)

        precision = self.calculate_precision(batch, shape, thresholds, pred_masks, gt_masks)
        recall = self.calculate_recall(batch, shape, thresholds, pred_masks, gt_masks)
        return 2 * (precision * recall) / (precision + recall)

    def calculate_precision(self, batch, shape, thresholds, pred_masks, gt_masks):
        # precision : tp / tp + fp

        precision = torch.zeros(batch, shape)
        for i in range(shape):
            # 각 픽셀값이 thresholds[i] 보다 크면 True 아니면 False
            # .float() 를 통해 True/False 를 1과 0으로 변경
            pred_mask = (pred_masks > thresholds[i]).float()
            # pred_mask 와 gt_mask 를 곱하는 이유는 binary 이기 때문에 서로의 픽셀값이 같지 않으면 0이 되기에
            # todo : 근데 배경을 배경으로 맞춘 경우에는 맞췄어도 둘다 0 이라서 sum 에 영향을 안 미치는데, 이건 배경이라 상관없는건가
            # 2차원과 3차원에 해당하는 height 와 width 에 대해서 sum
            # 그 결과로 torch.Size([batch_size, 3]) 출력
            tp = torch.sum(pred_mask * gt_masks, dim=[2, 3])
            # pred_mask 는 그 자체가 tp + fp
            results = (tp / (torch.sum(pred_mask, dim=[2, 3]) + self.smooth)).squeeze(-1)
            results = torch.nan_to_num(results, nan=0, posinf=0, neginf=0)
            precision[:, i] = results

        return precision.mean(dim=[0, 1]).detach().cpu().numpy().tolist()

    def calculate_recall(self, batch, shape, thresholds, pred_masks, gt_masks):
        # recall : tp / tp + fn

        recall = torch.zeros(batch, shape)
        for i in range(shape):
            pred_mask = (pred_masks > thresholds[i]).float()
            tp = torch.sum(pred_mask * gt_masks, dim=[2, 3])
            # gt_masks 는 그 자체가 tp + fn
            results = (tp / (torch.sum(gt_masks, dim=[2, 3]) + self.smooth)).squeeze(-1)
            results = torch.nan_to_num(results, nan=0, posinf=0, neginf=0)
            recall[:, i] = results

        return recall.mean(dim=[0, 1]).detach().cpu().numpy().tolist()
