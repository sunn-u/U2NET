import torch

'''

1. precision-recall curves
2. maximal F-measure
3. Mean Absolute Error
4. weighted F-measure
5. structure measure
6. relaxed F-measure of boundary

'''


class Measures:
    def __init__(self, gt_masks, pred_masks):
        self.gt_masks = gt_masks
        self.pred_masks = pred_masks

        # todo : 왜....?
        self.smooth = 0.0005
        self.batch = gt_masks.shape[0]
        self.shape = gt_masks.shape[2]
        self.thresholds = torch.linspace(0, 1 - self.smooth, self.shape)

        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()

    def calculate_precision(self):
        # precision : tp / tp + fp

        precision = torch.zeros(self.batch, self.shape)
        for i in range(self.shape):
            # 각 픽셀값이 thresholds[i] 보다 크면 True 아니면 False
            # .float() 를 통해 True/False 를 1과 0으로 변경
            pred_mask = (self.pred_masks > self.thresholds[i]).float()
            # pred_mask 와 gt_mask 를 곱하는 이유는 binary 이기 때문에 서로의 픽셀값이 같지 않으면 0이 되기에
            # todo : 근데 배경을 배경으로 맞춘 경우에는 맞췄어도 둘다 0 이라서 sum 에 영향을 안 미치는데, 이건 배경이라 상관없는건가
            # 2차원과 3차원에 해당하는 height 와 width 에 대해서 sum
            # 그 결과로 torch.Size([batch_size, 3]) 출력
            tp = torch.sum(pred_mask * self.gt_masks, dim=[2, 3])
            # pred_mask 는 그 자체가 tp + fp
            results = (tp / (torch.sum(pred_mask, dim=[2, 3]) + self.smooth)).squeeze(-1)
            results = torch.nan_to_num(results, nan=0, posinf=0, neginf=0)
            precision[:, i] = results

        return precision.mean(dim=[0, 1]).detach().cpu().numpy().tolist()

    def calculate_recall(self):
        # recall : tp / tp + fn

        recall = torch.zeros(self.batch, self.shape)
        for i in range(self.shape):
            pred_mask = (self.pred_masks > self.thresholds[i]).float()
            tp = torch.sum(pred_mask * self.gt_masks, dim=[2, 3])
            # gt_masks 는 그 자체가 tp + fn
            results = (tp / (torch.sum(self.gt_masks, dim=[2, 3]) + self.smooth)).squeeze(-1)
            results = torch.nan_to_num(results, nan=0, posinf=0, neginf=0)
            recall[:, i] = results

        return recall.mean(dim=[0, 1]).detach().cpu().numpy().tolist()

    def calculate_f1score(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
