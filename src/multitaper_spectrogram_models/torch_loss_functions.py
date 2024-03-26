import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class KLDivWithLogitsLoss(nn.KLDivLoss):

    def __init__(self):

        super().__init__(reduction='batchmean')

    def forward(self, inputs, targets):

        inputs = nn.functional.log_softmax(inputs, dim=1)
        loss = super().forward(inputs, targets)

        return loss


class WeightedKLDivWithLogitsLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):

        super(WeightedKLDivWithLogitsLoss, self).__init__(weight=weight, reduction=reduction)

        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):

        inputs = nn.functional.log_softmax(inputs, dim=1)
        loss = F.kl_div(inputs, targets, reduction='none')
        loss = loss * self.weight

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class LabelSmoothingBCEWithLogitsLoss(_WeightedLoss):

    def __init__(self, weight=None, smoothing_factor=0.0, reduction='mean'):

        super(LabelSmoothingBCEWithLogitsLoss, self).__init__(weight=weight, reduction=reduction)

        self.smoothing_factor = smoothing_factor
        self.weight = weight
        self.reduction = reduction

    def _smooth_labels(self, targets):

        with torch.no_grad():
            targets = targets * (1.0 - self.smoothing_factor) + 0.5 * self.smoothing_factor

        return targets

    def forward(self, inputs, targets):

        targets = self._smooth_labels(targets)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
