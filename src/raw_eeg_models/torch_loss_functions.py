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
