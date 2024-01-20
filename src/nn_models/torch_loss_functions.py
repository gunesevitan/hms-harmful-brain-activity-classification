import torch.nn as nn


class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):

        super(KLDivLossWithLogits).__init__(reduction='batch_mean')

    def forward(self, inputs, targets):

        inputs = nn.functional.log_softmax(inputs,  dim=1)
        loss = super().forward(inputs, targets)

        return loss
