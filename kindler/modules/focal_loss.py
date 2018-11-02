"""
Custom Focal loss which
"""
from __future__ import absolute_import
from __future__ import division

import torch


class FocalLoss(torch.nn.Module):
    """
    Focal loss for focusing on hard negatives during classification training

    Expects targets of values 0 and 1 and predictions of values between 0 and 1
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        # Orig implementation
        bce = torch.nn.functional.binary_cross_entropy(output, target, reduction='none')

        alpha_factor = target.clone()
        alpha_factor[target == 1] = self.alpha
        alpha_factor[target != 1] = 1 - self.alpha

        focal_weight = output.clone()
        focal_weight[target == 1] = 1 - output[target == 1]

        cls_loss = bce * alpha_factor * focal_weight ** self.gamma

        if self.reduction == 'none':
            return cls_loss

        elif self.reduction == 'sum':
            return torch.sum(cls_loss)

        elif self.reduction == 'elementwise_mean':
            return torch.sum(cls_loss) / cls_loss.numel()
