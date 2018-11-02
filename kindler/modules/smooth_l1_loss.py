"""
Custom SmoothL1Loss which accepts a beta value (equivalent to 1 / sigma_sq)
to control the threshold where loss switches between smooth and l1
"""
from __future__ import absolute_import
from __future__ import division

import torch


class SmoothL1Loss(torch.nn.Module):
    """
    Also knownn as Huber loss

    loss is given by
    f(x) = 0.5 * x ^ 2 / beta       if |x| < beta
           |x| - 0.5 * beta         otherwise
    where x is the difference between prediction and target

    Accepts 3 types of reduction/normalization
        'none' returns the loss for each element kept in their original shape
        'sum' gets the sum of all loss
        'elementwise_mean' gets the average of every element
    """
    def __init__(self, beta=1, reduction='elementwise_mean'):
        assert reduction in {'none', 'sum', 'elementwise_mean'}, \
            "reduction must have a value in ['none', 'sum', 'elementwise_mean']"

        self.beta = beta
        self.reduction = reduction

    def forward(self, output, target):
        diff = torch.abs(output - target)

        smooth_ids = diff < self.beta
        diff[smooth_ids]  = 0.5 * diff[smooth_ids] ** 2 / self.beta
        diff[~smooth_ids] = diff[~smooth_ids] - 0.5 * self.beta

        if self.reduction == 'none':
            return diff

        elif self.reduction == 'sum':
            return torch.sum(diff)

        elif self.reduction == 'elementwise_mean':
            return torch.sum(diff) / diff.numel()
