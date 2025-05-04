import numpy as np
from torch import nn
import math
import torch
import torch.nn.functional as F


class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1=0.2, lambda_2=0.05, cumpet_ce_loss=False, start_label=0):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cumpet_ce_loss = cumpet_ce_loss
        self.start_label = start_label

    def forward(self, input, target_label, years_last_followup, weights=None):
        class_dim = input.shape[-1]
        batch_size = input.shape[0]

        target_ = target_label.detach()
        target_[target_ > (class_dim - 1)] = class_dim - 1
        mask = 1 - ((target_.cpu() == (class_dim - 1)) & (years_last_followup.cpu() < (class_dim - 1))).int()
        count = sum(mask)
        if sum(mask) > 0:
            input = input[mask == 1, ...]
            target_ = target_[mask == 1, ...]
            years_last_followup = years_last_followup[mask == 1, ...]

            p = F.softmax(input, dim=-1)
            # mean loss
            a = torch.arange(class_dim, dtype=torch.float32).cuda()
            target = target_.cuda()
            mean = torch.squeeze((p * (a + self.start_label)).sum(1, keepdim=True), dim=1)
            mse = (mean - target)**2
            # new_weights = torch.zeros_like(target)
            # weights = None
            if weights is not None:
                weights_ = torch.tensor(weights, dtype=torch.float).view(1, -1)
                weights_ = weights_.repeat([count,1])
                weights_ = weights_[range(count), target_.cpu()].cuda()
                mean_loss = sum(mse * weights_) / sum(weights_) / 2.0
                # mean_loss = (mse * weights_).mean() / 2.0

                b = (a[None, :] - mean[:, None]) ** 2
                # variance_loss = ((p * b).sum(1, keepdim=False) * weights_).mean()
                variance_loss = sum((p * b).sum(1, keepdim=False) * weights_) / sum(weights_)
            else:
                mean_loss = mse.mean() / 2.0
                # variance loss
                b = (a[None, :] - mean[:, None]) ** 2
                variance_loss = (p * b).sum(1, keepdim=True).mean()
        else:
            mean_loss = torch.tensor(0.0).cuda()
            variance_loss = torch.tensor(0.0).cuda()

        loss = (self.lambda_1 * mean_loss) + (self.lambda_2 * variance_loss)
        return loss
