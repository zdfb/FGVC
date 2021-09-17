import torch
import torch.nn as nn
import torch.nn.functional as F

###### Function：Labelsmooth ######

SMOOTHING = 0.4


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smooth=SMOOTHING):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (1.0 - self.smooth) * \
            one_hot_label + self.smooth / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss
