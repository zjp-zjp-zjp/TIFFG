import torch.nn as nn

from . import base
from ..base.modules import Activation
import torch.nn.functional as F
from typing import Optional
import torch

# See change_detection_pytorch/losses
# class JaccardLoss(base.Loss):
#
#     def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.jaccard(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )
#
#
# class DiceLoss(base.Loss):
#
#     def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.f_score(
#             y_pr, y_gt,
#             beta=self.beta,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )


class FocalLoss(base.Loss):
    __name__ = "FocalLoss"

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.celoss=nn.CrossEntropyLoss()
    def forward(self, inputs, targets):
        BCE_loss = self.celoss(inputs,targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
class L1Loss(nn.L1Loss, base.Loss):
    pass

class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
