import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class FocalLoss(nn.Module):

    def __init__(self, gamma: int = 2, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean') -> None:
        """
        Focal Loss for addressing class imbalance in classification tasks.

        Args:
            gamma (int): Focusing parameter that controls the strength of the modulating factor (1-p_t).
            alpha (Optional[torch.Tensor]): Class weight factor. If provided, it should be a tensor of shape (num_classes,).
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Focal Loss.

        Args:
            inputs (torch.Tensor): Predictions from the model of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            torch.Tensor: The computed Focal Loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GCE(nn.Module):

    def __init__(self, num_classes: int, q: float = 0.7, label_smoothing: float = 0.0, reduction: str = "mean", class_weights: Optional[torch.Tensor] = None) -> None:
        """
        Generalized Cross Entropy (GCE) Loss.

        Args:
            num_classes (int): Number of classes.
            q (float): Hyperparameter for controlling the robustness of the loss. Default is 0.7.
            label_smoothing (float): Label smoothing factor. Default is 0.0.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            class_weights (Optional[torch.Tensor]): Class weight factors. If provided, it should be a tensor of shape (num_classes,).
        """
        super(GCE, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)
        self.class_weights = self.class_weights.to(device)
    

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized Cross Entropy (GCE) Loss.

        Args:
            pred (torch.Tensor): Predictions from the model of shape (batch_size, num_classes).
            labels (torch.Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            torch.Tensor: The computed GCE Loss.
        """
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(device)
        label_one_hot = (1.0 - self.label_smoothing) * label_one_hot + self.label_smoothing / self.num_classes
        
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        
        weights = self.class_weights[labels].to(device)
        gce = gce * weights
        
        if self.reduction == "mean":
            return gce.mean()
        return gce.sum()