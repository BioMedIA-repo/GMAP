from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after


class DiceFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    Supports pixel-wise weighting instead of masking.
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Callable | None = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: str = "mean",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            gamma: float = 2.0,
            focal_weight: Sequence[float] | float | int | torch.Tensor | None = None,
            weight: Sequence[float] | float | int | torch.Tensor | None = None,
            lambda_dice: float = 1.0,
            lambda_focal: float = 1.0,
    ) -> None:
        """
        Initialize DiceFocalLoss with options for Dice loss and Focal loss.
        """
        super().__init__()
        weight = focal_weight if focal_weight is not None else weight
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction="none",  # Use "none" to apply pixel-wise weights later
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            weight=weight,
            reduction="none",  # Use "none" to apply pixel-wise weights later
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y
        self.reduction = reduction

    def forward(
            self, input: torch.Tensor, target: torch.Tensor, pixel_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits.
            target: the shape should be BNH[WD] or B1H[WD]. Target tensor with -1 as invalid labels.
            pixel_weights: a weight tensor of the same shape as `target`, with values for pixel-wise weighting.

        Returns:
            A tensor representing the combined loss, reduced based on the specified reduction method.
        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

       
        valid_mask = target != -1 

        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        eps = 1e-8  # Prevent division by zero
        # Apply pixel-wise weighting if provided
        if pixel_weights is not None:
            if pixel_weights.shape != target.shape:
                raise ValueError(f"Pixel weights shape {pixel_weights.shape} must match target shape {target.shape}.")

            # Apply pixel-wise weighting and compute losses
            dice_loss = (dice_loss * pixel_weights * valid_mask).sum() / (valid_mask.sum() + eps)
            focal_loss = (focal_loss * pixel_weights * valid_mask).sum() / (valid_mask.sum() + eps)
        else:
            # If no pixel-wise weights, just use valid_mask and compute losses
            dice_loss = (dice_loss * valid_mask).sum() / (valid_mask.sum() + eps)
            focal_loss = (focal_loss * valid_mask).sum() / (valid_mask.sum() + eps)

        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss
