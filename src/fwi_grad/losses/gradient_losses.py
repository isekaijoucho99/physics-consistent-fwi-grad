"""Supervised direction, amplitude and uncertainty losses in normalized space."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Stage2GradientLoss(nn.Module):
    def __init__(
        self, mu: float = 0.5, lambda_unc: float = 0.1, eps: float = 1e-4,
        detach_err_for_unc: bool = False,
    ):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.mu = mu
        self.lambda_unc = lambda_unc
        self.eps = eps
        self.detach_err_for_unc = detach_err_for_unc

    def forward(self, g_pred: Tensor, sigma: Tensor, g_ref_norm: Tensor) -> dict[str, Tensor]:
        if g_pred.shape != g_ref_norm.shape or sigma.shape != g_pred.shape:
            raise ValueError("prediction, sigma and reference must have the same shape")
        pred, ref = g_pred.flatten(1), g_ref_norm.flatten(1)
        cosine = F.cosine_similarity(pred, ref, dim=1).clamp(-1, 1)
        # Identical images have zero directional loss, including two zero images.
        loss_cos = torch.where((pred == ref).all(dim=1), 0.0, 1 - cosine).mean()
        error = (g_pred - g_ref_norm).abs()
        loss_l1 = error.mean()
        loss_pred = loss_cos + self.mu * loss_l1
        scale = sigma + self.eps
        unc_error = error.detach() if self.detach_err_for_unc else error
        loss_unc = (unc_error / scale + scale.log()).mean()
        return {"loss": loss_pred + self.lambda_unc * loss_unc, "loss_pred": loss_pred,
                "loss_cos": loss_cos, "loss_l1": loss_l1, "loss_unc": loss_unc}
