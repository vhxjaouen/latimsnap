"""Learning losses extracted from the notebook's training loop.

Only the pieces needed to reproduce training are here (imported lazily so the
serving path never pulls in MONAI/generative/kornia).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def identity_loss(original_images: torch.Tensor, translated_images: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(original_images - translated_images))


def l1_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(real - fake))


def l1_ssim_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Combined pixel L1 + SSIM term (kornia-based, training only)."""
    import kornia  # noqa: PLC0415
    ssim = kornia.metrics.ssim(real, fake, window_size=11)
    return (1.0 - ssim).mean()


def l1_msssim_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    import kornia  # noqa: PLC0415
    msssim = kornia.metrics.ssim3d(real, fake) if real.ndim == 5 else kornia.metrics.ssim(real, fake)
    return (1.0 - msssim).mean()


class NGF(torch.nn.Module):
    """Normalized Gradient Field similarity loss (extracted from the notebook)."""

    def __init__(self, alpha: float = 0.25, eps: float = 1e-6):
        super(NGF, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, mov: torch.Tensor, fix: torch.Tensor) -> torch.Tensor:
        import kornia.filters as K  # noqa: PLC0415
        gx = K.spatial_gradient(mov)
        gy = K.spatial_gradient(fix)
        a = self.alpha
        gx_n = gx / (torch.sqrt(torch.sum(gx ** 2, dim=1, keepdim=True) + a ** 2 * self.eps ** 2) + self.eps)
        gy_n = gy / (torch.sqrt(torch.sum(gy ** 2, dim=1, keepdim=True) + a ** 2 * self.eps ** 2) + self.eps)
        return 1.0 - torch.mean(torch.abs(gx_n - gy_n))


def build_losses(cfg):
    """Return loss functions (adversarial + NGF + content) for training."""
    from generative.losses import PatchAdversarialLoss  # noqa: PLC0415
    adv = PatchAdversarialLoss(criterion="bce")
    ngf = NGF(alpha=cfg.alpha_ngf)
    content = l1_msssim_loss if cfg.lambda_l1_msssim > 0 else l1_loss
    return {"adversarial": adv, "ngf": ngf, "content": content}
