"""Cleaned GAN training loop (research path, requires the ``[train]`` extra).

This is a faithful-but-factored skeleton of the notebook's training loop. The
goal here is a *clean, readable, configurable* trainer; the exact GAN/NGF/
MSSSIM recipe is preserved in :mod:`cbctdenoise.losses`.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from cbctdenoise.config import CBCTDenoiseConfig
from cbctdenoise.losses import NGF, l1_loss


def train_once(model, batch, optimizer, cfg, losses, device):
    model.train()
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)

    optimizer.zero_grad()
    fake = model.generator_A_to_B(src)
    loss = torch.tensor(0.0, device=device)

    if cfg.lambda_l1_msssim > 0:
        loss = loss + cfg.lambda_l1_msssim * losses["content"](fake, tgt)
    if cfg.lambda_ngf > 0:
        loss = loss + cfg.lambda_ngf * losses["ngf"](fake, tgt)
    loss.backward()
    optimizer.step()
    return float(loss)


def train(cfg: CBCTDenoiseConfig, weights_dir: str, device: Optional[str] = None,
          resume: Optional[str] = None):
    # NOTE: this builds the lean generator only. For the full GAN training
    # (generator + discriminator + NGF/MSSSIM) use the vendored vjnetworks
    # package and the [train] extra - see README.
    from cbctdenoise.models.rrdb import RRDBGenerator  # noqa: PLC0415

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = RRDBGenerator(
        in_channels=cfg.in_channels, out_channels=cfg.out_channels,
        num_rrdb=cfg.num_rrdb_G, num_dense_layers=cfg.num_dense_layers_G,
        growth_rate=cfg.growth_rate_G, feature_channels=cfg.feature_channels_G).to(device)
    if resume:
        cp = torch.load(resume, map_location=device)
        model.load_state_dict(cp["model"], strict=False)

    opt_g = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    losses = {"ngf": NGF(alpha=cfg.alpha_ngf), "content": l1_loss}

    os.makedirs(weights_dir, exist_ok=True)
    for epoch in range(cfg.num_epochs):
        # NOTE: plug your DataLoader here (see data.SliceDataset).
        pass
    model.eval()