"""Lean, torch-only 2D RRDB generator (serving path).

This is a faithful, dependency-light transcription of the generator used by
``vjnetworks.Pix2PixRRDB`` (only torch, no MONAI/generative/kornia), so the
CBCT->CT model can be served without the heavy research stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return torch.cat([x, out], 1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_dense_layers):
        super(ResidualDenseBlock, self).__init__()
        self.dense_layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_dense_layers):
            self.dense_layers.append(DenseBlock(current_channels, growth_rate))
            current_channels += growth_rate
        self.conv1x1 = nn.Conv2d(current_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        for layer in self.dense_layers:
            x = layer(x)
        return self.conv1x1(x) + identity


class RRDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_dense_layers, num_rdb=3):
        super(RRDB, self).__init__()
        self.rdb_layers = nn.ModuleList(
            ResidualDenseBlock(in_channels, growth_rate, num_dense_layers) for _ in range(num_rdb)
        )
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        for layer in self.rdb_layers:
            x = layer(x)
        out = self.conv_final(x)
        return out * 0.2 + identity


class SEBlock2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RRDBGenerator(nn.Module):
    """2D bilateral RRDB generator (single-channel in/out by default)."""

    def __init__(self, in_channels=1, out_channels=1, num_rrdb=23,
                 num_dense_layers=3, growth_rate=32, feature_channels=64,
                 use_se=False):
        super(RRDBGenerator, self).__init__()
        self.use_se = use_se
        if use_se:
            self.se_input = SEBlock2D(in_channels)
            self.se_initial = SEBlock2D(feature_channels)
            self.se_trunk = SEBlock2D(feature_channels)

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.rrdb_trunk = nn.Sequential(
            *[RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3)
              for _ in range(num_rrdb)]
        )

        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        if self.use_se:
            x = self.se_input(x)
        initial_features = self.conv_in(x)
        if self.use_se:
            initial_features = self.se_initial(initial_features)
        trunk_features = self.rrdb_trunk(initial_features)
        trunk_output = self.conv_trunk(trunk_features)
        if self.use_se:
            trunk_output = self.se_trunk(trunk_output)
        return self.conv_out(initial_features + trunk_output)


def build_generator(cfg) -> RRDBGenerator:
    """Build an :class:`RRDBGenerator` from a :class:`CBCTDenoiseConfig`."""
    return RRDBGenerator(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_rrdb=cfg.num_rrdb_G,
        num_dense_layers=cfg.num_dense_layers_G,
        growth_rate=cfg.growth_rate_G,
        feature_channels=cfg.feature_channels_G,
        use_se=cfg.use_se,
    )
