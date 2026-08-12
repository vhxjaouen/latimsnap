"""Serving wrapper referenced by the ``latimsnap-i2i`` model spec.

The server's generic loader instantiates ``cls(**kwargs)`` and calls
``net(x)`` where ``x`` is a 2D patch ``(B, C, H, W)``. This wrapper exposes the
raw :class:`~cbctdenoise.models.rrdb.RRDBGenerator` through that interface.
"""

from __future__ import annotations

import torch.nn as nn

from cbctdenoise.models.rrdb import RRDBGenerator


class MRToCTGenerator(nn.Module):
    """Thin wrapper so the generic server can drive the RRDB generator."""

    def __init__(self, in_channels=1, out_channels=1, num_rrdb_G=9,
                 num_dense_layers_G=2, growth_rate_G=32,
                 feature_channels_G=64, use_se=False):
        super(MRToCTGenerator, self).__init__()
        self.generator = RRDBGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G,
            growth_rate=growth_rate_G,
            feature_channels=feature_channels_G,
            use_se=use_se,
        )

    def forward(self, x):
        # x: (B, C, H, W) 2D
        return self.generator(x)
