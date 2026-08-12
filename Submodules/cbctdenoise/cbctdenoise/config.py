"""Configuration for the CBCT -> CT denoiser.

This replaces the notebook's ``Options()`` class with a typed dataclass. All
hyperparameters that once lived scattered across notebook cells live here and
can be (de)serialized to/from JSON for reproducibility.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CBCTDenoiseConfig:
    # Model architecture (matches Options() in the notebook)
    in_channels: int = 1
    out_channels: int = 1
    num_rrdb_G: int = 9
    num_dense_layers_G: int = 2
    growth_rate_G: int = 32
    feature_channels_G: int = 64
    use_se: bool = False

    # Intensity scaling (HU clipping range)
    hu_clip: Tuple[float, float] = (-1000.0, 3000.0)

    # Inference
    patch_size: Tuple[int, int] = (256, 256)
    overlap: float = 0.66
    plane: str = "axial"                       # default plane
    padding: str = "replicate"
    ensemble_planes: List[str] = field(default_factory=lambda: ["axial"])
    ensemble: str = "none"                     # "none" | "mean" | "fourier_burst"
    ensemble_p: float = 5.0

    # Training (research path)
    num_epochs: int = 100
    learning_rate: float = 2e-4
    batch_size: int = 1
    lambda_gan: float = 1.0
    lambda_ngf: float = 100.0
    lambda_l1_msssim: float = 100.0
    alpha_ngf: float = 0.10

    # Data locations (no hardcoded /amadeo paths; defaults are relative/cwd)
    data_dir: str = "."
    results_dir: str = "."
    experiment_prefix: str = "cbctdenoise"

    def feature_string(self) -> str:
        return (
            f"{self.num_rrdb_G}rrdb-{self.num_dense_layers_G}ndl-"
            f"{self.growth_rate_G}gr-{self.feature_channels_G}fc"
        )

    def checkpoint_name(self, epoch: int) -> str:
        return f"{self.experiment_prefix}_{self.feature_string()}_e{epoch:04d}.h5"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_json(cls, path: str) -> "CBCTDenoiseConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


config = CBCTDenoiseConfig()
