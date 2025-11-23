from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from ..data.datasets import SlidingWindowDataset
from ..data.preprocessing import DatasetBundle
from ..models.masking import GeometricMasker


@dataclass
class DataModule:
    train_loader: DataLoader
    test_loader: DataLoader
    input_dim: int


def create_dataloaders(
    bundle: DatasetBundle,
    batch_size: int,
    mask_cfg: dict,
    device: torch.device,
) -> DataModule:
    masker = GeometricMasker(
        mask_ratio=mask_cfg["mask_ratio"],
        geometric_p=mask_cfg["geometric_p"],
        mask_value=mask_cfg["mask_value"],
    )
    train_dataset = SlidingWindowDataset(
        bundle.train_windows,
        bundle.train_labels,
        geometric_masker=masker,
        augment=True,
        device=device.type,
    )
    test_dataset = SlidingWindowDataset(
        bundle.test_windows,
        bundle.test_labels,
        geometric_masker=None,
        augment=False,
        device=device.type,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = bundle.train_windows.shape[-1]
    return DataModule(train_loader=train_loader, test_loader=test_loader, input_dim=input_dim)

