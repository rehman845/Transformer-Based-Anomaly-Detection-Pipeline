from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_csv_series(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    return df.values.astype(np.float32)


def load_labels(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    return df.values.squeeze().astype(int)


def sliding_window(
    data: np.ndarray,
    labels: Optional[np.ndarray],
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = (len(data) - window) // stride + 1
    windows = []
    window_labels = []
    for i in range(num_samples):
        start = i * stride
        end = start + window
        windows.append(data[start:end])
        if labels is not None:
            window_labels.append(labels[start:end].max())
        else:
            window_labels.append(0)
    return np.stack(windows), np.array(window_labels)


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        geometric_masker,
        augment: bool,
        device: str = "cpu",
    ) -> None:
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.masker = geometric_masker
        self.augment = augment
        self.device = device

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        label = self.labels[idx]
        if self.augment and self.masker is not None:
            masked, mask = self.masker(window)
        else:
            masked = window.clone()
            mask = torch.zeros_like(window[:, 0], dtype=torch.bool)
        return {
            "data": window.to(self.device),
            "masked": masked.to(self.device),
            "mask": mask.to(self.device),
            "label": label.to(self.device),
        }


def load_smd_dataset(root: str | Path, machine_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    train_path = root / "SMD" / "train" / f"{machine_id}.txt"
    test_path = root / "SMD" / "test" / f"{machine_id}.txt"
    label_path = root / "SMD" / "test_label" / f"{machine_id}.txt"

    train = load_csv_series(train_path)
    test = load_csv_series(test_path)
    labels = load_labels(label_path)

    return train, test, labels

