from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .datasets import load_smd_dataset, sliding_window


@dataclass
class DatasetBundle:
    train_windows: np.ndarray
    train_labels: np.ndarray
    test_windows: np.ndarray
    test_labels: np.ndarray
    scaler: StandardScaler


def prepare_dataset(
    root_dir: str,
    dataset_name: str,
    machine_id: str,
    sequence_length: int,
    stride: int,
    normalize: bool = True,
) -> DatasetBundle:
    dataset_name = dataset_name.lower()
    if dataset_name != "smd":
        raise NotImplementedError("Currently only the SMD dataset is implemented in this template.")

    train, test, labels = load_smd_dataset(root_dir, machine_id)
    scaler = StandardScaler()
    if normalize:
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

    train_windows, train_labels = sliding_window(train, None, sequence_length, stride)
    test_windows, test_labels = sliding_window(test, labels, sequence_length, stride)

    return DatasetBundle(
        train_windows=train_windows,
        train_labels=train_labels,
        test_windows=test_windows,
        test_labels=test_labels,
        scaler=scaler,
    )

