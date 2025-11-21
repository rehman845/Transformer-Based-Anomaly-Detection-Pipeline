from __future__ import annotations

import math
import torch


class GeometricMasker:
    def __init__(
        self,
        mask_ratio: float = 0.2,
        geometric_p: float = 0.2,
        mask_value: float = 0.0,
    ) -> None:
        self.mask_ratio = mask_ratio
        self.geometric_p = geometric_p
        self.mask_value = mask_value

    def __call__(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = window.size(0)
        num_masks = max(1, int(self.mask_ratio * seq_len))
        mask = torch.zeros(seq_len, dtype=torch.bool)
        masked_window = window.clone()

        lengths = torch.distributions.geometric.Geometric(torch.tensor(self.geometric_p)).sample((num_masks,))
        lengths = torch.clamp(lengths, 1, max(1, seq_len // 2))

        for length in lengths.int():
            length_value = min(seq_len, length.item())
            max_start = max(1, seq_len - length_value + 1)
            start = torch.randint(0, max_start, (1,)).item() if max_start > 1 else 0
            end = min(seq_len, start + length_value)
            mask[start:end] = True
            masked_window[start:end] = self.mask_value

        return masked_window, mask

