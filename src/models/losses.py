from __future__ import annotations

import torch
from torch import nn


def info_nce_loss(
    projections_1: torch.Tensor,
    projections_2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    projections_1 = nn.functional.normalize(projections_1, dim=-1)
    projections_2 = nn.functional.normalize(projections_2, dim=-1)
    logits = projections_1 @ projections_2.T / temperature
    labels = torch.arange(len(projections_1), device=projections_1.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return (loss + nn.functional.cross_entropy(logits.T, labels)) * 0.5

