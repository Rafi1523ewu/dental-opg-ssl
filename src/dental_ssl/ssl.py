"""Unlabeled-data objectives for multi-label classification: FixMatch and Mean Teacher."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def fixmatch_targets(
    logits_weak: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-label pseudo-labels from the weak view (Sohn et al., 2020, adapted to sigmoids).

    A label is used only when the model is confident either way: probability at least
    `threshold` (positive) or at most `1 - threshold` (negative).
    """
    probs = logits_weak.detach().sigmoid()
    targets = (probs >= 0.5).float()
    mask = ((probs >= threshold) | (probs <= 1 - threshold)).float()
    return targets, mask


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).mean()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Tarvainen & Valpola (2017): teacher weights track an EMA of the student."""
    for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
        t.lerp_(s, 1.0 - decay)
    for t, s in zip(teacher.buffers(), student.buffers(), strict=True):
        t.copy_(s)


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_logits.sigmoid(), teacher_logits.detach().sigmoid())


def sigmoid_rampup(step: int, length: int) -> float:
    if length <= 0 or step >= length:
        return 1.0
    phase = 1.0 - step / length
    return math.exp(-5.0 * phase * phase)
