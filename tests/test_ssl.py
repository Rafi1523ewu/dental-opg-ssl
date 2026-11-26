import torch
from torch import nn

from dental_ssl import ssl


def test_fixmatch_uses_only_confident_labels_in_either_direction():
    logits = torch.logit(torch.tensor([[0.99, 0.5, 0.01]]))
    targets, mask = ssl.fixmatch_targets(logits, threshold=0.95)
    assert targets.tolist() == [[1.0, 1.0, 0.0]]
    assert mask.tolist() == [[1.0, 0.0, 1.0]]


def test_masked_bce_is_zero_when_nothing_passes():
    logits = torch.randn(4, 3)
    loss = ssl.masked_bce(logits, torch.zeros(4, 3), torch.zeros(4, 3))
    assert loss.item() == 0.0


def test_ema_update_moves_teacher_toward_student():
    student, teacher = nn.Linear(2, 1), nn.Linear(2, 1)
    nn.init.ones_(student.weight)
    nn.init.zeros_(teacher.weight)
    ssl.ema_update(teacher, student, decay=0.9)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.1))


def test_consistency_loss_is_zero_for_identical_predictions():
    logits = torch.randn(4, 3)
    assert ssl.consistency_loss(logits, logits.clone()).item() == 0.0


def test_sigmoid_rampup_bounds():
    assert ssl.sigmoid_rampup(0, 100) < 0.01
    assert ssl.sigmoid_rampup(100, 100) == 1.0
    assert ssl.sigmoid_rampup(5, 0) == 1.0
