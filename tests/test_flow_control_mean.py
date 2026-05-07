import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.flow.flow_core import RectifiedFlow


def test_batch_control_mean_broadcasts_per_cell_set():
    flow = RectifiedFlow(SimpleNamespace())
    control = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    actual = flow.batch_control_mean(control)
    expected = control.mean(dim=1, keepdim=True).expand_as(control)

    assert actual.shape == control.shape
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0, 0], actual[0, 1])
    assert torch.equal(actual[1, 0], actual[1, 1])
    assert not torch.equal(actual[0, 0], actual[1, 0])
