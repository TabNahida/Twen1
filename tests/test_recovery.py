from __future__ import annotations

import pytest
import torch

from twen.recovery import _stable_digest


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.int64, torch.bfloat16),
)
def test_stable_digest_supports_scalar_tensors(dtype: torch.dtype) -> None:
    first = torch.tensor(1, dtype=dtype)
    same = torch.tensor(1, dtype=dtype)
    different = torch.tensor(2, dtype=dtype)

    assert _stable_digest(first) == _stable_digest(same)
    assert _stable_digest(first) != _stable_digest(different)
