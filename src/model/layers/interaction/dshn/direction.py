import math

import torch


def encode_directionality(direction: torch.Tensor, q: float) -> torch.Tensor:
    """Map tail (+1) and head (-1) incidences to complex unit weights."""
    phase = torch.polar(
        torch.ones((), device=direction.device, dtype=direction.dtype),
        torch.as_tensor(2.0 * math.pi * q, device=direction.device),
    )
    ones = torch.ones_like(direction, dtype=phase.dtype)
    return torch.where(direction == -1, phase, ones)
