from typing import Tuple

import torch


def masked_anchor_distances(
    anchors: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if anchors.ndim not in (3, 4) or anchors.shape[-1] != 2:
        raise ValueError("anchors must have shape [K, T, 2] or [B, K, T, 2]")
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError("targets must have shape [B, T, 2]")
    if anchors.ndim == 3 and anchors.shape[1:] != targets.shape[1:]:
        raise ValueError("anchors and targets must share [T, 2]")
    if anchors.ndim == 4 and (
        anchors.shape[0] != targets.shape[0]
        or anchors.shape[2:] != targets.shape[1:]
    ):
        raise ValueError("batched anchors must have shape [B, K, T, 2]")
    if valid_mask.shape != targets.shape[:2]:
        raise ValueError("valid_mask must have shape [B, T]")

    anchor_batch = anchors[None] if anchors.ndim == 3 else anchors
    point_distance = torch.linalg.vector_norm(
        targets[:, None] - anchor_batch,
        dim=-1,
    )
    weights = valid_mask[:, None].to(point_distance.dtype)
    counts = weights.sum(dim=-1)
    distances = (point_distance * weights).sum(dim=-1)
    distances = distances / counts.clamp_min(1)
    return distances


def hard_match_anchors(
    anchors: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    distances = masked_anchor_distances(anchors, targets, valid_mask)
    valid_rows = valid_mask.any(dim=-1)
    matched = distances.argmin(dim=-1)
    matched = torch.where(valid_rows, matched, torch.zeros_like(matched))
    return matched, valid_rows


def soft_match_anchors(
    anchors: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    distances = masked_anchor_distances(anchors, targets, valid_mask)
    valid_rows = valid_mask.any(dim=-1)
    probabilities = torch.softmax(-distances / temperature, dim=-1)
    probabilities = probabilities * valid_rows[:, None]
    return probabilities, valid_rows


def gather_modes(values: torch.Tensor, mode_index: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have a mode dimension")
    if mode_index.shape != (values.shape[0],):
        raise ValueError("mode_index must have shape [B]")
    batch_index = torch.arange(values.shape[0], device=values.device)
    return values[batch_index, mode_index]
