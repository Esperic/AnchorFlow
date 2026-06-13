from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .flow import LinearFlowSample, build_linear_flow_sample, masked_velocity_mse
from .matching import gather_modes, hard_match_anchors


@dataclass(frozen=True)
class MatchedFlowBatch:
    matched_mode: torch.Tensor
    valid_rows: torch.Tensor
    reference: torch.Tensor
    target_residual: torch.Tensor
    flow: LinearFlowSample


def build_matched_flow_batch(
    anchors: torch.Tensor,
    residual_scale: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    source: Optional[torch.Tensor] = None,
    time: Optional[torch.Tensor] = None,
) -> MatchedFlowBatch:
    with torch.no_grad():
        matched_mode, valid_rows = hard_match_anchors(
            anchors,
            targets,
            valid_mask,
        )
        if anchors.ndim == 3:
            expanded_anchors = anchors.unsqueeze(0).expand(
                targets.shape[0], *anchors.shape
            )
        elif anchors.ndim == 4 and anchors.shape[0] == targets.shape[0]:
            expanded_anchors = anchors
        else:
            raise ValueError(
                "anchors must have shape [K,T,2] or [B,K,T,2]"
            )
        reference = gather_modes(expanded_anchors, matched_mode)
        if residual_scale.shape == (2,):
            scale = residual_scale.view(1, 1, 2)
        elif residual_scale.shape == (targets.shape[0], 2):
            scale = residual_scale[:, None]
        else:
            raise ValueError(
                "residual_scale must have shape [2] or [B,2]"
            )
        target_residual = (targets - reference) / scale
        target_residual = target_residual.masked_fill(
            ~valid_mask.unsqueeze(-1), 0.0
        )
    if source is None:
        source = torch.randn_like(target_residual, dtype=torch.float32)
    if time is None:
        time = torch.rand(
            targets.shape[0],
            device=targets.device,
            dtype=torch.float32,
        )
    flow = build_linear_flow_sample(
        source=source,
        target=target_residual,
        time=time,
        valid_mask=valid_mask,
    )
    return MatchedFlowBatch(
        matched_mode=matched_mode,
        valid_rows=valid_rows,
        reference=reference,
        target_residual=target_residual,
        flow=flow,
    )


def masked_other_agent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("other-agent prediction and target shapes must match")
    if valid_mask.shape != prediction.shape[:-1]:
        raise ValueError("other_valid_mask must have shape [B, N, T]")
    coordinate_mask = valid_mask.unsqueeze(-1).expand_as(prediction)
    if not coordinate_mask.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(
        prediction[coordinate_mask],
        target[coordinate_mask],
        reduction="mean",
    )


def compute_stage3_losses(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    mode_logits: torch.Tensor,
    matched_mode: torch.Tensor,
    focal_valid_mask: torch.Tensor,
    y_hat_others: torch.Tensor,
    y_others: torch.Tensor,
    other_valid_mask: torch.Tensor,
    flow_weight: float = 1.0,
    score_weight: float = 1.0,
    other_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    flow_loss, valid_rows = masked_velocity_mse(
        predicted_velocity,
        target_velocity,
        focal_valid_mask,
    )
    score_loss = F.cross_entropy(
        mode_logits[valid_rows],
        matched_mode[valid_rows],
    )
    other_loss = masked_other_agent_loss(
        y_hat_others,
        y_others,
        other_valid_mask,
    )
    total = (
        float(flow_weight) * flow_loss
        + float(score_weight) * score_loss
        + float(other_weight) * other_loss
    )
    valid_count = valid_rows.sum()
    return {
        "loss": total,
        "flow_loss": flow_loss,
        "score_loss": score_loss,
        "other_loss": other_loss,
        "valid_focal_count": valid_count,
        "invalid_focal_count": valid_rows.numel() - valid_count,
    }
