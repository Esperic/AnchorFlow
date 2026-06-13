import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch


ANCHOR_ARTIFACT_VERSION = 1
MIN_RESIDUAL_SCALE = 0.5


def _validate_tensors(
    anchors: torch.Tensor,
    residual_scale: torch.Tensor,
) -> None:
    if anchors.ndim != 3 or anchors.shape[-1] != 2:
        raise ValueError(
            "anchors must have shape [num_modes, future_steps, 2], "
            f"got {tuple(anchors.shape)}"
        )
    if not torch.isfinite(anchors).all():
        raise ValueError("anchors must contain only finite values")
    if residual_scale.shape != (2,):
        raise ValueError(
            f"residual_scale must have shape [2], got {tuple(residual_scale.shape)}"
        )
    if not torch.isfinite(residual_scale).all():
        raise ValueError("residual_scale must contain only finite values")
    if torch.any(residual_scale < MIN_RESIDUAL_SCALE):
        raise ValueError(
            f"residual_scale values must be at least {MIN_RESIDUAL_SCALE}"
        )


def compute_anchor_content_hash(artifact: Dict[str, Any]) -> str:
    anchors = artifact["anchors"].detach().cpu().to(torch.float32).contiguous()
    scale = (
        artifact["residual_scale"]
        .detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
    )
    metadata = json.dumps(
        artifact["metadata"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(str(int(artifact["version"])).encode("ascii"))
    digest.update(str(tuple(anchors.shape)).encode("ascii"))
    digest.update(anchors.numpy().tobytes())
    digest.update(scale.numpy().tobytes())
    digest.update(metadata)
    return digest.hexdigest()


def build_anchor_artifact(
    anchors: torch.Tensor,
    residual_scale: torch.Tensor,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    anchors = anchors.detach().cpu().to(torch.float32).contiguous()
    residual_scale = (
        residual_scale.detach().cpu().to(torch.float32).contiguous()
    )
    _validate_tensors(anchors, residual_scale)
    if metadata.get("split") != "train":
        raise ValueError("anchor metadata split must be 'train'")

    artifact = {
        "version": ANCHOR_ARTIFACT_VERSION,
        "anchors": anchors,
        "residual_scale": residual_scale,
        "metadata": dict(metadata),
    }
    artifact["content_hash"] = compute_anchor_content_hash(artifact)
    return artifact


def save_anchor_artifact(
    path: Path,
    artifact: Dict[str, Any],
    overwrite: bool = False,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"anchor artifact already exists: {path}")
    validated = build_anchor_artifact(
        anchors=artifact["anchors"],
        residual_scale=artifact["residual_scale"],
        metadata=artifact["metadata"],
    )
    if artifact.get("content_hash") not in (None, validated["content_hash"]):
        raise ValueError("anchor artifact content_hash does not match its content")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(validated, path)


def load_anchor_artifact(
    path: Path,
    expected_num_modes: int = None,
    expected_future_steps: int = None,
) -> Dict[str, Any]:
    path = Path(path)
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    required = {
        "version",
        "anchors",
        "residual_scale",
        "metadata",
        "content_hash",
    }
    if not isinstance(artifact, dict) or set(artifact) != required:
        raise ValueError(
            f"anchor artifact must contain exactly {sorted(required)}"
        )
    if artifact["version"] != ANCHOR_ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported anchor artifact version {artifact['version']}"
        )
    _validate_tensors(artifact["anchors"], artifact["residual_scale"])
    if artifact["metadata"].get("split") != "train":
        raise ValueError("anchor artifact must be generated from the train split")
    expected_hash = compute_anchor_content_hash(artifact)
    if artifact["content_hash"] != expected_hash:
        raise ValueError(
            "anchor artifact content hash mismatch: "
            f"expected {expected_hash}, got {artifact['content_hash']}"
        )
    anchors = artifact["anchors"]
    if expected_num_modes is not None and anchors.shape[0] != expected_num_modes:
        raise ValueError(
            f"expected {expected_num_modes} anchors, got {anchors.shape[0]}"
        )
    if (
        expected_future_steps is not None
        and anchors.shape[1] != expected_future_steps
    ):
        raise ValueError(
            f"expected {expected_future_steps} future steps, got {anchors.shape[1]}"
        )
    return artifact


def compute_residual_scale(
    targets: torch.Tensor,
    references: torch.Tensor,
    valid_mask: torch.Tensor,
    minimum: float = MIN_RESIDUAL_SCALE,
) -> torch.Tensor:
    if targets.shape != references.shape:
        raise ValueError("targets and references must have identical shape")
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError("targets and references must have shape [B, T, 2]")
    if valid_mask.shape != targets.shape[:2]:
        raise ValueError("valid_mask must have shape [B, T]")
    residual = (targets - references).to(torch.float64)
    valid_coordinates = valid_mask.unsqueeze(-1).expand_as(residual)
    if not valid_mask.any():
        raise ValueError("cannot compute residual scale without valid futures")
    values = residual[valid_coordinates].view(-1, 2)
    scale = values.std(dim=0, unbiased=False)
    scale = scale.clamp_min(float(minimum))
    if not torch.isfinite(scale).all():
        raise ValueError("computed residual scale is not finite")
    return scale.to(torch.float32)

