import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch


ANCHOR_ARTIFACT_VERSION = 1
MIN_RESIDUAL_SCALE = 0.5
ANCHOR_FAMILIES = ("vehicle", "cyclist", "pedestrian")
ANCHOR_FAMILY_OBJECT_TYPES = {
    "vehicle": (0, 4),
    "cyclist": (2, 3),
    "pedestrian": (1,),
}
_FAMILY_INDEX = {
    family: index for index, family in enumerate(ANCHOR_FAMILIES)
}


@dataclass(frozen=True)
class AnchorSelection:
    anchors: torch.Tensor
    residual_scales: torch.Tensor
    family_index: torch.Tensor
    fallback_mask: torch.Tensor
    map_applicable_mask: torch.Tensor


@dataclass(frozen=True)
class AnchorBank:
    anchors: torch.Tensor
    residual_scales: torch.Tensor
    metadata: Dict[str, Dict[str, Any]]
    content_hashes: Dict[str, str]

    def select(self, raw_actor_types: torch.Tensor) -> AnchorSelection:
        family_index, fallback_mask, map_applicable_mask = (
            actor_types_to_anchor_family(raw_actor_types)
        )
        return AnchorSelection(
            anchors=self.anchors.to(raw_actor_types.device)[family_index],
            residual_scales=self.residual_scales.to(raw_actor_types.device)[
                family_index
            ],
            family_index=family_index,
            fallback_mask=fallback_mask,
            map_applicable_mask=map_applicable_mask,
        )


def actor_types_to_anchor_family(
    raw_actor_types: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_actor_types = raw_actor_types.long()
    pedestrian_index = _FAMILY_INDEX["pedestrian"]
    family_index = torch.full_like(raw_actor_types, pedestrian_index)

    vehicle_mask = (raw_actor_types == 0) | (raw_actor_types == 4)
    cyclist_mask = (raw_actor_types == 2) | (raw_actor_types == 3)
    pedestrian_mask = raw_actor_types == 1
    riderless_bicycle_mask = raw_actor_types == 8

    family_index[vehicle_mask] = _FAMILY_INDEX["vehicle"]
    family_index[cyclist_mask | riderless_bicycle_mask] = _FAMILY_INDEX[
        "cyclist"
    ]
    supported_mask = vehicle_mask | cyclist_mask | pedestrian_mask
    fallback_mask = ~supported_mask
    map_applicable_mask = vehicle_mask | cyclist_mask
    return family_index, fallback_mask, map_applicable_mask


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
    actor_family = metadata.get("actor_family")
    if actor_family not in ANCHOR_FAMILIES:
        raise ValueError(
            f"anchor metadata actor_family must be one of {ANCHOR_FAMILIES}"
        )
    object_type_ids = tuple(metadata.get("object_type_ids", ()))
    if object_type_ids != ANCHOR_FAMILY_OBJECT_TYPES[actor_family]:
        raise ValueError(
            f"actor_family {actor_family!r} requires object_type_ids "
            f"{ANCHOR_FAMILY_OBJECT_TYPES[actor_family]}, got {object_type_ids}"
        )

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
    actor_family = artifact["metadata"].get("actor_family")
    if actor_family not in ANCHOR_FAMILIES:
        raise ValueError("anchor artifact has invalid actor_family metadata")
    object_type_ids = tuple(artifact["metadata"].get("object_type_ids", ()))
    if object_type_ids != ANCHOR_FAMILY_OBJECT_TYPES[actor_family]:
        raise ValueError("anchor artifact object_type_ids do not match actor_family")
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


def load_anchor_bank(
    anchor_paths: Mapping[str, str],
    expected_num_modes: int,
    expected_future_steps: int,
) -> AnchorBank:
    if set(anchor_paths) != set(ANCHOR_FAMILIES):
        raise ValueError(
            f"anchor_paths must contain exactly {ANCHOR_FAMILIES}, "
            f"got {tuple(anchor_paths)}"
        )
    artifacts = {}
    for family in ANCHOR_FAMILIES:
        artifact = load_anchor_artifact(
            anchor_paths[family],
            expected_num_modes=expected_num_modes,
            expected_future_steps=expected_future_steps,
        )
        if artifact["metadata"]["actor_family"] != family:
            raise ValueError(
                f"anchor path for {family!r} contains "
                f"{artifact['metadata']['actor_family']!r} artifact"
            )
        artifacts[family] = artifact
    return AnchorBank(
        anchors=torch.stack(
            [artifacts[family]["anchors"] for family in ANCHOR_FAMILIES]
        ),
        residual_scales=torch.stack(
            [
                artifacts[family]["residual_scale"]
                for family in ANCHOR_FAMILIES
            ]
        ),
        metadata={
            family: artifacts[family]["metadata"] for family in ANCHOR_FAMILIES
        },
        content_hashes={
            family: artifacts[family]["content_hash"]
            for family in ANCHOR_FAMILIES
        },
    )


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
