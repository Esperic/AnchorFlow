#!/usr/bin/env python
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import sklearn
import torch
from sklearn.cluster import KMeans


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.anchorflow.anchors import (
    ANCHOR_FAMILIES,
    ANCHOR_FAMILY_OBJECT_TYPES,
    build_anchor_artifact,
    compute_residual_scale,
    save_anchor_artifact,
)
from src.model.anchorflow.matching import hard_match_anchors


def _load_cache(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_complete_futures(
    cache_dir: Path,
    actor_family: str,
    future_steps: int = 60,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if actor_family not in ANCHOR_FAMILIES:
        raise ValueError(
            f"actor_family must be one of {ANCHOR_FAMILIES}"
        )
    allowed_object_types = set(ANCHOR_FAMILY_OBJECT_TYPES[actor_family])
    cache_files = sorted(Path(cache_dir).rglob("*.pt"))
    futures = []
    skipped_partial = 0
    skipped_missing = 0
    skipped_other_families = 0
    cache_metadata_hashes = set()
    for cache_file in cache_files:
        sample = _load_cache(cache_file)
        if sample.get("metadata_hash") is not None:
            cache_metadata_hashes.add(str(sample["metadata_hash"]))
        if (
            sample.get("y") is None
            or "x_padding_mask" not in sample
            or "x_attr" not in sample
        ):
            skipped_missing += 1
            continue
        raw_actor_type = int(sample["x_attr"][0, 0])
        if raw_actor_type not in allowed_object_types:
            skipped_other_families += 1
            continue
        focal_future = sample["y"][0, :future_steps]
        focal_mask = sample["x_padding_mask"][0, 50 : 50 + future_steps]
        if (
            focal_future.shape != (future_steps, 2)
            or focal_mask.shape != (future_steps,)
            or focal_mask.any()
        ):
            skipped_partial += 1
            continue
        if not torch.isfinite(focal_future).all():
            skipped_partial += 1
            continue
        futures.append(focal_future.to(torch.float32))
    if not futures:
        raise ValueError(
            f"no complete focal futures found under {Path(cache_dir)}"
        )
    return torch.stack(futures), {
        "cache_files": len(cache_files),
        "complete_futures": len(futures),
        "skipped_partial_futures": skipped_partial,
        "skipped_missing_futures": skipped_missing,
        "skipped_other_families": skipped_other_families,
        "cache_metadata_hashes": sorted(cache_metadata_hashes),
    }


def fit_sorted_kmeans(
    futures: torch.Tensor,
    num_modes: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> torch.Tensor:
    if futures.ndim != 3 or futures.shape[-1] != 2:
        raise ValueError("futures must have shape [S, T, 2]")
    if futures.shape[0] < num_modes:
        raise ValueError(
            f"need at least {num_modes} samples, got {futures.shape[0]}"
        )
    estimator = KMeans(
        n_clusters=num_modes,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="lloyd",
    )
    estimator.fit(futures.to(torch.float64).reshape(futures.shape[0], -1).numpy())
    centers = torch.from_numpy(estimator.cluster_centers_).to(torch.float32)
    centers = centers.view(num_modes, futures.shape[1], 2)
    order = sorted(
        range(num_modes),
        key=lambda index: tuple(centers[index].flatten().tolist()),
    )
    return centers[order].contiguous()


def default_visualization_path(output_path: Path) -> Path:
    return Path(output_path).with_suffix(".png")


def visualize_anchors(
    anchors: torch.Tensor,
    actor_family: str,
    output_path: Path,
) -> Path:
    if anchors.ndim != 3 or anchors.shape[-1] != 2:
        raise ValueError("anchors must have shape [K, T, 2]")
    if not torch.isfinite(anchors).all():
        raise ValueError("anchors must contain only finite values")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    try:
        colors = plt.get_cmap("tab10")
        anchor_array = anchors.detach().cpu().numpy()
        for mode_index, trajectory in enumerate(anchor_array):
            color = colors(mode_index % 10)
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                linewidth=2,
                label=f"Mode {mode_index}",
            )
            axis.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=[color],
                marker="o",
                s=30,
            )
            axis.annotate(
                str(mode_index),
                (trajectory[-1, 0], trajectory[-1, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
            )
        axis.scatter(
            0.0,
            0.0,
            color="black",
            marker="x",
            s=60,
            label="Origin",
        )
        axis.set_title(
            f"AV2 {actor_family} anchors ({anchors.shape[0]} modes)"
        )
        axis.set_xlabel("Longitudinal displacement (m)")
        axis.set_ylabel("Lateral displacement (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path


def generate_artifact(
    cache_dir: Path,
    output_path: Path,
    actor_family: str,
    num_modes: int = 6,
    future_steps: int = 60,
    seed: int = 2333,
    n_init: int = 20,
    max_iter: int = 300,
    residual_scale_min: float = 0.5,
    overwrite: bool = False,
) -> Dict:
    futures, collection_report = collect_complete_futures(
        cache_dir,
        actor_family=actor_family,
        future_steps=future_steps,
    )
    anchors = fit_sorted_kmeans(
        futures,
        num_modes=num_modes,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
    )
    valid_mask = torch.ones(
        futures.shape[:2],
        dtype=torch.bool,
    )
    matched_mode, _ = hard_match_anchors(anchors, futures, valid_mask)
    references = anchors[matched_mode]
    residual_scale = compute_residual_scale(
        futures.to(torch.float64),
        references.to(torch.float64),
        valid_mask,
        minimum=residual_scale_min,
    )
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {
        "dataset": "av2",
        "split": "train",
        "actor_family": actor_family,
        "object_type_ids": list(
            ANCHOR_FAMILY_OBJECT_TYPES[actor_family]
        ),
        "num_modes": num_modes,
        "future_steps": future_steps,
        "seed": seed,
        "num_samples": futures.shape[0],
        "kmeans": {
            "n_init": n_init,
            "max_iter": max_iter,
            "algorithm": "lloyd",
            "sklearn_version": sklearn.__version__,
        },
        "residual_scale": {
            "statistic": "population_std",
            "matching": "hard_masked_mean_l2",
            "minimum": residual_scale_min,
        },
        "cache_dir": str(Path(cache_dir).resolve()),
        "collection": collection_report,
        "generator_sha256": script_hash,
    }
    artifact = build_anchor_artifact(anchors, residual_scale, metadata)
    save_anchor_artifact(output_path, artifact, overwrite=overwrite)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic AV2 static motion anchors."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--actor-family",
        choices=ANCHOR_FAMILIES,
        required=True,
    )
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--residual-scale-min", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--visualization-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = generate_artifact(
        cache_dir=args.cache_dir,
        output_path=args.output,
        actor_family=args.actor_family,
        num_modes=args.num_modes,
        future_steps=args.future_steps,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
        residual_scale_min=args.residual_scale_min,
        overwrite=args.overwrite,
    )
    visualization_path = (
        args.visualization_output
        if args.visualization_output is not None
        else default_visualization_path(args.output)
    )
    visualize_anchors(
        artifact["anchors"],
        actor_family=args.actor_family,
        output_path=visualization_path,
    )
    report = {
        "output": str(args.output),
        "visualization_output": str(visualization_path),
        "content_hash": artifact["content_hash"],
        "anchor_shape": list(artifact["anchors"].shape),
        "residual_scale": artifact["residual_scale"].tolist(),
        "metadata": artifact["metadata"],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
