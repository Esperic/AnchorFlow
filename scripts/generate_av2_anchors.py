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
    future_steps: int = 60,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    cache_files = sorted(Path(cache_dir).rglob("*.pt"))
    futures = []
    skipped_partial = 0
    skipped_missing = 0
    cache_metadata_hashes = set()
    for cache_file in cache_files:
        sample = _load_cache(cache_file)
        if sample.get("metadata_hash") is not None:
            cache_metadata_hashes.add(str(sample["metadata_hash"]))
        if sample.get("y") is None or "x_padding_mask" not in sample:
            skipped_missing += 1
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


def generate_artifact(
    cache_dir: Path,
    output_path: Path,
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
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--residual-scale-min", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = generate_artifact(
        cache_dir=args.cache_dir,
        output_path=args.output,
        num_modes=args.num_modes,
        future_steps=args.future_steps,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
        residual_scale_min=args.residual_scale_min,
        overwrite=args.overwrite,
    )
    report = {
        "output": str(args.output),
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
