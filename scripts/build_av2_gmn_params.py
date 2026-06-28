#!/usr/bin/env python3
"""
Build DriftTraj-compatible GMN parameters from Forecast-MAE AV2 caches.

Expected input:
  data_root/forecast-mae/train/*.pt

Each cache file must contain ``y`` with shape [N, T, 2]. This script uses
``y[0]`` because the single-agent Forecast-MAE baseline predicts the focal
agent only.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build AV2 GMN Params")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--data-folder", type=str, default="forecast-mae")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm", type=str, default="none", choices=["none", "mean_range"])
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--std-floor", type=float, default=0.03)
    parser.add_argument("--x-weight", type=float, default=1.0)
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def load_cache(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_focal_futures(cache_dir: Path, max_samples: int = 0) -> np.ndarray:
    files = sorted(cache_dir.glob("*.pt"))
    if max_samples > 0:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .pt cache files found in {cache_dir}")

    futures = []
    for path in files:
        sample = load_cache(path)
        y = sample.get("y")
        if y is None:
            continue
        if not isinstance(y, torch.Tensor) or y.dim() != 3 or y.size(-1) != 2:
            raise ValueError(f"{path} has invalid y shape: {getattr(y, 'shape', None)}")
        focal = y[0]
        if torch.isfinite(focal).all():
            futures.append(focal.float().numpy())

    if not futures:
        raise ValueError(f"No valid focal future trajectories found in {cache_dir}")
    return np.stack(futures).astype(np.float32)


def weighted_features(trajs: np.ndarray, x_weight: float) -> np.ndarray:
    weighted = trajs.copy()
    weighted[..., 0] *= float(x_weight)
    return weighted.reshape(weighted.shape[0], -1)


def run_minibatch_kmeans(
    features: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if features.shape[0] < k:
        raise ValueError(f"Not enough samples for k={k}: {features.shape[0]}")

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        batch_size=8192,
        reassignment_ratio=0.01,
    )
    model.fit(features)
    return model.labels_.astype(np.int64, copy=False), model.cluster_centers_.astype(np.float32), float(model.inertia_)


def stable_sort_by_endpoint(anchors: np.ndarray, counts: np.ndarray) -> np.ndarray:
    endpoints = anchors[:, -1, :]
    angles = np.arctan2(endpoints[:, 1], endpoints[:, 0])
    dists = np.linalg.norm(endpoints, axis=1)
    return np.lexsort((-counts, dists, angles))


def compute_gmn_stats(
    trajs: np.ndarray,
    assignments: np.ndarray,
    cluster_trajs: np.ndarray,
    std_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = cluster_trajs.shape[0]
    counts = np.bincount(assignments, minlength=k).astype(np.int64)
    center_points = np.zeros((k, 2), dtype=np.float32)
    center_std = np.zeros((k, 2), dtype=np.float32)

    for cid in range(k):
        if counts[cid] > 0:
            vals = trajs[assignments == cid].reshape(-1, 2)
            center_points[cid] = vals.mean(axis=0)
            center_std[cid] = vals.std(axis=0)
        else:
            vals = cluster_trajs[cid].reshape(-1, 2)
            center_points[cid] = vals.mean(axis=0)
            center_std[cid] = np.array([std_floor, std_floor], dtype=np.float32)

    center_std = np.maximum(center_std, float(std_floor)).astype(np.float32)
    mixture_weights = (counts / max(int(counts.sum()), 1)).astype(np.float32)
    return center_points, center_std, mixture_weights, counts


def compute_mean_range_stats(trajs: np.ndarray, eps: float):
    flat = trajs.reshape(-1, 2)
    mean = flat.mean(axis=0).astype(np.float32)
    vmin = flat.min(axis=0).astype(np.float32)
    vmax = flat.max(axis=0).astype(np.float32)
    value_range = np.maximum(vmax - mean, mean - vmin).astype(np.float32)
    value_range = np.maximum(value_range, float(eps)).astype(np.float32)
    return mean, value_range, vmin, vmax


def normalize_mean_range(trajs: np.ndarray, mean: np.ndarray, value_range: np.ndarray):
    return ((trajs - mean[None, None, :]) / value_range[None, None, :]).astype(np.float32)


def denormalize_mean_range(trajs: np.ndarray, mean: np.ndarray, value_range: np.ndarray):
    return (trajs * value_range[None, None, :] + mean[None, None, :]).astype(np.float32)


def build_gmn_from_cache(
    cache_dir: Path,
    output: Path,
    k: int = 6,
    max_samples: int = 0,
    n_init: int = 5,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
    norm: str = "none",
    norm_eps: float = 1e-6,
    std_floor: float = 0.03,
    x_weight: float = 1.0,
    save_json: bool = False,
) -> Path:
    cache_dir = Path(cache_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    trajs = extract_focal_futures(cache_dir=cache_dir, max_samples=max_samples)
    num_samples, future_steps, coord_dim = trajs.shape
    if coord_dim != 2:
        raise ValueError(f"Expected coord dim 2, got {coord_dim}")
    norm = str(norm).lower()
    if norm not in ("none", "mean_range"):
        raise ValueError(f"Unsupported norm: {norm}")

    if norm == "mean_range":
        norm_mean, norm_range, norm_min, norm_max = compute_mean_range_stats(
            trajs,
            eps=norm_eps,
        )
        trajs_for_clustering = normalize_mean_range(
            trajs,
            mean=norm_mean,
            value_range=norm_range,
        )
    else:
        norm_mean = norm_range = norm_min = norm_max = None
        trajs_for_clustering = trajs

    features = weighted_features(trajs_for_clustering, x_weight=x_weight)
    assignments, centers_weighted, inertia = run_minibatch_kmeans(
        features=features,
        k=k,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    )

    cluster_trajs = centers_weighted.reshape(k, future_steps, 2)
    cluster_trajs[..., 0] /= float(x_weight)
    center_points, center_std, mixture_weights, counts = compute_gmn_stats(
        trajs=trajs_for_clustering,
        assignments=assignments,
        cluster_trajs=cluster_trajs,
        std_floor=std_floor,
    )

    if norm == "mean_range":
        cluster_trajs_raw = denormalize_mean_range(
            cluster_trajs,
            mean=norm_mean,
            value_range=norm_range,
        )
        center_points_raw = (center_points * norm_range[None, :] + norm_mean[None, :]).astype(np.float32)
        center_std_raw = (center_std * norm_range[None, :]).astype(np.float32)
        sort_trajs = cluster_trajs_raw
    else:
        cluster_trajs_raw = center_points_raw = center_std_raw = None
        sort_trajs = cluster_trajs

    order = stable_sort_by_endpoint(sort_trajs, counts)
    cluster_trajs = cluster_trajs[order]
    center_points = center_points[order]
    center_std = center_std[order]
    mixture_weights = mixture_weights[order]
    counts = counts[order]
    if norm == "mean_range":
        cluster_trajs_raw = cluster_trajs_raw[order]
        center_points_raw = center_points_raw[order]
        center_std_raw = center_std_raw[order]

    metadata = {
        "dataset": "av2",
        "cache_dir": str(cache_dir),
        "k": int(k),
        "t": int(future_steps),
        "coord_dim": 2,
        "repr": "xy",
        "norm": norm,
        "seed": int(seed),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "std_floor": float(std_floor),
        "norm_eps": float(norm_eps),
        "x_weight": float(x_weight),
        "used_for_gmn": int(num_samples),
        "inertia": float(inertia),
        "agent_centric_assumed": True,
        "focal_agent_only": True,
        "backend": "sklearn",
    }
    if norm == "mean_range":
        metadata.update(
            {
                "norm_mean": [float(norm_mean[0]), float(norm_mean[1])],
                "norm_range": [float(norm_range[0]), float(norm_range[1])],
                "norm_min": [float(norm_min[0]), float(norm_min[1])],
                "norm_max": [float(norm_max[0]), float(norm_max[1])],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = {
        "cluster_trajs": cluster_trajs.astype(np.float32),
        "anchors": cluster_trajs.astype(np.float32),
        "center_points": center_points.astype(np.float32),
        "center_std": center_std.astype(np.float32),
        "mixture_weights": mixture_weights.astype(np.float32),
        "proportions": mixture_weights.astype(np.float32),
        "counts": counts.astype(np.int64),
        "order": order.astype(np.int64),
        "metadata_json": json.dumps(metadata, ensure_ascii=True),
    }
    if norm == "mean_range":
        output_fields.update(
            {
                "cluster_trajs_raw": cluster_trajs_raw.astype(np.float32),
                "center_points_raw": center_points_raw.astype(np.float32),
                "center_std_raw": center_std_raw.astype(np.float32),
                "norm_mean": norm_mean.astype(np.float32),
                "norm_range": norm_range.astype(np.float32),
                "norm_min": norm_min.astype(np.float32),
                "norm_max": norm_max.astype(np.float32),
            }
        )
    np.savez_compressed(output, **output_fields)

    if save_json:
        json_summary = {
            **metadata,
            "counts": counts.tolist(),
            "mixture_weights": mixture_weights.tolist(),
            "center_points": center_points.tolist(),
            "center_std": center_std.tolist(),
        }
        if norm == "mean_range":
            json_summary.update(
                {
                    "center_points_raw": center_points_raw.tolist(),
                    "center_std_raw": center_std_raw.tolist(),
                }
            )
        with open(output.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)

    return output


def main() -> None:
    args = parse_args()
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(args.data_root) / args.data_folder / args.split

    if args.output:
        output = Path(args.output)
    else:
        repr_name = "mean_range" if args.norm == "mean_range" else "xy"
        output = Path("anchors") / "gmn" / f"av2_gmn_{repr_name}_k{args.k}_t60.npz"

    out_path = build_gmn_from_cache(
        cache_dir=cache_dir,
        output=output,
        k=args.k,
        max_samples=args.max_samples,
        n_init=args.n_init,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        norm=args.norm,
        norm_eps=args.norm_eps,
        std_floor=args.std_floor,
        x_weight=args.x_weight,
        save_json=args.save_json,
    )
    print(f"[Done] saved GMN params to: {out_path}")


if __name__ == "__main__":
    main()
