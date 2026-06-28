#!/usr/bin/env python3
"""
Visualize DriftTraj-compatible GMN parameters generated from AV2 caches.

The figure follows DriftTraj's visualizer:
1) clustered future trajectories,
2) Gaussian point clouds sampled from center_points/center_std.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize AV2 GMN Params")
    parser.add_argument("--gmn", type=str, required=True, help="Path to GMN .npz file")
    parser.add_argument("--output", type=str, default="", help="Output figure path")
    parser.add_argument("--space", type=str, default="raw", choices=["raw", "norm"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--fig-width", type=float, default=12.5)
    parser.add_argument("--fig-height", type=float, default=5.2)
    parser.add_argument("--samples-per-comp", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _field(pack, key: str, fallback: str = None):
    if key in pack:
        return pack[key].astype(np.float32, copy=False)
    if fallback is not None and fallback in pack:
        return pack[fallback].astype(np.float32, copy=False)
    missing = key if fallback is None else f"{key} or {fallback}"
    raise KeyError(f"Missing field in GMN file: {missing}")


def load_gmn(path: Path, space: str = "raw") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    pack = np.load(Path(path).expanduser(), allow_pickle=False)
    space = str(space).lower()
    if space not in ("raw", "norm"):
        raise ValueError(f"Unsupported space: {space}")

    if space == "raw":
        cluster_trajs = _field(pack, "cluster_trajs_raw", fallback="cluster_trajs")
        center_points = _field(pack, "center_points_raw", fallback="center_points")
        center_std = _field(pack, "center_std_raw", fallback="center_std")
    else:
        cluster_trajs = _field(pack, "cluster_trajs")
        center_points = _field(pack, "center_points")
        center_std = _field(pack, "center_std")

    if "mixture_weights" in pack:
        mixture_weights = pack["mixture_weights"].astype(np.float32, copy=False)
    elif "proportions" in pack:
        mixture_weights = pack["proportions"].astype(np.float32, copy=False)
    elif "counts" in pack:
        counts = pack["counts"].astype(np.float32, copy=False)
        mixture_weights = counts / max(float(counts.sum()), 1.0)
    else:
        k = cluster_trajs.shape[0]
        mixture_weights = np.full((k,), 1.0 / max(k, 1), dtype=np.float32)

    metadata = {}
    if "metadata_json" in pack:
        try:
            metadata = json.loads(str(pack["metadata_json"]))
        except Exception:
            metadata = {}

    if cluster_trajs.ndim != 3 or cluster_trajs.shape[-1] != 2:
        raise ValueError(f"cluster_trajs must be [K, T, 2], got {cluster_trajs.shape}")
    if center_points.shape != (cluster_trajs.shape[0], 2):
        raise ValueError(f"center_points must be [K, 2], got {center_points.shape}")
    if center_std.shape != (cluster_trajs.shape[0], 2):
        raise ValueError(f"center_std must be [K, 2], got {center_std.shape}")

    return cluster_trajs, center_points, center_std, mixture_weights, metadata


def compute_axis_limits(points: np.ndarray, margin_ratio: float = 0.08):
    x = points[..., 0]
    y = points[..., 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    x_pad = max(0.3, x_span * margin_ratio)
    y_pad = max(0.3, y_span * margin_ratio)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def make_output_path(gmn_path: Path, output: Path = None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    out_dir = Path("visualization") / "anchors"
    return (out_dir / f"{Path(gmn_path).stem}_gmn_vis.png").resolve()


def visualize_gmn_file(
    gmn_path: Path,
    output: Path = None,
    space: str = "raw",
    dpi: int = 180,
    fig_width: float = 12.5,
    fig_height: float = 5.2,
    samples_per_comp: int = 2000,
    seed: int = 42,
    title: str = "",
    show: bool = False,
) -> Path:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("matplotlib is required for GMN visualization") from exc

    gmn_path = Path(gmn_path).expanduser().resolve()
    if not gmn_path.exists():
        raise FileNotFoundError(f"GMN file not found: {gmn_path}")

    rng = np.random.RandomState(seed)
    cluster_trajs, center_points, center_std, mixture_weights, metadata = load_gmn(
        gmn_path,
        space=space,
    )
    k, t, _ = cluster_trajs.shape
    order = np.argsort(-mixture_weights)

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    ax_traj, ax_noise = axes
    cmap = plt.cm.get_cmap("tab10", max(k, 10))
    xlim, ylim = compute_axis_limits(cluster_trajs)

    for rank, cid in enumerate(order):
        traj = cluster_trajs[cid]
        weight = float(mixture_weights[cid])
        color = cmap(rank)
        linewidth = 1.8 + 2.8 * weight
        alpha = 0.62 + 0.33 * weight
        label = f"C{cid} ({weight * 100:.1f}%)"

        ax_traj.plot(traj[:, 0], traj[:, 1], color=color, lw=linewidth, alpha=alpha, label=label)
        ax_traj.scatter(traj[0, 0], traj[0, 1], color=color, s=18, alpha=0.9, marker="o")
        ax_traj.scatter(traj[-1, 0], traj[-1, 1], color=color, s=30, alpha=0.95, marker="*")

    ax_traj.set_title("Cluster Trajectories")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_xlim(*xlim)
    ax_traj.set_ylim(*ylim)
    ax_traj.grid(alpha=0.25)
    ax_traj.legend(loc="upper left", fontsize=8, frameon=True)

    sampled_clouds = []
    for rank, cid in enumerate(order):
        mean = center_points[cid]
        std = center_std[cid]
        weight = float(mixture_weights[cid])
        color = cmap(rank)

        pts = rng.randn(samples_per_comp, 2).astype(np.float32)
        pts = pts * std[None, :] + mean[None, :]
        sampled_clouds.append(pts)

        alpha_pts = 0.1 + min(0.12, 0.25 * weight)
        ax_noise.scatter(pts[:, 0], pts[:, 1], s=12, color=color, alpha=alpha_pts, edgecolors="none")
        ax_noise.scatter(mean[0], mean[1], s=56, color=color, edgecolors="black", linewidths=0.8, marker="*", zorder=5)

    ax_noise.set_title("Gaussian Noise Point Clouds")
    ax_noise.set_xlabel("X")
    ax_noise.set_ylabel("Y")
    if sampled_clouds:
        noise_points = np.concatenate([np.concatenate(sampled_clouds, axis=0), cluster_trajs.reshape(-1, 2)], axis=0)
        noise_xlim, noise_ylim = compute_axis_limits(noise_points)
        ax_noise.set_xlim(*noise_xlim)
        ax_noise.set_ylim(*noise_ylim)
    ax_noise.grid(alpha=0.25)

    if title:
        fig_title = title
    else:
        dataset = str(metadata.get("dataset", "unknown"))
        norm = str(metadata.get("norm", "none"))
        fig_title = f"GMN Visualization | dataset={dataset} | K={k}, T={t} | norm={norm} | space={space}"
    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = make_output_path(gmn_path, output=output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    out_path = visualize_gmn_file(
        gmn_path=Path(args.gmn),
        output=Path(args.output) if args.output else None,
        space=args.space,
        dpi=args.dpi,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        samples_per_comp=args.samples_per_comp,
        seed=args.seed,
        title=args.title,
        show=args.show,
    )
    print(f"[Done] saved GMN visualization: {out_path}")
    print(f"[Info] samples_per_comp={args.samples_per_comp}")


if __name__ == "__main__":
    main()
