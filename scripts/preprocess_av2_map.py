#!/usr/bin/env python3
import argparse
import concurrent.futures
import importlib
import json
import multiprocessing
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datamodule.av2_extractor_map import (  # noqa: E402
    Av2MapExtractor,
    validate_cache_sample,
)
from src.datamodule.av2_map_utils import (  # noqa: E402
    EDGE_TYPE_MAP,
    local_to_world,
    metadata_hash,
)

_WORKER_EXTRACTOR = None


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {str(key): _json_value(item) for key, item in value.items()}
        if "successful_scenarios" in result:
            result["successful_scenarios"] = sorted(
                result["successful_scenarios"],
                key=lambda item: (
                    item.get("scenario_id", "")
                    if isinstance(item, Mapping)
                    else str(item)
                ),
            )
        if "failed_scenarios" in result:
            result["failed_scenarios"] = sorted(
                result["failed_scenarios"],
                key=lambda item: item.get("scenario_id", ""),
            )
        return result
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                _json_value(payload),
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _compare(expected: Any, actual: Any, path: str, issues: List[str]) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            issues.append(f"{path}: expected tensor, got {type(actual).__name__}")
        elif expected.shape != actual.shape:
            issues.append(
                f"{path}: tensor shape differs "
                f"({tuple(expected.shape)} != {tuple(actual.shape)})"
            )
        elif expected.dtype != actual.dtype:
            issues.append(
                f"{path}: tensor dtype differs " f"({expected.dtype} != {actual.dtype})"
            )
        elif not torch.equal(expected, actual):
            issues.append(f"{path}: tensor values differ")
        return

    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(expected, actual):
            issues.append(f"{path}: array values differ")
        return

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            issues.append(f"{path}: expected mapping, got {type(actual).__name__}")
            return
        expected_keys = set(expected)
        actual_keys = set(actual)
        for missing in sorted(expected_keys - actual_keys):
            issues.append(f"{path}.{missing}: missing from actual")
        for extra in sorted(actual_keys - expected_keys):
            issues.append(f"{path}.{extra}: unexpected in actual")
        for key in sorted(expected_keys & actual_keys):
            child_path = f"{path}.{key}" if path else str(key)
            _compare(expected[key], actual[key], child_path, issues)
        return

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            issues.append(f"{path}: expected sequence, got {type(actual).__name__}")
        elif len(expected) != len(actual):
            issues.append(
                f"{path}: sequence length differs "
                f"({len(expected)} != {len(actual)})"
            )
        else:
            for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
                _compare(
                    expected_item,
                    actual_item,
                    f"{path}[{index}]",
                    issues,
                )
        return

    if expected != actual:
        issues.append(f"{path}: value differs ({expected!r} != {actual!r})")


def compare_samples(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> List[str]:
    issues: List[str] = []
    _compare(expected, actual, "", issues)
    return issues


def _load_cache(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _class_path(extractor_class: type) -> str:
    return f"{extractor_class.__module__}:{extractor_class.__qualname__}"


def _import_class(class_path: str) -> type:
    module_name, qualname = class_path.split(":", maxsplit=1)
    value = importlib.import_module(module_name)
    for attribute in qualname.split("."):
        value = getattr(value, attribute)
    return cast(type, value)


def _cache_matches_metadata(cache_file: Path, expected_hash: str) -> bool:
    if not cache_file.is_file():
        return False
    try:
        sample = _load_cache(cache_file)
    except Exception:
        return False
    return (
        isinstance(sample, Mapping)
        and sample.get("metadata_hash") == expected_hash
        and isinstance(sample.get("metadata"), Mapping)
        and metadata_hash(sample["metadata"]) == expected_hash
    )


def _initialize_preprocess_worker(
    output_dir: str,
    extractor_config: Mapping[str, Any],
    extractor_class_path: str,
) -> None:
    global _WORKER_EXTRACTOR
    extractor_class = _import_class(extractor_class_path)
    _WORKER_EXTRACTOR = extractor_class(
        save_path=Path(output_dir),
        **dict(extractor_config),
    )


def _preprocess_worker(source_file: str) -> Dict[str, Any]:
    source_path = Path(source_file)
    try:
        if _WORKER_EXTRACTOR is None:
            raise RuntimeError("preprocess worker was not initialized")
        output_file = _WORKER_EXTRACTOR.save(source_path)
        return {
            "status": "success",
            "scenario_id": source_path.stem,
            "cache_file": output_file.name,
        }
    except Exception as error:
        return {
            "status": "failed",
            "scenario_id": source_path.stem,
            "source_file": str(source_path),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def preprocess_files(
    source_files: Sequence[Path],
    *,
    output_dir: Path,
    extractor_config: Mapping[str, Any],
    workers: int,
    force: bool = False,
    fail_fast: bool = False,
    progress_every: int = 0,
    progress_label: str = "",
    extractor_class: type = Av2MapExtractor,
) -> Dict[str, List[Dict[str, Any]]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(Path(path) for path in source_files)
    probe_extractor = extractor_class(
        save_path=output_dir,
        **dict(extractor_config),
    )
    expected_hash = probe_extractor.metadata_hash
    extractor_class_path = _class_path(extractor_class)

    pending = []
    skipped = []
    for source_file in source_files:
        cache_file = output_dir / f"{source_file.stem}.pt"
        if not force and _cache_matches_metadata(cache_file, expected_hash):
            skipped.append(
                {
                    "scenario_id": source_file.stem,
                    "cache_file": cache_file.name,
                }
            )
        else:
            pending.append(source_file)

    successful = []
    failed = []
    processed_count = len(skipped)
    total_count = len(source_files)

    def record(result: Mapping[str, Any]) -> bool:
        nonlocal processed_count
        processed_count += 1
        if result["status"] == "success":
            successful.append(
                {
                    "scenario_id": result["scenario_id"],
                    "cache_file": result["cache_file"],
                }
            )
            failed_now = False
        else:
            failed.append(
                {key: value for key, value in result.items() if key != "status"}
            )
            failed_now = True
        if progress_every > 0 and (
            processed_count % progress_every == 0 or processed_count == total_count
        ):
            prefix = f"[{progress_label}] " if progress_label else ""
            print(
                f"{prefix}processed {processed_count}/{total_count} "
                f"(saved={len(successful)}, skipped={len(skipped)}, "
                f"failed={len(failed)})",
                flush=True,
            )
        return failed_now

    if workers == 1:
        _initialize_preprocess_worker(
            str(output_dir),
            extractor_config,
            extractor_class_path,
        )
        for source_file in pending:
            failed_now = record(_preprocess_worker(str(source_file)))
            if failed_now and fail_fast:
                break
    elif pending:
        context = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_preprocess_worker,
            initargs=(
                str(output_dir),
                extractor_config,
                extractor_class_path,
            ),
        )
        pending_iterator = iter(pending)
        futures = {}

        def submit_next() -> bool:
            try:
                source_file = next(pending_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _preprocess_worker,
                str(source_file),
            )
            futures[future] = source_file
            return True

        try:
            for _ in range(min(len(pending), workers * 2)):
                submit_next()
            stop_submitting = False
            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    source_file = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {
                            "status": "failed",
                            "scenario_id": source_file.stem,
                            "source_file": str(source_file),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    failed_now = record(result)
                    if failed_now and fail_fast:
                        stop_submitting = True
                    if not stop_submitting:
                        submit_next()
        finally:
            executor.shutdown(wait=True)

    if (
        progress_every > 0
        and processed_count == total_count
        and not pending
        and skipped
    ):
        prefix = f"[{progress_label}] " if progress_label else ""
        print(
            f"{prefix}processed {processed_count}/{total_count} "
            f"(saved=0, skipped={len(skipped)}, failed=0)",
            flush=True,
        )

    return {
        "successful_scenarios": sorted(
            successful, key=lambda item: item["scenario_id"]
        ),
        "skipped_scenarios": sorted(skipped, key=lambda item: item["scenario_id"]),
        "failed_scenarios": sorted(failed, key=lambda item: item["scenario_id"]),
    }


def _scenario_files(data_root: Path, split: str) -> List[Path]:
    return sorted((Path(data_root) / split).rglob("*.parquet"))


def _limit(items: Sequence[Path], limit: Optional[int]) -> Sequence[Path]:
    if limit is None or limit < 0:
        return items
    return items[:limit]


def run_preprocess(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    output_root = data_root / args.output_folder
    has_failures = False
    workers = getattr(args, "workers", 1)
    force = getattr(args, "force", False)
    progress_every = getattr(args, "progress_every", 100)

    for split in args.splits:
        split_output = output_root / split
        extractor_config = {
            "radius": args.radius,
            "boundary_points": args.boundary_points,
            "mode": split,
            "ignore_type": (5, 6, 7, 8, 9),
            "remove_outlier_actors": True,
        }
        extractor = Av2MapExtractor(
            save_path=split_output,
            **extractor_config,
        )
        files = _limit(_scenario_files(data_root, split), args.limit)
        if not files:
            has_failures = True
            result = {
                "successful_scenarios": [],
                "skipped_scenarios": [],
                "failed_scenarios": [
                    {
                        "scenario_id": split,
                        "source_file": str(data_root / split),
                        "error_type": "NoScenarioFiles",
                        "error": f"no parquet files found for split {split}",
                        "traceback": "",
                    }
                ],
            }
        else:
            result = preprocess_files(
                files,
                output_dir=split_output,
                extractor_config=extractor_config,
                workers=workers,
                force=force,
                fail_fast=args.fail_fast,
                progress_every=progress_every,
                progress_label=split,
            )

        successful = result["successful_scenarios"]
        skipped = result["skipped_scenarios"]
        failed = result["failed_scenarios"]
        processed_count = len(successful) + len(skipped) + len(failed)
        unprocessed_count = len(files) - processed_count
        print(
            f"[{split}] complete: saved={len(successful)}, "
            f"skipped={len(skipped)}, failed={len(failed)}, "
            f"unprocessed={unprocessed_count}",
            flush=True,
        )
        for item in failed:
            print(
                f"[{split}] failed {item['source_file']}: "
                f"{item['error_type']}: {item['error']}",
                file=sys.stderr,
            )
        has_failures = has_failures or bool(failed)

        manifest = {
            "split": split,
            "metadata": extractor.metadata,
            "metadata_hash": extractor.metadata_hash,
            "successful_scenarios": successful,
            "skipped_scenarios": skipped,
            "failed_scenarios": failed,
            "source_count": len(files),
            "success_count": len(successful),
            "skipped_count": len(skipped),
            "failure_count": len(failed),
            "unprocessed_count": unprocessed_count,
            "workers": workers,
            "force": force,
        }
        write_json_atomic(split_output / "manifest.json", manifest)
        if failed and args.fail_fast:
            break
    return int(has_failures)


def _point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)[:, :2]
    polyline = np.asarray(polyline, dtype=np.float64)[:, :2]
    starts = polyline[:-1]
    ends = polyline[1:]
    segments = ends - starts
    squared_lengths = np.sum(segments * segments, axis=1)
    if not len(segments) or np.all(squared_lengths == 0):
        return np.linalg.norm(points - polyline[0], axis=1)

    offsets = points[:, None, :] - starts[None, :, :]
    projection = np.divide(
        np.sum(offsets * segments[None, :, :], axis=-1),
        squared_lengths[None, :],
        out=np.zeros((len(points), len(segments)), dtype=np.float64),
        where=squared_lengths[None, :] > 0,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[None, :, :] + projection[..., None] * segments[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - closest, axis=-1)
    return distances.min(axis=1)


def _boundary_alignment_error(sample: Mapping[str, Any], raw_path: Path) -> float:
    from src.datamodule.av2_data_utils import load_av2_df

    _, static_map, _ = load_av2_df(raw_path)
    origin = sample["origin"].reshape(2)
    theta = sample["theta"].reshape(())
    maximum_error = 0.0

    for lane_index, lane_id in enumerate(sample["lane_ids"].tolist()):
        if not bool(sample["lane_boundary_mask"][lane_index].all()):
            continue
        lane = static_map.vector_lane_segments[int(lane_id)]
        cached_left_world = local_to_world(
            sample["lane_left_boundaries"][lane_index], origin, theta
        ).numpy()
        cached_right_world = local_to_world(
            sample["lane_right_boundaries"][lane_index], origin, theta
        ).numpy()
        left_error = _point_to_polyline_distance(
            cached_left_world, lane.left_lane_boundary.xyz
        ).max()
        right_error = _point_to_polyline_distance(
            cached_right_world, lane.right_lane_boundary.xyz
        ).max()
        maximum_error = max(maximum_error, float(left_error), float(right_error))
    return maximum_error


def run_validate(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root) / args.split
    cache_files = _limit(sorted(cache_root.glob("*.pt")), args.limit)
    raw_files = {
        path.stem: path for path in _scenario_files(Path(args.data_root), args.split)
    }
    results = []
    total_errors = 0
    maximum_alignment_error = 0.0

    for cache_file in cache_files:
        scenario_id = cache_file.stem
        errors = []
        repeat_issues = []
        alignment_error = None
        invalid_lane_count = None
        lane_count = None
        edge_count = None

        try:
            sample = _load_cache(cache_file)
            scenario_id = sample.get("scenario_id", scenario_id)
            errors.extend(validate_cache_sample(sample))
            raw_path = raw_files.get(cache_file.stem)
            if raw_path is None:
                errors.append(f"raw scenario not found for {cache_file.stem}")
            else:
                metadata = sample["metadata"]
                extractor = Av2MapExtractor(
                    radius=metadata["radius_m"],
                    boundary_points=metadata["boundary_points"],
                    mode=args.split,
                    ignore_type=metadata["ignore_type"],
                    remove_outlier_actors=metadata["remove_outlier_actors"],
                    av2_version=metadata["av2_version"],
                )
                repeated = extractor.get_data(raw_path)
                repeat_issues = compare_samples(sample, repeated)
                if repeat_issues:
                    errors.extend(
                        [f"repeat extraction: {issue}" for issue in repeat_issues]
                    )
                alignment_error = _boundary_alignment_error(sample, raw_path)
                maximum_alignment_error = max(maximum_alignment_error, alignment_error)
                if alignment_error > args.boundary_tolerance:
                    errors.append(
                        "boundary alignment error "
                        f"{alignment_error:.8f} exceeds "
                        f"{args.boundary_tolerance:.8f}"
                    )

            invalid_lane_count = int((~sample["lane_valid_mask"]).sum().item())
            lane_count = int(sample["lane_ids"].numel())
            edge_count = int(sample["lane_edge_type"].numel())
        except Exception as error:
            errors.append(
                "failed to validate cache: " f"{type(error).__name__}: {error}"
            )

        total_errors += len(errors)
        results.append(
            {
                "scenario_id": scenario_id,
                "cache_file": str(cache_file),
                "errors": errors,
                "repeat_issue_count": len(repeat_issues),
                "boundary_alignment_error_m": alignment_error,
                "invalid_lane_count": invalid_lane_count,
                "lane_count": lane_count,
                "edge_count": edge_count,
            }
        )
        status = "PASS" if not errors else "FAIL"
        print(f"[{status}] {cache_file.name}")

    report = {
        "split": args.split,
        "cache_root": str(cache_root),
        "checked_scenarios": len(cache_files),
        "error_count": total_errors,
        "maximum_boundary_alignment_error_m": maximum_alignment_error,
        "boundary_tolerance_m": args.boundary_tolerance,
        "results": results,
    }
    if args.report:
        write_json_atomic(Path(args.report), report)
    else:
        print(json.dumps(_json_value(report), indent=2, sort_keys=True))
    return int(total_errors > 0 or len(cache_files) == 0)


def _draw_scene(sample: Mapping[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 10))
    lane_positions = sample["lane_positions"]
    valid_mask = sample["lane_valid_mask"]
    for lane_index, lane_id in enumerate(sample["lane_ids"].tolist()):
        color = "tab:blue" if bool(valid_mask[lane_index]) else "tab:red"
        center = lane_positions[lane_index]
        left = sample["lane_left_boundaries"][lane_index]
        right = sample["lane_right_boundaries"][lane_index]
        axis.plot(center[:, 0], center[:, 1], color=color, linewidth=1)
        axis.plot(left[:, 0], left[:, 1], color="tab:green", linewidth=0.7)
        axis.plot(right[:, 0], right[:, 1], color="tab:orange", linewidth=0.7)
        label_position = sample["lane_centers"][lane_index]
        axis.text(
            float(label_position[0]),
            float(label_position[1]),
            str(lane_id),
            fontsize=6,
            color=color,
        )

    for edge_index in range(sample["lane_edge_type"].numel()):
        if int(sample["lane_edge_type"][edge_index]) != EDGE_TYPE_MAP["successor"]:
            continue
        source = int(sample["lane_edge_index"][0, edge_index])
        target = int(sample["lane_edge_index"][1, edge_index])
        start = lane_positions[source, -1]
        end = lane_positions[target, 0]
        axis.annotate(
            "",
            xy=(float(end[0]), float(end[1])),
            xytext=(float(start[0]), float(start[1])),
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.5},
        )

    focal_history = sample["x_positions"][0]
    focal_valid = ~sample["x_padding_mask"][0, :50]
    focal_history = focal_history[focal_valid]
    axis.plot(
        focal_history[:, 0],
        focal_history[:, 1],
        color="black",
        linewidth=2,
        label="focal history",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{sample['scenario_id']} | lanes={len(sample['lane_ids'])} | "
        f"invalid={int((~valid_mask).sum())}"
    )
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_inspect(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root) / args.split
    output_dir = Path(args.output_dir)
    cache_files = _limit(sorted(cache_root.glob("*.pt")), args.limit)
    entries = []
    for cache_file in cache_files:
        sample = _load_cache(cache_file)
        issues = validate_cache_sample(sample)
        image_path = output_dir / f"{cache_file.stem}.png"
        _draw_scene(sample, image_path)
        entries.append(
            {
                "scenario_id": sample["scenario_id"],
                "cache_file": str(cache_file),
                "image_file": image_path.name,
                "lane_count": int(sample["lane_ids"].numel()),
                "invalid_lane_count": int((~sample["lane_valid_mask"]).sum().item()),
                "contract_issues": issues,
            }
        )
        print(f"saved {image_path}")

    write_json_atomic(
        output_dir / "index.json",
        {
            "split": args.split,
            "cache_root": str(cache_root),
            "scene_count": len(entries),
            "scenes": entries,
        },
    )
    return int(not cache_files)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate Forecast-MAE AV2 map cache v2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--data-root", required=True)
    preprocess.add_argument("--output-folder", default="forecast-mae-map-v2")
    preprocess.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    preprocess.add_argument("--boundary-points", type=int, default=20)
    preprocess.add_argument("--radius", type=float, default=150.0)
    preprocess.add_argument("--limit", type=int)
    preprocess.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="number of spawn-based worker processes",
    )
    preprocess.add_argument(
        "--force",
        action="store_true",
        help="rebuild caches even when metadata hashes match",
    )
    preprocess.add_argument(
        "--progress-every",
        type=_nonnegative_int,
        default=100,
        help="print progress after this many completed scenarios; 0 disables",
    )
    preprocess.add_argument("--fail-fast", action="store_true")
    preprocess.set_defaults(handler=run_preprocess)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--data-root", required=True)
    validate.add_argument("--cache-root", required=True)
    validate.add_argument("--split", choices=("train", "val", "test"), default="val")
    validate.add_argument("--limit", type=int, default=100)
    validate.add_argument("--boundary-tolerance", type=float, default=1e-3)
    validate.add_argument("--report")
    validate.set_defaults(handler=run_validate)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--cache-root", required=True)
    inspect.add_argument("--split", choices=("train", "val", "test"), default="val")
    inspect.add_argument("--limit", type=int, default=100)
    inspect.add_argument("--output-dir", required=True)
    inspect.set_defaults(handler=run_inspect)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
