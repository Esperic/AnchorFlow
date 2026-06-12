import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


EDGE_TYPE_MAP = {
    "successor": 0,
    "predecessor": 1,
    "left": 2,
    "right": 3,
}

INVALID_REASON = {
    "valid": 0,
    "degenerate_left": 1,
    "degenerate_right": 2,
    "non_finite": 3,
    "implausible_width": 4,
    "self_intersection": 5,
    "outside_crop": 6,
}

GEOMETRY_EPSILON_M = 1e-6
MINIMUM_LANE_WIDTH_M = 0.1
CACHE_SCHEMA_NAME = "forecast-mae-av2-map-cache"
CACHE_SCHEMA_VERSION = 2
CENTERLINE_POINTS = 20


def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    theta = torch.as_tensor(theta)
    theta = theta.squeeze()
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta, cos_theta]),
        ]
    )


def world_to_local(
    points: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    points = torch.as_tensor(points)
    origin = torch.as_tensor(origin, dtype=points.dtype, device=points.device)
    rotate_mat = rotation_matrix(
        torch.as_tensor(theta, dtype=points.dtype, device=points.device)
    )
    return torch.matmul(points - origin, rotate_mat)


def local_to_world(
    points: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    points = torch.as_tensor(points)
    origin = torch.as_tensor(origin, dtype=points.dtype, device=points.device)
    rotate_mat = rotation_matrix(
        torch.as_tensor(theta, dtype=points.dtype, device=points.device)
    )
    return torch.matmul(points, rotate_mat.transpose(-1, -2)) + origin


def _deduplicate_consecutive(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > epsilon
    return points[keep]


def resample_polyline(
    points: Any,
    num_points: int,
    epsilon: float = GEOMETRY_EPSILON_M,
) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] < 2:
        raise ValueError("polyline must have shape [N, 2+] ")

    points_xy = points_array[:, :2]
    if not np.isfinite(points_xy).all():
        raise ValueError("polyline contains non-finite coordinates")

    points_xy = _deduplicate_consecutive(points_xy, epsilon)
    if len(points_xy) < 2:
        raise ValueError("polyline must contain at least two distinct points")

    segment_lengths = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if not math.isfinite(total_length) or total_length <= epsilon:
        raise ValueError("polyline length is degenerate")

    targets = np.linspace(0.0, total_length, num_points)
    result = np.column_stack(
        [
            np.interp(targets, cumulative, points_xy[:, dimension])
            for dimension in range(2)
        ]
    )
    return result.astype(np.float32)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _on_segment(
    a: np.ndarray, b: np.ndarray, point: np.ndarray, epsilon: float
) -> bool:
    return (
        min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
        and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
    )


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    epsilon: float,
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and (
        (o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon)
    ):
        return True

    collinear = (
        abs(o1) <= epsilon
        and _on_segment(a, b, c, epsilon)
        or abs(o2) <= epsilon
        and _on_segment(a, b, d, epsilon)
        or abs(o3) <= epsilon
        and _on_segment(c, d, a, epsilon)
        or abs(o4) <= epsilon
        and _on_segment(c, d, b, epsilon)
    )
    return bool(collinear)


def polygon_self_intersects(
    polygon: Any,
    epsilon: float = GEOMETRY_EPSILON_M,
) -> bool:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 4:
        return False
    points = points[:, :2]
    if np.linalg.norm(points[0] - points[-1]) > epsilon:
        points = np.concatenate([points, points[:1]], axis=0)

    num_segments = len(points) - 1
    for first in range(num_segments):
        for second in range(first + 1, num_segments):
            if second == first + 1:
                continue
            if first == 0 and second == num_segments - 1:
                continue
            if _segments_intersect(
                points[first],
                points[first + 1],
                points[second],
                points[second + 1],
                epsilon,
            ):
                return True
    return False


def lane_polygon(
    left_boundary: Any,
    right_boundary: Any,
) -> np.ndarray:
    left = np.asarray(left_boundary, dtype=np.float64)[:, :2]
    right = np.asarray(right_boundary, dtype=np.float64)[:, :2]
    polygon = np.concatenate([left, right[::-1], left[:1]], axis=0)
    return polygon


def build_lane_graph(
    lane_segments: Iterable[Any],
) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    lanes_by_id = {int(lane.id): lane for lane in lane_segments}
    lane_ids = sorted(lanes_by_id)
    local_index = {lane_id: index for index, lane_id in enumerate(lane_ids)}
    edges = set()
    external_ref_count = [0] * len(EDGE_TYPE_MAP)

    relation_fields = (
        ("successor", "successors"),
        ("predecessor", "predecessors"),
        ("left", "left_neighbor_id"),
        ("right", "right_neighbor_id"),
    )
    for source_id in lane_ids:
        lane = lanes_by_id[source_id]
        source_index = local_index[source_id]
        for relation_name, field_name in relation_fields:
            references = getattr(lane, field_name, None)
            if references is None:
                continue
            if relation_name in ("left", "right"):
                references = [references]
            for target_id in references:
                target_id = int(target_id)
                relation_type = EDGE_TYPE_MAP[relation_name]
                if target_id in local_index:
                    edges.add((source_index, relation_type, local_index[target_id]))
                else:
                    external_ref_count[relation_type] += 1

    sorted_edges = sorted(edges)
    if sorted_edges:
        edge_index = torch.tensor(
            [
                [source for source, _, _ in sorted_edges],
                [target for _, _, target in sorted_edges],
            ],
            dtype=torch.long,
        )
        edge_types = torch.tensor(
            [relation for _, relation, _ in sorted_edges], dtype=torch.long
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_types = torch.empty((0,), dtype=torch.long)

    return (
        lane_ids,
        edge_index,
        edge_types,
        torch.tensor(external_ref_count, dtype=torch.long),
    )


def validate_graph_reciprocity(
    edge_index: torch.Tensor, edge_type: torch.Tensor
) -> List[str]:
    if edge_index.shape[0] != 2:
        return ["lane_edge_index must have shape [2, E]"]
    edge_set = {
        (int(source), int(relation), int(target))
        for source, relation, target in zip(
            edge_index[0].tolist(),
            edge_type.tolist(),
            edge_index[1].tolist(),
        )
    }
    issues = []
    for source, relation, target in sorted(edge_set):
        if relation == EDGE_TYPE_MAP["successor"]:
            reverse = (target, EDGE_TYPE_MAP["predecessor"], source)
            if reverse not in edge_set:
                issues.append(
                    f"successor edge {source}->{target} is missing "
                    f"predecessor edge {target}->{source}"
                )
        elif relation == EDGE_TYPE_MAP["predecessor"]:
            reverse = (target, EDGE_TYPE_MAP["successor"], source)
            if reverse not in edge_set:
                issues.append(
                    f"predecessor edge {source}->{target} is missing "
                    f"successor edge {target}->{source}"
                )
    return issues


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    return value


def build_cache_metadata(
    *,
    av2_version: str,
    boundary_points: int,
    radius_m: float,
    ignore_type: Sequence[int],
    remove_outlier_actors: bool,
) -> Dict[str, Any]:
    return {
        "schema_name": CACHE_SCHEMA_NAME,
        "schema_version": CACHE_SCHEMA_VERSION,
        "av2_version": str(av2_version),
        "coordinate_system": "focal_actor_t49_row_vector",
        "centerline_points": CENTERLINE_POINTS,
        "boundary_points": int(boundary_points),
        "radius_m": float(radius_m),
        "ignore_type": [int(value) for value in ignore_type],
        "remove_outlier_actors": bool(remove_outlier_actors),
        "edge_type_map": dict(EDGE_TYPE_MAP),
        "geometry_epsilon_m": GEOMETRY_EPSILON_M,
        "minimum_lane_width_m": MINIMUM_LANE_WIDTH_M,
    }


def metadata_hash(metadata: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        _json_value(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
