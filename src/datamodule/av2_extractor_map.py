import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .av2_map_utils import (
    CACHE_SCHEMA_NAME,
    CACHE_SCHEMA_VERSION,
    CENTERLINE_POINTS,
    INVALID_REASON,
    MINIMUM_LANE_WIDTH_M,
    build_cache_metadata,
    build_lane_graph,
    lane_polygon,
    metadata_hash,
    polygon_self_intersects,
    resample_polyline,
    validate_graph_reciprocity,
    world_to_local,
)


LANE_TYPE_MAP = {
    "VEHICLE": 0,
    "BIKE": 1,
    "BUS": 2,
}

LANE_MARK_TYPE_MAP = {
    "DASH_SOLID_YELLOW": 0,
    "DASH_SOLID_WHITE": 1,
    "DASHED_WHITE": 2,
    "DASHED_YELLOW": 3,
    "DOUBLE_SOLID_YELLOW": 4,
    "DOUBLE_SOLID_WHITE": 5,
    "DOUBLE_DASH_YELLOW": 6,
    "DOUBLE_DASH_WHITE": 7,
    "SOLID_YELLOW": 8,
    "SOLID_WHITE": 9,
    "SOLID_DASH_WHITE": 10,
    "SOLID_DASH_YELLOW": 11,
    "SOLID_BLUE": 12,
    "NONE": 13,
    "UNKNOWN": 14,
}


def _enum_value(value: Any) -> str:
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def _installed_av2_version() -> str:
    try:
        return importlib.metadata.version("av2")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


class Av2MapExtractor:
    def __init__(
        self,
        radius: float = 150,
        boundary_points: int = 20,
        save_path: Optional[Path] = None,
        mode: str = "train",
        ignore_type: Sequence[int] = (5, 6, 7, 8, 9),
        remove_outlier_actors: bool = True,
        av2_version: Optional[str] = None,
    ) -> None:
        if boundary_points < 2:
            raise ValueError("boundary_points must be at least 2")
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of train, val, test")

        self.save_path = Path(save_path) if save_path is not None else None
        self.mode = mode
        self.radius = float(radius)
        self.boundary_points = int(boundary_points)
        self.remove_outlier_actors = bool(remove_outlier_actors)
        self.ignore_type = tuple(int(value) for value in ignore_type)
        self.metadata = build_cache_metadata(
            av2_version=av2_version or _installed_av2_version(),
            boundary_points=self.boundary_points,
            radius_m=self.radius,
            ignore_type=self.ignore_type,
            remove_outlier_actors=self.remove_outlier_actors,
        )
        self.metadata_hash = metadata_hash(self.metadata)

    def save(self, file: Path) -> Path:
        if self.save_path is None:
            raise ValueError("save_path must be configured before saving")

        self.save_path.mkdir(parents=True, exist_ok=True)
        data = self.get_data(file)
        save_file = self.save_path / f"{Path(file).stem}.pt"
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.save_path,
                prefix=f".{save_file.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_file = Path(handle.name)
            torch.save(data, temp_file)
            os.replace(temp_file, save_file)
        finally:
            if temp_file is not None and temp_file.exists():
                temp_file.unlink()
        return save_file

    def get_data(self, file: Path) -> Dict[str, Any]:
        return self.process(file)

    def extract_lane_features(
        self,
        lane_segments: Sequence[Any],
        origin: torch.Tensor,
        theta: torch.Tensor,
    ) -> Dict[str, Any]:
        sorted_segments = sorted(lane_segments, key=lambda lane: int(lane.id))
        lane_ids, edge_index, edge_type, external_ref_count = build_lane_graph(
            sorted_segments
        )

        lane_positions = []
        left_boundaries = []
        right_boundaries = []
        boundary_masks = []
        boundary_types = []
        lane_centers = []
        lane_angles = []
        lane_attrs = []
        lane_padding_masks = []
        intersections = []
        valid_masks = []
        invalid_reasons = []

        origin = torch.as_tensor(origin, dtype=torch.float32).reshape(2)
        theta = torch.as_tensor(theta, dtype=torch.float32).squeeze()

        for lane in sorted_segments:
            reason = INVALID_REASON["valid"]
            left_world = np.asarray(lane.left_lane_boundary.xyz)[:, :2]
            right_world = np.asarray(lane.right_lane_boundary.xyz)[:, :2]
            left_sampled = None
            right_sampled = None

            if not np.isfinite(left_world).all() or not np.isfinite(right_world).all():
                reason = INVALID_REASON["non_finite"]
            else:
                try:
                    left_sampled = resample_polyline(left_world, self.boundary_points)
                except ValueError:
                    reason = INVALID_REASON["degenerate_left"]
                if reason == INVALID_REASON["valid"]:
                    try:
                        right_sampled = resample_polyline(
                            right_world, self.boundary_points
                        )
                    except ValueError:
                        reason = INVALID_REASON["degenerate_right"]

            has_geometry = left_sampled is not None and right_sampled is not None
            if has_geometry:
                center_boundary = (left_sampled + right_sampled) * 0.5
                try:
                    center_sampled = resample_polyline(
                        center_boundary, CENTERLINE_POINTS
                    )
                except ValueError:
                    center_sampled = np.zeros((CENTERLINE_POINTS, 2), dtype=np.float32)
                    reason = INVALID_REASON["self_intersection"]
                width = float(
                    np.linalg.norm(left_sampled - right_sampled, axis=1).mean()
                )
                if not np.isfinite(center_sampled).all() or not np.isfinite(width):
                    reason = INVALID_REASON["non_finite"]
                elif width <= MINIMUM_LANE_WIDTH_M:
                    reason = INVALID_REASON["implausible_width"]
                elif polygon_self_intersects(lane_polygon(left_sampled, right_sampled)):
                    reason = INVALID_REASON["self_intersection"]

                left_local = world_to_local(
                    torch.from_numpy(left_sampled), origin, theta
                ).float()
                right_local = world_to_local(
                    torch.from_numpy(right_sampled), origin, theta
                ).float()
                center_local = world_to_local(
                    torch.from_numpy(center_sampled), origin, theta
                ).float()
                padding_mask = (center_local[:, 0].abs() > self.radius) | (
                    center_local[:, 1].abs() > self.radius
                )
                if padding_mask.all() and reason == INVALID_REASON["valid"]:
                    reason = INVALID_REASON["outside_crop"]
                boundary_mask = torch.ones(self.boundary_points, dtype=torch.bool)
            else:
                width = 0.0
                left_local = torch.zeros(self.boundary_points, 2, dtype=torch.float32)
                right_local = torch.zeros_like(left_local)
                center_local = torch.zeros(CENTERLINE_POINTS, 2, dtype=torch.float32)
                padding_mask = torch.ones(CENTERLINE_POINTS, dtype=torch.bool)
                boundary_mask = torch.zeros(self.boundary_points, dtype=torch.bool)

            center = center_local[9:11].mean(dim=0)
            direction = center_local[10] - center_local[9]
            angle = torch.atan2(direction[1], direction[0])
            lane_type = LANE_TYPE_MAP.get(_enum_value(lane.lane_type), -1)
            left_mark_type = LANE_MARK_TYPE_MAP.get(
                _enum_value(lane.left_mark_type), LANE_MARK_TYPE_MAP["UNKNOWN"]
            )
            right_mark_type = LANE_MARK_TYPE_MAP.get(
                _enum_value(lane.right_mark_type), LANE_MARK_TYPE_MAP["UNKNOWN"]
            )
            is_intersection = bool(lane.is_intersection)

            lane_positions.append(center_local)
            left_boundaries.append(left_local)
            right_boundaries.append(right_local)
            boundary_masks.append(boundary_mask)
            boundary_types.append(
                torch.tensor([left_mark_type, right_mark_type], dtype=torch.long)
            )
            lane_centers.append(center)
            lane_angles.append(angle)
            lane_attrs.append(
                torch.tensor(
                    [lane_type, width, float(is_intersection)],
                    dtype=torch.float32,
                )
            )
            lane_padding_masks.append(padding_mask)
            intersections.append(is_intersection)
            valid_masks.append(reason == INVALID_REASON["valid"])
            invalid_reasons.append(reason)

        num_lanes = len(sorted_segments)
        if num_lanes == 0:
            return {
                "lane_positions": torch.empty(
                    0, CENTERLINE_POINTS, 2, dtype=torch.float32
                ),
                "lane_left_boundaries": torch.empty(
                    0, self.boundary_points, 2, dtype=torch.float32
                ),
                "lane_right_boundaries": torch.empty(
                    0, self.boundary_points, 2, dtype=torch.float32
                ),
                "lane_boundary_mask": torch.empty(
                    0, self.boundary_points, dtype=torch.bool
                ),
                "lane_boundary_type": torch.empty(0, 2, dtype=torch.long),
                "lane_centers": torch.empty(0, 2, dtype=torch.float32),
                "lane_angles": torch.empty(0, dtype=torch.float32),
                "lane_attr": torch.empty(0, 3, dtype=torch.float32),
                "lane_padding_mask": torch.empty(
                    0, CENTERLINE_POINTS, dtype=torch.bool
                ),
                "is_intersections": torch.empty(0, dtype=torch.float32),
                "lane_ids": torch.empty(0, dtype=torch.long),
                "lane_valid_mask": torch.empty(0, dtype=torch.bool),
                "lane_invalid_reason": torch.empty(0, dtype=torch.long),
                "lane_edge_index": edge_index,
                "lane_edge_type": edge_type,
                "lane_external_ref_count": external_ref_count,
            }

        return {
            "lane_positions": torch.stack(lane_positions),
            "lane_left_boundaries": torch.stack(left_boundaries),
            "lane_right_boundaries": torch.stack(right_boundaries),
            "lane_boundary_mask": torch.stack(boundary_masks),
            "lane_boundary_type": torch.stack(boundary_types),
            "lane_centers": torch.stack(lane_centers),
            "lane_angles": torch.stack(lane_angles),
            "lane_attr": torch.stack(lane_attrs),
            "lane_padding_mask": torch.stack(lane_padding_masks),
            "is_intersections": torch.tensor(intersections, dtype=torch.float32),
            "lane_ids": torch.tensor(lane_ids, dtype=torch.long),
            "lane_valid_mask": torch.tensor(valid_masks, dtype=torch.bool),
            "lane_invalid_reason": torch.tensor(invalid_reasons, dtype=torch.long),
            "lane_edge_index": edge_index,
            "lane_edge_type": edge_type,
            "lane_external_ref_count": external_ref_count,
        }

    def process(self, raw_path: Path, agent_id: Optional[str] = None) -> Dict[str, Any]:
        from .av2_data_utils import (
            OBJECT_TYPE_MAP,
            OBJECT_TYPE_MAP_COMBINED,
            load_av2_df,
        )

        df, static_map, scenario_id = load_av2_df(raw_path)
        city = df.city.values[0]
        agent_id = agent_id or df["focal_track_id"].values[0]

        local_df = df[df["track_id"] == agent_id].iloc
        origin = torch.tensor(
            [local_df[49]["position_x"], local_df[49]["position_y"]],
            dtype=torch.float32,
        )
        theta = torch.tensor(local_df[49]["heading"], dtype=torch.float32)

        timestamps = list(np.sort(df["timestep"].unique()))
        current_df = df[df["timestep"] == timestamps[49]]
        actor_ids = list(current_df["track_id"].unique())
        current_positions = torch.from_numpy(
            current_df[["position_x", "position_y"]].values
        ).float()
        out_of_range = (
            torch.linalg.norm(current_positions - origin, dim=1).numpy() > self.radius
        )
        actor_ids = [
            actor for index, actor in enumerate(actor_ids) if not out_of_range[index]
        ]
        actor_ids.remove(agent_id)
        actor_ids = [agent_id] + actor_ids
        num_nodes = len(actor_ids)
        df = df[df["track_id"].isin(actor_ids)]

        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float32)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.uint8)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float32)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float32)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)

        for current_actor_id, actor_df in df.groupby("track_id"):
            node_index = actor_ids.index(current_actor_id)
            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]
            object_name = actor_df["object_type"].values[0]
            object_type = OBJECT_TYPE_MAP[object_name]
            x_attr[node_index, 0] = object_type
            x_attr[node_index, 1] = int(actor_df["object_category"].values[0])
            x_attr[node_index, 2] = OBJECT_TYPE_MAP_COMBINED[object_name]

            padding_mask[node_index, node_steps] = False
            if padding_mask[node_index, 49] or object_type in self.ignore_type:
                padding_mask[node_index, 50:] = True

            positions = torch.from_numpy(
                actor_df[["position_x", "position_y"]].values
            ).float()
            headings = torch.from_numpy(actor_df["heading"].values).float()
            velocity = torch.from_numpy(
                actor_df[["velocity_x", "velocity_y"]].values
            ).float()
            x[node_index, node_steps] = world_to_local(positions, origin, theta)
            x_heading[node_index, node_steps] = (headings - theta + np.pi) % (
                2 * np.pi
            ) - np.pi
            x_velocity[node_index, node_steps] = torch.linalg.norm(velocity, dim=1)

        lane_segments = static_map.get_nearby_lane_segments(origin.numpy(), self.radius)
        map_features = self.extract_lane_features(
            lane_segments, origin=origin, theta=theta
        )

        if self.remove_outlier_actors and len(lane_segments) > 0:
            usable_lanes = map_features["lane_positions"][
                map_features["lane_valid_mask"]
            ]
            if usable_lanes.numel() > 0:
                lane_samples = usable_lanes.reshape(-1, 2)
                nearest_distance = torch.cdist(x[:, 49], lane_samples).min(dim=1).values
                valid_actor_mask = nearest_distance < 5
                valid_actor_mask[0] = True
                x = x[valid_actor_mask]
                x_heading = x_heading[valid_actor_mask]
                x_velocity = x_velocity[valid_actor_mask]
                x_attr = x_attr[valid_actor_mask]
                padding_mask = padding_mask[valid_actor_mask]
                num_nodes = x.shape[0]

        x_centers = x[:, 49].clone()
        x_positions = x[:, :50].clone()
        x_velocity_diff = x_velocity[:, :50].clone()

        x[:, 50:] = torch.where(
            (padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
            torch.zeros(num_nodes, 60, 2),
            x[:, 50:] - x[:, 49].unsqueeze(-2),
        )
        x[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
            torch.zeros(num_nodes, 49, 2),
            x[:, 1:50] - x[:, :49],
        )
        x[:, 0] = 0

        x_velocity_diff[:, 1:50] = torch.where(
            padding_mask[:, :49] | padding_mask[:, 1:50],
            torch.zeros(num_nodes, 49),
            x_velocity_diff[:, 1:50] - x_velocity_diff[:, :49],
        )
        x_velocity_diff[:, 0] = 0

        sample = {
            "x": x[:, :50],
            "y": None if self.mode == "test" else x[:, 50:],
            "x_attr": x_attr,
            "x_positions": x_positions,
            "x_centers": x_centers,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_padding_mask": padding_mask,
            "origin": origin.view(1, 2),
            "theta": theta.view(1),
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
            "metadata": self.metadata,
            "metadata_hash": self.metadata_hash,
        }
        sample.update(map_features)
        issues = validate_cache_sample(sample)
        if issues:
            raise ValueError(
                f"invalid cache sample for {scenario_id}: " + "; ".join(issues)
            )
        return sample


def validate_cache_sample(sample: Mapping[str, Any]) -> List[str]:
    required_keys = {
        "x",
        "y",
        "x_attr",
        "x_positions",
        "x_centers",
        "x_angles",
        "x_velocity",
        "x_velocity_diff",
        "x_padding_mask",
        "lane_positions",
        "lane_left_boundaries",
        "lane_right_boundaries",
        "lane_boundary_mask",
        "lane_boundary_type",
        "lane_centers",
        "lane_angles",
        "lane_attr",
        "lane_padding_mask",
        "is_intersections",
        "lane_ids",
        "lane_valid_mask",
        "lane_invalid_reason",
        "lane_edge_index",
        "lane_edge_type",
        "lane_external_ref_count",
        "metadata",
        "metadata_hash",
        "origin",
        "theta",
        "scenario_id",
        "track_id",
        "city",
    }
    missing = sorted(required_keys.difference(sample))
    if missing:
        return [f"missing required key: {key}" for key in missing]

    issues = []
    expected_hash = metadata_hash(sample["metadata"])
    if sample["metadata_hash"] != expected_hash:
        issues.append(
            "metadata_hash does not match metadata: " f"expected {expected_hash}"
        )
    if (
        sample["metadata"].get("schema_name") != CACHE_SCHEMA_NAME
        or sample["metadata"].get("schema_version") != CACHE_SCHEMA_VERSION
    ):
        issues.append(
            f"metadata schema must be {CACHE_SCHEMA_NAME} "
            f"version {CACHE_SCHEMA_VERSION}"
        )

    num_actors = int(sample["x"].shape[0])
    actor_shapes = {
        "x": (num_actors, 50, 2),
        "x_attr": (num_actors, 3),
        "x_positions": (num_actors, 50, 2),
        "x_centers": (num_actors, 2),
        "x_angles": (num_actors, 110),
        "x_velocity": (num_actors, 110),
        "x_velocity_diff": (num_actors, 50),
        "x_padding_mask": (num_actors, 110),
    }
    if sample["y"] is not None:
        actor_shapes["y"] = (num_actors, 60, 2)
    for key, expected_shape in actor_shapes.items():
        if tuple(sample[key].shape) != expected_shape:
            formatted = ", ".join(str(value) for value in expected_shape)
            issues.append(
                f"{key} has shape {tuple(sample[key].shape)}, "
                f"expected [{formatted}]"
            )

    num_lanes = int(sample["lane_positions"].shape[0])
    boundary_points = int(sample["metadata"]["boundary_points"])
    expected_shapes = {
        "lane_positions": (num_lanes, CENTERLINE_POINTS, 2),
        "lane_left_boundaries": (num_lanes, boundary_points, 2),
        "lane_right_boundaries": (num_lanes, boundary_points, 2),
        "lane_boundary_mask": (num_lanes, boundary_points),
        "lane_boundary_type": (num_lanes, 2),
        "lane_centers": (num_lanes, 2),
        "lane_angles": (num_lanes,),
        "lane_attr": (num_lanes, 3),
        "lane_padding_mask": (num_lanes, CENTERLINE_POINTS),
        "is_intersections": (num_lanes,),
        "lane_ids": (num_lanes,),
        "lane_valid_mask": (num_lanes,),
        "lane_invalid_reason": (num_lanes,),
        "lane_external_ref_count": (4,),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(sample[key].shape) != expected_shape:
            issues.append(
                f"{key} has shape {tuple(sample[key].shape)}, "
                f"expected {expected_shape}"
            )

    edge_index = sample["lane_edge_index"]
    edge_type = sample["lane_edge_type"]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        issues.append(
            f"lane_edge_index has shape {tuple(edge_index.shape)}, expected [2, E]"
        )
    elif edge_type.shape != (edge_index.shape[1],):
        issues.append(
            f"lane_edge_type has shape {tuple(edge_type.shape)}, "
            f"expected {(edge_index.shape[1],)}"
        )
    else:
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= num_lanes
        ):
            issues.append("lane_edge_index contains an out-of-range lane index")
        issues.extend(validate_graph_reciprocity(edge_index, edge_type))

    if sample["lane_ids"].dtype != torch.long:
        issues.append("lane_ids must use torch.int64")
    if sample["lane_edge_index"].dtype != torch.long:
        issues.append("lane_edge_index must use torch.int64")
    if sample["lane_edge_type"].dtype != torch.long:
        issues.append("lane_edge_type must use torch.int64")
    if sample["lane_boundary_mask"].dtype != torch.bool:
        issues.append("lane_boundary_mask must use torch.bool")
    if sample["lane_valid_mask"].dtype != torch.bool:
        issues.append("lane_valid_mask must use torch.bool")

    float_keys = [
        "x",
        "x_positions",
        "x_centers",
        "x_angles",
        "x_velocity",
        "x_velocity_diff",
        "lane_positions",
        "lane_left_boundaries",
        "lane_right_boundaries",
        "lane_centers",
        "lane_angles",
        "lane_attr",
        "is_intersections",
        "origin",
        "theta",
    ]
    if sample["y"] is not None:
        float_keys.append("y")
    for key in float_keys:
        if not torch.isfinite(sample[key]).all():
            issues.append(f"{key} contains non-finite values")

    lane_ids = sample["lane_ids"].tolist()
    if lane_ids != sorted(lane_ids):
        issues.append("lane_ids must be sorted")
    if len(lane_ids) != len(set(lane_ids)):
        issues.append("lane_ids must be unique")
    return issues
