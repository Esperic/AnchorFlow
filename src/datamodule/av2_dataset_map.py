from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .av2_extractor_map import Av2MapExtractor, validate_cache_sample
from .av2_map_utils import metadata_hash


ACTOR_DENSE_KEYS = (
    "x",
    "x_attr",
    "x_positions",
    "x_centers",
    "x_angles",
    "x_velocity",
    "x_velocity_diff",
)

LANE_DENSE_KEYS = (
    "lane_positions",
    "lane_left_boundaries",
    "lane_right_boundaries",
    "lane_centers",
    "lane_angles",
    "lane_attr",
    "is_intersections",
)

BOOL_MASK_KEYS = (
    "x_padding_mask",
    "lane_padding_mask",
    "lane_boundary_mask",
    "lane_valid_mask",
)


class Av2MapDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cached_split: Optional[str] = None,
        extractor: Optional[Av2MapExtractor] = None,
        validate_on_load: bool = False,
    ) -> None:
        super().__init__()
        self.validate_on_load = validate_on_load

        if cached_split is not None:
            self.data_folder = Path(data_root) / cached_split
            self.file_list = sorted(self.data_folder.glob("*.pt"))
            self.load = True
            self.extractor = None
        elif extractor is not None:
            self.extractor = extractor
            self.data_folder = Path(data_root)
            self.file_list = sorted(self.data_folder.rglob("*.parquet"))
            self.load = False
        else:
            raise ValueError("Either cached_split or extractor must be specified")

        print(
            f"data root: {self.data_folder}, "
            f"total number of files: {len(self.file_list)}"
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.load:
            try:
                data = torch.load(
                    self.file_list[index],
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                data = torch.load(self.file_list[index], map_location="cpu")
        else:
            data = self.extractor.get_data(self.file_list[index])

        expected_hash = metadata_hash(data["metadata"])
        if data["metadata_hash"] != expected_hash:
            raise ValueError(
                f"invalid cache file {self.file_list[index]}: "
                f"metadata hash mismatch, expected {expected_hash}"
            )
        if self.validate_on_load:
            issues = validate_cache_sample(data)
            if issues:
                raise ValueError(
                    f"invalid cache file {self.file_list[index]}: " + "; ".join(issues)
                )
        return data


def _pad(
    batch: Sequence[Dict[str, Any]],
    key: str,
    padding_value: Any = 0,
) -> torch.Tensor:
    return pad_sequence(
        [sample[key] for sample in batch],
        batch_first=True,
        padding_value=padding_value,
    )


def collate_fn_map(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("batch must contain at least one sample")

    data: Dict[str, Any] = {}
    for key in ACTOR_DENSE_KEYS + LANE_DENSE_KEYS:
        data[key] = _pad(batch, key)

    if batch[0]["y"] is not None:
        if any(sample["y"] is None for sample in batch):
            raise ValueError("a batch cannot mix samples with and without y")
        data["y"] = _pad(batch, "y")

    for key in BOOL_MASK_KEYS:
        padding_value = key in ("x_padding_mask", "lane_padding_mask")
        data[key] = _pad(batch, key, padding_value=padding_value)

    data["lane_boundary_type"] = _pad(batch, "lane_boundary_type", padding_value=-1)
    data["lane_ids"] = _pad(batch, "lane_ids", padding_value=-1)
    data["lane_invalid_reason"] = _pad(batch, "lane_invalid_reason", padding_value=-1)
    data["lane_external_ref_count"] = torch.stack(
        [sample["lane_external_ref_count"] for sample in batch]
    )

    num_lanes = torch.tensor(
        [sample["lane_positions"].shape[0] for sample in batch],
        dtype=torch.long,
    )
    lane_ptr = torch.cat([torch.zeros(1, dtype=torch.long), num_lanes.cumsum(dim=0)])
    lane_batch = torch.repeat_interleave(
        torch.arange(len(batch), dtype=torch.long), num_lanes
    )

    edge_indices = []
    edge_types = []
    edge_batches = []
    for batch_index, sample in enumerate(batch):
        edge_index = sample["lane_edge_index"]
        if edge_index.shape[1] == 0:
            continue
        edge_indices.append(edge_index + lane_ptr[batch_index])
        edge_types.append(sample["lane_edge_type"])
        edge_batches.append(
            torch.full(
                (edge_index.shape[1],),
                batch_index,
                dtype=torch.long,
            )
        )

    if edge_indices:
        data["lane_edge_index"] = torch.cat(edge_indices, dim=1)
        data["lane_edge_type"] = torch.cat(edge_types, dim=0)
        data["lane_edge_batch"] = torch.cat(edge_batches, dim=0)
    else:
        data["lane_edge_index"] = torch.empty((2, 0), dtype=torch.long)
        data["lane_edge_type"] = torch.empty((0,), dtype=torch.long)
        data["lane_edge_batch"] = torch.empty((0,), dtype=torch.long)

    data["lane_ptr"] = lane_ptr
    data["lane_batch"] = lane_batch
    data["x_key_padding_mask"] = data["x_padding_mask"].all(dim=-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(dim=-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(dim=-1)
    data["num_lanes"] = num_lanes

    data["scenario_id"] = [sample["scenario_id"] for sample in batch]
    data["track_id"] = [sample["track_id"] for sample in batch]
    data["city"] = [sample["city"] for sample in batch]
    data["metadata"] = [sample["metadata"] for sample in batch]
    data["metadata_hash"] = [sample["metadata_hash"] for sample in batch]
    data["origin"] = torch.cat([sample["origin"] for sample in batch], dim=0)
    data["theta"] = torch.cat([sample["theta"] for sample in batch], dim=0)
    return data
