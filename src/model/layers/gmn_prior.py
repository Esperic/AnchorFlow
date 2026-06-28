from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class GMNPrior(nn.Module):
    def __init__(
        self,
        gmn_path: str,
        num_modes: int,
        sampling: str = "query_aligned",
        std_scale: float = 1.0,
        min_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.sampling = str(sampling).lower()
        self.std_scale = float(std_scale)
        self.min_std = float(min_std)

        data = np.load(Path(gmn_path).expanduser(), allow_pickle=False)
        center_points = self._load_required(data, "center_points", gmn_path)
        center_std = self._load_required(data, "center_std", gmn_path)
        mixture_weights = self._load_mixture_weights(data, center_points)

        if center_points.ndim != 2 or center_points.shape[1] != 2:
            raise ValueError(f"center_points must have shape [K, 2], got {center_points.shape}")
        if center_std.shape != center_points.shape:
            raise ValueError(f"center_std must match center_points, got {center_std.shape}")
        if mixture_weights.ndim != 1 or mixture_weights.shape[0] != center_points.shape[0]:
            raise ValueError(f"mixture_weights must have shape [K], got {mixture_weights.shape}")

        mixture_sum = float(mixture_weights.sum())
        if mixture_sum <= 0.0:
            raise ValueError("mixture_weights sum must be positive")
        mixture_weights = mixture_weights / mixture_sum

        self.register_buffer("center_points", torch.from_numpy(center_points), persistent=True)
        self.register_buffer("center_std", torch.from_numpy(center_std), persistent=True)
        self.register_buffer("mixture_weights", torch.from_numpy(mixture_weights), persistent=True)

    @staticmethod
    def _load_required(data, key, gmn_path):
        if key not in data:
            raise KeyError(f"{key} not found in GMN file: {gmn_path}")
        return data[key].astype(np.float32)

    @staticmethod
    def _load_mixture_weights(data, center_points):
        if "mixture_weights" in data:
            return data["mixture_weights"].astype(np.float32)
        if "proportions" in data:
            return data["proportions"].astype(np.float32)
        if "counts" in data:
            counts = data["counts"].astype(np.float32)
            return counts / max(float(counts.sum()), 1.0)
        weights = np.ones((center_points.shape[0],), dtype=np.float32)
        return weights / max(float(weights.sum()), 1.0)

    def _sample_component_indices(self, batch_size, device):
        num_components = self.center_points.shape[0]
        if self.sampling == "query_aligned":
            query_ids = torch.arange(self.num_modes, device=device, dtype=torch.long)
            query_ids = query_ids % num_components
            return query_ids.unsqueeze(0).expand(batch_size, -1)
        if self.sampling == "random":
            probs = self.mixture_weights.to(device=device)
            idx = torch.multinomial(probs, batch_size * self.num_modes, replacement=True)
            return idx.view(batch_size, self.num_modes)
        raise ValueError(f"Unsupported gmn_sampling: {self.sampling}")

    def sample(self, batch_size, device, dtype):
        comp_idx = self._sample_component_indices(batch_size=batch_size, device=device)
        center_points = self.center_points.to(device=device, dtype=dtype)
        center_std = self.center_std.to(device=device, dtype=dtype)
        means = center_points[comp_idx]
        stds = (center_std[comp_idx] * self.std_scale).clamp_min(self.min_std)
        eps = torch.randn(batch_size, self.num_modes, 2, device=device, dtype=dtype)
        return means + eps * stds
