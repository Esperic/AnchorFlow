import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LinearFlowSample:
    source: torch.Tensor
    target: torch.Tensor
    state: torch.Tensor
    velocity: torch.Tensor
    time: torch.Tensor
    valid_mask: torch.Tensor


def build_linear_flow_sample(
    source: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    valid_mask: torch.Tensor,
) -> LinearFlowSample:
    if source.shape != target.shape:
        raise ValueError("source and target must have identical shape")
    if source.ndim != 3 or source.shape[-1] != 2:
        raise ValueError("source and target must have shape [B, T, 2]")
    if time.shape != (source.shape[0],):
        raise ValueError("time must have shape [B]")
    if valid_mask.shape != source.shape[:2]:
        raise ValueError("valid_mask must have shape [B, T]")

    source = source.to(torch.float32)
    target = target.to(torch.float32)
    time = time.to(device=source.device, dtype=torch.float32)
    coordinate_mask = valid_mask.unsqueeze(-1)
    source = source.masked_fill(~coordinate_mask, 0.0)
    target = target.masked_fill(~coordinate_mask, 0.0)
    time_view = time[:, None, None]
    state = (1.0 - time_view) * source + time_view * target
    velocity = target - source
    return LinearFlowSample(
        source=source,
        target=target,
        state=state,
        velocity=velocity,
        time=time,
        valid_mask=valid_mask,
    )


def masked_velocity_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shape")
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must have shape [B, T, 2]")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError("valid_mask must have shape [B, T]")

    valid_rows = valid_mask.any(dim=-1)
    if not valid_rows.any():
        raise ValueError("batch contains no valid focal future")
    coordinate_mask = valid_mask.unsqueeze(-1).expand_as(prediction)
    squared_error = (prediction - target).pow(2)
    numerator = (squared_error * coordinate_mask).sum(dim=(1, 2))
    denominator = coordinate_mask.sum(dim=(1, 2)).clamp_min(1)
    per_sample = numerator / denominator
    return per_sample[valid_rows].mean(), valid_rows


def _stable_seed(global_seed: int, scenario_id: str, mode_id: int) -> int:
    payload = f"{global_seed}\0{scenario_id}\0{mode_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def stable_source_noise(
    scenario_ids: Sequence[str],
    num_modes: int,
    future_steps: int,
    global_seed: int,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    if num_modes <= 0 or future_steps <= 0:
        raise ValueError("num_modes and future_steps must be positive")
    samples = []
    for scenario_id in scenario_ids:
        modes = []
        for mode_id in range(num_modes):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_seed(global_seed, str(scenario_id), mode_id)
            )
            modes.append(
                torch.randn(
                    future_steps,
                    2,
                    generator=generator,
                    dtype=torch.float32,
                )
            )
        samples.append(torch.stack(modes))
    if not samples:
        return torch.empty(
            0,
            num_modes,
            future_steps,
            2,
            device=device,
            dtype=dtype,
        )
    return torch.stack(samples).to(device=device, dtype=dtype)


def euler_integrate(
    velocity_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    steps: int,
    return_states: bool = False,
):
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    state = source
    states = [state] if return_states else None
    step_size = 1.0 / steps
    for step in range(steps):
        time = state.new_full((state.shape[0],), step / steps)
        velocity = velocity_field(state, time)
        if velocity.shape != state.shape:
            raise ValueError("velocity field output must match state shape")
        state = state + step_size * velocity
        if return_states:
            states.append(state)
    if return_states:
        return state, torch.stack(states, dim=1)
    return state


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim < 8 or embed_dim % 2 != 0:
            raise ValueError("embed_dim must be an even integer of at least 8")
        self.embed_dim = embed_dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.to(torch.float32)
        remaining = (1.0 - time).clamp_min(1e-6)
        frequency_count = (self.embed_dim - 4) // 2
        frequencies = torch.exp(
            torch.linspace(
                0.0,
                math.log(64.0),
                frequency_count,
                device=time.device,
                dtype=torch.float32,
            )
        )
        angles = (
            2.0
            * math.pi
            * time.unsqueeze(-1)
            * frequencies
        )
        endpoint_features = torch.stack(
            [
                time,
                remaining,
                time * remaining,
                -remaining.log(),
            ],
            dim=-1,
        )
        return torch.cat(
            [endpoint_features, angles.sin(), angles.cos()],
            dim=-1,
        )


class TimeConditionedResidualSkip(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        zero_init_output: bool,
    ) -> None:
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.matrix_projection = nn.Linear(embed_dim, 4)
        if zero_init_output:
            nn.init.zeros_(self.matrix_projection.weight)
            nn.init.zeros_(self.matrix_projection.bias)

    def forward(
        self,
        residual_state: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if residual_state.ndim != 3 or residual_state.shape[-1] != 2:
            raise ValueError("residual_state must have shape [N, T, 2]")
        if time_embedding.shape != (
            residual_state.shape[0],
            self.matrix_projection.in_features,
        ):
            raise ValueError("time_embedding must have shape [N, D]")
        matrix = self.matrix_projection(
            self.conditioner(time_embedding)
        ).view(-1, 2, 2)
        return torch.einsum("nij,ntj->nti", matrix, residual_state)


class AnchorResidualVelocityField(nn.Module):
    def __init__(
        self,
        future_steps: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.embed_dim = embed_dim
        self.residual_encoder = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.residual_film = nn.Linear(embed_dim, embed_dim * 2)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.temporal_position = nn.Embedding(future_steps, embed_dim)
        nn.init.normal_(self.temporal_position.weight, std=0.02)
        hidden_dim = int(embed_dim * mlp_ratio)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=2,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.output_block = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.velocity_projection = nn.Linear(embed_dim, 2)
        self.residual_skip = TimeConditionedResidualSkip(
            embed_dim,
            zero_init_output=zero_init_output,
        )
        if zero_init_output:
            nn.init.zeros_(self.velocity_projection.weight)
            nn.init.zeros_(self.velocity_projection.bias)

    def _expand_time(
        self,
        time: torch.Tensor,
        batch_size: int,
        num_modes: int,
    ) -> torch.Tensor:
        if time.shape == (batch_size,):
            return time[:, None].expand(batch_size, num_modes)
        if time.shape == (batch_size, num_modes):
            return time
        raise ValueError(
            "time must have shape [B] or [B, K], "
            f"got {tuple(time.shape)}"
        )

    def forward(
        self,
        residual_state: torch.Tensor,
        reference: torch.Tensor,
        time: torch.Tensor,
        target_token: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = residual_state.shape
        if residual_state.ndim != 4 or residual_state.shape[-2:] != (
            self.future_steps,
            2,
        ):
            raise ValueError(
                "residual_state must have shape [B, K, T, 2]"
            )
        if reference.shape != expected_shape:
            raise ValueError("reference must match residual_state shape")
        batch_size, num_modes = residual_state.shape[:2]
        if target_token.shape != (batch_size, self.embed_dim):
            raise ValueError(
                "target_token must have shape [B, D], "
                f"got {tuple(target_token.shape)}"
            )
        if scene_tokens.shape[0] != batch_size:
            raise ValueError("scene_tokens batch dimension must match residual_state")
        if scene_tokens.shape[-1] != self.embed_dim:
            raise ValueError("scene_tokens embedding dimension does not match field")
        if scene_padding_mask.shape != scene_tokens.shape[:2]:
            raise ValueError("scene_padding_mask must have shape [B, S]")

        flat_residual = residual_state.reshape(
            batch_size * num_modes,
            self.future_steps,
            2,
        )
        flat_reference = reference.reshape(
            batch_size * num_modes,
            self.future_steps,
            2,
        )
        flat_time = self._expand_time(
            time, batch_size, num_modes
        ).reshape(-1)
        time_embedding = self.time_encoder(flat_time)[:, None].expand(
            -1,
            self.future_steps,
            -1,
        )
        residual_features = self.residual_encoder(flat_residual)
        residual_scale, residual_bias = self.residual_film(
            time_embedding[:, 0]
        ).chunk(2, dim=-1)
        residual_features = residual_features * (
            1.0 + residual_scale[:, None]
        ) + residual_bias[:, None]
        target_embedding = self.target_encoder(target_token)
        target_embedding = target_embedding[:, None].expand(
            batch_size,
            num_modes,
            self.embed_dim,
        ).reshape(batch_size * num_modes, self.embed_dim)
        target_embedding = target_embedding[:, None].expand(
            -1,
            self.future_steps,
            -1,
        )
        query = self.fusion_projection(
            torch.cat(
                [
                    residual_features,
                    self.anchor_encoder(flat_reference),
                    time_embedding,
                    target_embedding,
                ],
                dim=-1,
            )
        )
        query = self.temporal_encoder(
            query + self.temporal_position.weight.unsqueeze(0)
        )

        memory = scene_tokens[:, None].expand(
            batch_size,
            num_modes,
            scene_tokens.shape[1],
            self.embed_dim,
        ).reshape(batch_size * num_modes, scene_tokens.shape[1], self.embed_dim)
        memory_mask = scene_padding_mask[:, None].expand(
            batch_size,
            num_modes,
            scene_tokens.shape[1],
        ).reshape(batch_size * num_modes, scene_tokens.shape[1])
        attended, _ = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=False,
        )
        hidden = query + attended
        hidden = hidden + self.output_block(hidden)
        velocity = self.velocity_projection(self.output_norm(hidden))
        velocity = velocity + self.residual_skip(
            flat_residual,
            time_embedding[:, 0],
        )
        return velocity.view(
            batch_size,
            num_modes,
            self.future_steps,
            2,
        )
