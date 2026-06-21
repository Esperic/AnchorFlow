import torch
import torch.nn as nn

from .diffusion import (
    DiagonalSheafDiffusion,
    _complex_dropout,
    _complex_relu,
    build_normalized_diffusion,
)
from .restriction import DiagonalRestrictionPredictor, RealLinear


class DynamicDiagonalDSHN(nn.Module):
    """Dynamic diagonal DSHN encoder adapted from the reference implementation."""

    def __init__(
        self,
        embed_dim: int,
        stalk_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        q: float = 0.15,
        light: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.stalk_dim = stalk_dim
        self.dropout = dropout
        self.q = q
        self.light = light
        self.input_projection = RealLinear(
            embed_dim, embed_dim * stalk_dim, bias=False
        )
        self.restriction = DiagonalRestrictionPredictor(embed_dim, stalk_dim)
        self.layers = nn.ModuleList(
            DiagonalSheafDiffusion(embed_dim) for _ in range(num_layers)
        )

    def _hyperedge_features(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        node_index, hyperedge_index = edge_index
        pooled = torch.zeros(
            num_edges,
            inputs.shape[-1],
            device=inputs.device,
            dtype=inputs.dtype,
        )
        pooled.index_add_(0, hyperedge_index, inputs[node_index])
        count = torch.zeros(
            num_edges, device=inputs.device, dtype=inputs.real.dtype
        )
        count.index_add_(
            0,
            hyperedge_index,
            torch.ones_like(hyperedge_index, dtype=count.dtype),
        )
        return pooled / count.clamp_min(1.0).unsqueeze(-1)

    def forward(
        self,
        actor_features: torch.Tensor,
        edge_index: torch.Tensor,
        direction: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        if num_edges == 0:
            return torch.cat(
                (
                    actor_features.repeat(1, self.stalk_dim),
                    actor_features.repeat(1, self.stalk_dim),
                ),
                dim=-1,
            )
        complex_input = torch.complex(actor_features, actor_features.detach())
        hyperedge_input = self._hyperedge_features(
            complex_input, edge_index, num_edges
        )
        projected = self.input_projection(complex_input)
        features = projected.view(-1, self.stalk_dim, self.embed_dim).reshape(
            -1, self.embed_dim
        )
        projected_hyperedges = self.input_projection(hyperedge_input)
        hyperedge_features = projected_hyperedges.view(
            num_edges, self.stalk_dim, self.embed_dim
        ).mean(dim=1)
        node_features = features.view(
            -1, self.stalk_dim, self.embed_dim
        ).mean(dim=1)
        restriction = self.restriction(
            node_features, hyperedge_features, edge_index
        )
        diffusion = build_normalized_diffusion(
            edge_index=edge_index,
            restriction=restriction,
            direction=direction,
            q=self.q,
            num_nodes=actor_features.shape[0],
            num_edges=num_edges,
            stalk_dim=self.stalk_dim,
            light=self.light,
        )
        for index, layer in enumerate(self.layers):
            features = layer(features, diffusion)
            if index < len(self.layers) - 1:
                features = _complex_relu(features)
                features = _complex_dropout(
                    features, self.dropout, self.training
                )
        features = features.view(
            actor_features.shape[0], self.stalk_dim * self.embed_dim
        )
        return torch.cat((features.real, features.imag), dim=-1)
