import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .direction import encode_directionality
from .restriction import RealLinear

try:
    import torch_sparse
except ImportError:  # The differentiable full path remains available.
    torch_sparse = None


def _sparse_coo_tensor(*args, **kwargs) -> torch.Tensor:
    kwargs["check_invariants"] = False
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse invariant checks are implicitly disabled"
        )
        return torch.sparse_coo_tensor(*args, **kwargs)


def _complex_dropout(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p == 0.0:
        return inputs
    return inputs * F.dropout(torch.ones_like(inputs.real), p=p, training=True)


def _complex_relu(inputs: torch.Tensor) -> torch.Tensor:
    mask = inputs.real >= 0
    return torch.complex(inputs.real * mask, inputs.imag * mask)


def build_normalized_diffusion(
    edge_index: torch.Tensor,
    restriction: torch.Tensor,
    direction: torch.Tensor,
    q: float,
    num_nodes: int,
    num_edges: int,
    stalk_dim: int,
    light: bool,
) -> torch.Tensor:
    """Build symmetric block-normalized H B^-1 H* for diagonal stalks."""
    node_index, hyperedge_index = edge_index
    stalk = torch.arange(stalk_dim, device=edge_index.device)
    rows = (node_index[:, None] * stalk_dim + stalk).reshape(-1)
    columns = (hyperedge_index[:, None] * stalk_dim + stalk).reshape(-1)
    directed = restriction * encode_directionality(direction, q)[:, None]
    values = directed.reshape(-1)
    if light:
        values = values.detach()

    node_degree = torch.zeros(
        num_nodes * stalk_dim, device=values.device, dtype=values.real.dtype
    )
    node_degree.index_add_(0, rows, values.abs().square())
    node_scale = (node_degree + 1.0).clamp_min(1e-6).pow(-0.5)

    edge_degree = torch.zeros(
        num_edges * stalk_dim, device=values.device, dtype=values.real.dtype
    )
    edge_degree.index_add_(0, columns, torch.ones_like(values.real))
    edge_scale = edge_degree.clamp_min(1.0).pow(-0.5)
    normalized = values * node_scale[rows] * edge_scale[columns]

    indices = torch.stack((rows, columns))
    transpose_indices = torch.stack((columns, rows))
    shape = (num_nodes * stalk_dim, num_edges * stalk_dim)
    if light and torch_sparse is not None:
        q_indices, q_values = torch_sparse.spspmm(
            indices,
            normalized,
            transpose_indices,
            normalized.conj(),
            shape[0],
            shape[1],
            shape[0],
        )
        return _sparse_coo_tensor(
            q_indices,
            q_values,
            (shape[0], shape[0]),
        ).coalesce()

    incidence = _sparse_coo_tensor(indices, normalized, shape).coalesce()
    adjoint = _sparse_coo_tensor(
        transpose_indices,
        normalized.conj(),
        (shape[1], shape[0]),
    ).coalesce()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        return torch.sparse.mm(incidence, adjoint).coalesce()


class DiagonalSheafDiffusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_projection = RealLinear(channels, channels, bias=False)

    def forward(
        self, features: torch.Tensor, diffusion: torch.Tensor
    ) -> torch.Tensor:
        projected = self.channel_projection(features)
        return features + torch.sparse.mm(diffusion, projected)


__all__ = [
    "DiagonalSheafDiffusion",
    "_complex_dropout",
    "_complex_relu",
    "build_normalized_diffusion",
]
