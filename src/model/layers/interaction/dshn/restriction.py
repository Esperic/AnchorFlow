import torch
import torch.nn as nn


class RealLinear(nn.Module):
    """Apply one real-valued linear map to both complex components."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            self.linear(inputs.real), self.linear(inputs.imag)
        )


class DiagonalRestrictionPredictor(nn.Module):
    """Official MLP_var3 diagonal restriction-map predictor."""

    def __init__(self, channels: int, stalk_dim: int) -> None:
        super().__init__()
        self.node_to_hyperedge = RealLinear(channels, channels, bias=False)
        self.restriction = nn.Linear(channels * 4, stalk_dim, bias=False)

    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        node_index, hyperedge_index = edge_index
        messages = self.node_to_hyperedge(node_features)[node_index]
        pooled = torch.zeros_like(hyperedge_features)
        pooled.index_add_(0, hyperedge_index, messages)
        node_incidence = node_features[node_index]
        hyperedge_incidence = pooled[hyperedge_index]
        inputs = torch.cat(
            (
                node_incidence.real,
                node_incidence.imag,
                hyperedge_incidence.real,
                hyperedge_incidence.imag,
            ),
            dim=-1,
        )
        return torch.sigmoid(self.restriction(inputs))
