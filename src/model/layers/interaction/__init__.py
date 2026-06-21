import torch.nn as nn

from .directed_hypergraph import DirectedHypergraphInteraction
from .dshn_adapter import DSHNInteraction
from .simple_gnn import SimpleGNNInteraction


class IdentityInteraction(nn.Module):
    def forward(self, actor_feat, *args, **kwargs):
        return actor_feat


def build_interaction(
    interaction_type="none",
    embed_dim=128,
    radius=50.0,
    max_neighbors=16,
    dropout=0.1,
    num_layers=2,
    stalk_dim=2,
    q=0.15,
    lateral_threshold=6.0,
    ttc_threshold=5.0,
):
    interaction_type = interaction_type.lower()
    if interaction_type in ("none", "identity"):
        return IdentityInteraction()
    if interaction_type == "gnn":
        return SimpleGNNInteraction(
            embed_dim=embed_dim,
            dropout=dropout,
            num_layers=num_layers,
            radius=radius,
            max_neighbors=max_neighbors,
            lateral_threshold=lateral_threshold,
            ttc_threshold=ttc_threshold,
        )
    if interaction_type in ("directed_hypergraph", "dhg"):
        return DirectedHypergraphInteraction(
            embed_dim=embed_dim,
            dropout=dropout,
            radius=radius,
            max_neighbors=max_neighbors,
            lateral_threshold=lateral_threshold,
            ttc_threshold=ttc_threshold,
        )
    if interaction_type in ("dshn_light", "dshnlight"):
        return DSHNInteraction(
            embed_dim=embed_dim,
            dropout=dropout,
            stalk_dim=stalk_dim,
            q=q,
            light=True,
            num_layers=num_layers,
            radius=radius,
            max_neighbors=max_neighbors,
            lateral_threshold=lateral_threshold,
            ttc_threshold=ttc_threshold,
        )
    if interaction_type == "dshn":
        return DSHNInteraction(
            embed_dim=embed_dim,
            dropout=dropout,
            stalk_dim=stalk_dim,
            q=q,
            light=False,
            num_layers=num_layers,
            radius=radius,
            max_neighbors=max_neighbors,
            lateral_threshold=lateral_threshold,
            ttc_threshold=ttc_threshold,
        )
    raise ValueError(f"Unknown interaction type: {interaction_type}")


__all__ = ["build_interaction"]
