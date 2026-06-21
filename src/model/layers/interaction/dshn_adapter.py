import torch
import torch.nn as nn

from .common import FocalInteractionBase
from .dshn import DynamicDiagonalDSHN


class DSHNInteraction(FocalInteractionBase):
    """Dynamic diagonal directional-sheaf adapter for batched AV2 scenes."""

    def __init__(
        self,
        embed_dim,
        dropout=0.1,
        stalk_dim=2,
        q=0.15,
        light=False,
        num_layers=2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = DynamicDiagonalDSHN(
            embed_dim=embed_dim,
            stalk_dim=stalk_dim,
            num_layers=num_layers,
            dropout=dropout,
            q=q,
            light=light,
        )
        self.output_projection = nn.Linear(
            embed_dim * stalk_dim * 2, embed_dim
        )
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, actor_feat, centers, angles, padding_mask, actor_attr, velocity):
        graph = self._build(centers, angles, padding_mask, actor_attr, velocity)
        incidence = self.graph_builder.build_incidence(graph)
        has_neighbors = graph.mask.any(dim=-1)
        if incidence.num_hyperedges == 0:
            return actor_feat

        edge_index = torch.stack(
            (incidence.node_index, incidence.hyperedge_index)
        )
        flat_features = actor_feat.flatten(0, 1)
        device_type = actor_feat.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            encoded = self.encoder(
                flat_features.float(),
                edge_index,
                incidence.direction,
                num_edges=incidence.num_hyperedges,
            )
            focal_encoded = encoded[incidence.focal_node_index]
            update = self.output_projection(focal_encoded)
        return self._replace_focal(
            actor_feat, update.to(actor_feat.dtype), has_neighbors
        )
