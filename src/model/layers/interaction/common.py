import torch
import torch.nn as nn

from .graph_builder import FocalInteractionBuilder, FocalInteractionGraph


class FocalInteractionBase(nn.Module):
    def __init__(self, radius=50.0, max_neighbors=16, **builder_kwargs) -> None:
        super().__init__()
        self.graph_builder = FocalInteractionBuilder(
            radius=radius, max_neighbors=max_neighbors, **builder_kwargs
        )

    def _build(self, centers, angles, padding_mask, actor_attr, velocity):
        return self.graph_builder(centers, angles, padding_mask, actor_attr, velocity)

    @staticmethod
    def _neighbors(actor_feat, graph: FocalInteractionGraph):
        return actor_feat.gather(
            1,
            graph.indices.unsqueeze(-1).expand(-1, -1, actor_feat.shape[-1]),
        )

    @staticmethod
    def _replace_focal(actor_feat, update, has_neighbors):
        output = actor_feat.clone()
        output[:, 0] = torch.where(
            has_neighbors.unsqueeze(-1), actor_feat[:, 0] + update, actor_feat[:, 0]
        )
        return output


def masked_softmax(logits, mask, dim=-1):
    logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
    weights = torch.softmax(logits, dim=dim)
    return weights.masked_fill(~mask, 0.0)
