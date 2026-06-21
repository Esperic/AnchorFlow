import torch
import torch.nn as nn

from .common import FocalInteractionBase, masked_softmax


class DirectedHypergraphInteraction(FocalInteractionBase):
    num_hyperedge_types = 5

    def __init__(self, embed_dim, dropout=0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.type_embed = nn.Parameter(torch.empty(self.num_hyperedge_types, embed_dim))
        self.tail = nn.Sequential(
            nn.LayerNorm(embed_dim + 8),
            nn.Linear(embed_dim + 8, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tail_score = nn.Linear(embed_dim, self.num_hyperedge_types)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.output = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.type_embed, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_output_projection(self) -> None:
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, actor_feat, centers, angles, padding_mask, actor_attr, velocity):
        graph = self._build(centers, angles, padding_mask, actor_attr, velocity)
        masks = self.graph_builder.hyperedge_mask(graph)
        neighbors = self._neighbors(actor_feat, graph)
        tails = self.tail(torch.cat((neighbors, graph.edge_attr), dim=-1))
        scores = self.tail_score(tails).transpose(1, 2)
        weights = masked_softmax(scores, masks)
        pooled = torch.einsum("btk,bkc->btc", weights, tails)
        focal = actor_feat[:, :1].expand(-1, self.num_hyperedge_types, -1)
        types = self.type_embed.unsqueeze(0).expand(actor_feat.shape[0], -1, -1)
        messages = self.head(torch.cat((pooled, focal, types), dim=-1))
        valid_hyperedges = masks.any(dim=-1)
        messages = messages.masked_fill(~valid_hyperedges.unsqueeze(-1), 0.0)
        count = valid_hyperedges.sum(dim=-1, keepdim=True).clamp_min(1)
        update = self.output(messages.sum(dim=1) / count)
        return self._replace_focal(actor_feat, update, valid_hyperedges.any(dim=-1))
