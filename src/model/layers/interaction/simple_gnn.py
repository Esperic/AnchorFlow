import torch
import torch.nn as nn

from .common import FocalInteractionBase, masked_softmax


class SimpleGNNInteraction(FocalInteractionBase):
    def __init__(self, embed_dim, dropout=0.1, num_layers=2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            message = nn.Sequential(
                nn.LayerNorm(embed_dim * 2 + 8),
                nn.Linear(embed_dim * 2 + 8, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
            score = nn.Linear(embed_dim, 1)
            output = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            self.layers.append(
                nn.ModuleDict({"message": message, "score": score, "output": output})
            )

    def reset_output_projection(self) -> None:
        for layer in self.layers:
            nn.init.zeros_(layer["output"].weight)
            nn.init.zeros_(layer["output"].bias)

    def forward(self, actor_feat, centers, angles, padding_mask, actor_attr, velocity):
        graph = self._build(centers, angles, padding_mask, actor_attr, velocity)
        neighbors = self._neighbors(actor_feat, graph)
        has_neighbors = graph.mask.any(dim=-1)
        output = actor_feat
        for layer in self.layers:
            focal = output[:, :1].expand_as(neighbors)
            messages = layer["message"](
                torch.cat((neighbors, focal, graph.edge_attr), dim=-1)
            )
            weights = masked_softmax(layer["score"](messages).squeeze(-1), graph.mask)
            update = layer["output"]((messages * weights.unsqueeze(-1)).sum(dim=1))
            output = self._replace_focal(output, update, has_neighbors)
        return output
