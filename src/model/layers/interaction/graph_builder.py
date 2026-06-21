from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FocalInteractionGraph:
    indices: torch.Tensor
    mask: torch.Tensor
    edge_attr: torch.Tensor
    num_actors: int


@dataclass
class FocalDirectedIncidence:
    node_index: torch.Tensor
    hyperedge_index: torch.Tensor
    direction: torch.Tensor
    scene_index: torch.Tensor
    focal_node_index: torch.Tensor
    num_hyperedges: int


class FocalInteractionBuilder(nn.Module):
    """Build a fixed-size, focal-centered neighborhood without CPU transfers."""

    def __init__(
        self,
        radius: float = 50.0,
        max_neighbors: int = 16,
        lateral_threshold: float = 6.0,
        ttc_threshold: float = 5.0,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.lateral_threshold = lateral_threshold
        self.ttc_threshold = ttc_threshold

    def forward(
        self,
        centers: torch.Tensor,
        angles: torch.Tensor,
        padding_mask: torch.Tensor,
        actor_attr: torch.Tensor,
        velocity: torch.Tensor,
    ) -> FocalInteractionGraph:
        batch_size, num_actors, _ = centers.shape
        rel_pos = centers - centers[:, :1]
        focal_angle = angles[:, :1]
        cos, sin = torch.cos(focal_angle), torch.sin(focal_angle)
        rel_x = rel_pos[..., 0] * cos + rel_pos[..., 1] * sin
        rel_y = -rel_pos[..., 0] * sin + rel_pos[..., 1] * cos
        rel_pos = torch.stack((rel_x, rel_y), dim=-1)

        speed = velocity
        world_velocity = torch.stack(
            (speed * torch.cos(angles), speed * torch.sin(angles)), dim=-1
        )
        rel_velocity = world_velocity - world_velocity[:, :1]
        rel_vx = rel_velocity[..., 0] * cos + rel_velocity[..., 1] * sin
        rel_vy = -rel_velocity[..., 0] * sin + rel_velocity[..., 1] * cos
        rel_velocity = torch.stack((rel_vx, rel_vy), dim=-1)

        distance = torch.linalg.vector_norm(rel_pos, dim=-1)
        actor_type = actor_attr[..., 2]
        valid_type = (actor_type == 0) | (actor_type == 2)
        valid = (~padding_mask) & valid_type & (distance < self.radius)
        valid[:, 0] = False

        rank_distance = distance.masked_fill(~valid, torch.inf)
        selected_count = min(self.max_neighbors, num_actors)
        _, indices = torch.topk(
            rank_distance, k=selected_count, dim=1, largest=False, sorted=True
        )
        if selected_count < self.max_neighbors:
            indices = torch.cat(
                [
                    indices,
                    torch.zeros(
                        batch_size,
                        self.max_neighbors - selected_count,
                        dtype=torch.long,
                        device=centers.device,
                    ),
                ],
                dim=1,
            )

        selected_distance = self._gather(distance, indices)
        mask = self._gather(valid, indices) & torch.isfinite(selected_distance)
        selected_pos = self._gather(rel_pos, indices)
        selected_velocity = self._gather(rel_velocity, indices)
        heading_delta = self._gather(angles - focal_angle, indices)
        closing_speed = -(selected_pos * selected_velocity).sum(
            dim=-1
        ) / selected_distance.clamp_min(1e-3)
        ttc = selected_distance / closing_speed.clamp_min(1e-3)
        ttc_risk = torch.where(
            closing_speed > 0,
            (1.0 - ttc / self.ttc_threshold).clamp(min=0.0, max=1.0),
            torch.zeros_like(ttc),
        )
        edge_attr = torch.cat(
            [
                selected_pos / self.radius,
                selected_velocity / 30.0,
                (selected_distance / self.radius).unsqueeze(-1),
                torch.sin(heading_delta).unsqueeze(-1),
                torch.cos(heading_delta).unsqueeze(-1),
                ttc_risk.unsqueeze(-1),
            ],
            dim=-1,
        )
        edge_attr = edge_attr.masked_fill(~mask.unsqueeze(-1), 0.0)
        return FocalInteractionGraph(
            indices=indices,
            mask=mask,
            edge_attr=edge_attr,
            num_actors=num_actors,
        )

    def hyperedge_mask(self, graph: FocalInteractionGraph) -> torch.Tensor:
        x = graph.edge_attr[..., 0] * self.radius
        y = graph.edge_attr[..., 1] * self.radius
        corridor = y.abs() < self.lateral_threshold
        masks = (
            graph.mask,
            graph.mask & (x > 0) & corridor,
            graph.mask & (x < 0) & corridor,
            graph.mask & (~corridor),
            graph.mask & (graph.edge_attr[..., -1] > 0),
        )
        return torch.stack(masks, dim=1)

    def build_incidence(
        self, graph: FocalInteractionGraph
    ) -> FocalDirectedIncidence:
        masks = self.hyperedge_mask(graph)
        valid_hyperedges = masks.any(dim=-1)
        edge_ids = (
            valid_hyperedges.flatten().cumsum(dim=0).view_as(valid_hyperedges)
            - 1
        )

        tail_scene, tail_type, tail_slot = masks.nonzero(as_tuple=True)
        tail_nodes = (
            tail_scene * graph.num_actors
            + graph.indices[tail_scene, tail_slot]
        )
        tail_edges = edge_ids[tail_scene, tail_type]

        head_scene, head_type = valid_hyperedges.nonzero(as_tuple=True)
        head_nodes = head_scene * graph.num_actors
        head_edges = edge_ids[head_scene, head_type]

        direction = torch.cat(
            (
                torch.ones_like(tail_nodes, dtype=torch.float32),
                -torch.ones_like(head_nodes, dtype=torch.float32),
            )
        )
        return FocalDirectedIncidence(
            node_index=torch.cat((tail_nodes, head_nodes)),
            hyperedge_index=torch.cat((tail_edges, head_edges)),
            direction=direction,
            scene_index=head_scene,
            focal_node_index=(
                torch.arange(
                    graph.indices.shape[0], device=graph.indices.device
                )
                * graph.num_actors
            ),
            num_hyperedges=int(valid_hyperedges.sum().item()),
        )

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return values.gather(1, indices)
        return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
