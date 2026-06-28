import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gmn_prior import GMNPrior


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")


class DriftTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DriftTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class DriftGmnDecoder(nn.Module):
    """
    DriftTraj-style GMN query decoder adapted to Forecast-MAE memory tokens.

    The decoder samples one GMN latent per mode, uses it to build query content
    and query position, FiLM-conditions scene memory, then cross-attends memory
    with a lightweight Transformer decoder.
    """

    def __init__(
        self,
        embed_dim,
        future_steps,
        num_modes=6,
        num_heads=8,
        decoder_depth=2,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        gmn_path=None,
        gmn_sampling="query_aligned",
        gmn_std_scale=1.0,
        gmn_min_std=1e-3,
        gmn_latent_dim=16,
        gmn_temperature=1.0,
    ) -> None:
        super().__init__()
        if not gmn_path:
            raise ValueError("DriftGmnDecoder requires gmn_path")

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.latent_dim = gmn_latent_dim
        self.gmn_temperature = float(gmn_temperature)

        self.gmn_prior = GMNPrior(
            gmn_path=gmn_path,
            num_modes=num_modes,
            sampling=gmn_sampling,
            std_scale=gmn_std_scale,
            min_std=gmn_min_std,
        )
        self.gmn_xy_to_latent = nn.Linear(2, gmn_latent_dim)
        self.query_position_encoding = nn.Embedding(num_modes, embed_dim)
        self.latent_projection = nn.Linear(gmn_latent_dim, embed_dim)
        self.query_noise_proj = nn.Linear(gmn_latent_dim, embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.memory_film = nn.Linear(gmn_latent_dim, embed_dim * 2)

        decoder_layer = DriftTransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = DriftTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=decoder_depth,
            norm=nn.LayerNorm(embed_dim),
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def _sample_latent(self, batch_size, device, dtype):
        gmn_xy = self.gmn_prior.sample(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        z = self.gmn_xy_to_latent(gmn_xy)
        return z * self.gmn_temperature

    def forward(self, memory, memory_key_padding_mask=None):
        batch_size = memory.shape[0]
        z = self._sample_latent(
            batch_size=batch_size,
            device=memory.device,
            dtype=memory.dtype,
        )

        tgt = self.latent_projection(z)
        query_noise = self.query_noise_proj(z)
        query_pos = self.query_position_encoding.weight.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        query_pos = query_pos + query_noise

        global_z = z.mean(dim=1)
        memory_gamma, memory_beta = self.memory_film(global_z).chunk(2, dim=-1)
        conditioned_memory = self.memory_norm(memory)
        conditioned_memory = conditioned_memory * (1.0 + memory_gamma.unsqueeze(1))
        conditioned_memory = conditioned_memory + memory_beta.unsqueeze(1)

        decoder_output = self.decoder(
            tgt=tgt,
            memory=conditioned_memory,
            memory_key_padding_mask=memory_key_padding_mask,
            query_pos=query_pos,
        )
        loc = self.output_head(decoder_output).view(
            batch_size,
            self.num_modes,
            self.future_steps,
            2,
        )
        pi = self.pi(decoder_output).squeeze(-1)
        return loc, pi
