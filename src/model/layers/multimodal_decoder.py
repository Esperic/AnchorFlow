import torch.nn as nn

from .gmn_prior import GMNPrior


class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(
        self,
        embed_dim,
        future_steps,
        num_modes=6,
        gmn_path=None,
        gmn_sampling="query_aligned",
        gmn_std_scale=1.0,
        gmn_min_std=1e-3,
        gmn_latent_dim=16,
        gmn_temperature=1.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.gmn_temperature = float(gmn_temperature)

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.gmn_prior = None
        if gmn_path:
            self.gmn_prior = GMNPrior(
                gmn_path=gmn_path,
                num_modes=num_modes,
                sampling=gmn_sampling,
                std_scale=gmn_std_scale,
                min_std=gmn_min_std,
            )
            self.gmn_xy_to_latent = nn.Linear(2, gmn_latent_dim)
            self.gmn_mode_proj = nn.Linear(gmn_latent_dim, embed_dim)
            self.gmn_context_norm = nn.LayerNorm(embed_dim)
            self.gmn_context_film = nn.Linear(gmn_latent_dim, embed_dim * 2)

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        mode_feat = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        if self.gmn_prior is not None:
            gmn_xy = self.gmn_prior.sample(
                batch_size=x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )
            z = self.gmn_xy_to_latent(gmn_xy) * self.gmn_temperature
            global_z = z.mean(dim=1)
            gamma, beta = self.gmn_context_film(global_z).chunk(2, dim=-1)
            conditioned = self.gmn_context_norm(x)
            conditioned = conditioned * (1.0 + gamma) + beta
            mode_feat = mode_feat + conditioned.unsqueeze(1) + self.gmn_mode_proj(z)

        loc = self.loc(mode_feat).view(-1, self.num_modes, self.future_steps, 2)
        pi = self.pi(mode_feat).squeeze(-1)

        return loc, pi
