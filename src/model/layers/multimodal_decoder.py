import torch.nn as nn

from .mrgd import ModeDeltaLinear


class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(
        self,
        embed_dim,
        future_steps,
        num_modes=6,
        use_mrgd=False,
        mrgd_delta_scale=1.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.use_mrgd = use_mrgd

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        if use_mrgd:
            last_loc = ModeDeltaLinear(
                embed_dim,
                future_steps * 2,
                num_modes=num_modes,
                delta_scale=mrgd_delta_scale,
                zero_init_delta=True,
            )
        else:
            last_loc = nn.Linear(embed_dim, future_steps * 2)

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            last_loc,
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        loc = self.loc(x).view(-1, self.num_modes, self.future_steps, 2)
        pi = self.pi(x).squeeze(-1)

        return loc, pi

    def get_mrgd_layers(self):
        return [module for module in self.modules() if isinstance(module, ModeDeltaLinear)]

    def get_mrgd_param_pairs(self):
        pairs = []
        for layer_idx, layer in enumerate(self.get_mrgd_layers()):
            for name, shared_param, private_param in layer.mrgd_param_pairs():
                pairs.append(
                    {
                        "name": f"mrgd_layer_{layer_idx}.{name}",
                        "shared": shared_param,
                        "private": private_param,
                    }
                )
        return pairs
