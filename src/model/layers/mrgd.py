import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeDeltaLinear(nn.Module):
    """Shared linear layer plus mode-private additive delta."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modes: int = 6,
        bias: bool = True,
        delta_scale: float = 1.0,
        zero_init_delta: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modes = num_modes
        self.delta_scale = delta_scale

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.delta_weight = nn.Parameter(
            torch.zeros(num_modes, out_features, in_features)
        )
        self.delta_bias = (
            nn.Parameter(torch.zeros(num_modes, out_features)) if bias else None
        )

        self.reset_parameters(zero_init_delta=zero_init_delta)

    def reset_parameters(self, zero_init_delta: bool = True) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        if zero_init_delta:
            nn.init.zeros_(self.delta_weight)
            if self.delta_bias is not None:
                nn.init.zeros_(self.delta_bias)
        else:
            nn.init.normal_(self.delta_weight, std=1e-4)
            if self.delta_bias is not None:
                nn.init.zeros_(self.delta_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = torch.einsum("bki,koi->bko", x, self.delta_weight)
        if self.delta_bias is not None:
            delta = delta + self.delta_bias.unsqueeze(0)
        return base + self.delta_scale * delta

    def mrgd_param_pairs(self):
        pairs = [("weight", self.weight, self.delta_weight)]
        if self.bias is not None and self.delta_bias is not None:
            pairs.append(("bias", self.bias, self.delta_bias))
        return pairs
