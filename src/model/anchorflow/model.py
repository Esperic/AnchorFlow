from typing import Dict, Optional

import torch
import torch.nn as nn

from src.model.model_forecast import ModelForecast

from .anchors import load_anchor_artifact
from .flow import (
    AnchorResidualVelocityField,
    euler_integrate,
    stable_source_noise,
)
from .matching import gather_modes


class StaticAnchorFlowModel(nn.Module):
    def __init__(
        self,
        anchor_path: str,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: float = 0.2,
        future_steps: int = 60,
        num_modes: int = 6,
        flow_num_heads: int = 8,
        flow_mlp_ratio: float = 2.0,
        integration_steps: int = 10,
        eval_noise_seed: int = 2333,
        velocity_output_zero_init: bool = True,
    ) -> None:
        super().__init__()
        artifact = load_anchor_artifact(
            anchor_path,
            expected_num_modes=num_modes,
            expected_future_steps=future_steps,
        )
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.integration_steps = integration_steps
        self.eval_noise_seed = eval_noise_seed
        self.anchor_metadata = artifact["metadata"]
        self.anchor_content_hash = artifact["content_hash"]
        self.register_buffer(
            "anchor_prototypes",
            artifact["anchors"].clone(),
            persistent=True,
        )
        self.register_buffer(
            "residual_scale",
            artifact["residual_scale"].clone(),
            persistent=True,
        )

        self.scene_encoder = ModelForecast(
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
        )
        self.scene_encoder.decoder.requires_grad_(False)
        self.prototype_encoder = nn.Sequential(
            nn.Linear(future_steps * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.velocity_field = AnchorResidualVelocityField(
            future_steps=future_steps,
            embed_dim=embed_dim,
            num_heads=flow_num_heads,
            mlp_ratio=flow_mlp_ratio,
            zero_init_output=velocity_output_zero_init,
        )

    def load_scene_encoder_checkpoint(self, checkpoint_path: str):
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        network_state = {
            key[len("net.") :]: value
            for key, value in state_dict.items()
            if key.startswith("net.")
        }
        if not network_state:
            network_state = state_dict
        return self.scene_encoder.load_state_dict(network_state, strict=True)

    def encode_context(self, data: Dict[str, torch.Tensor]):
        scene = self.scene_encoder.encode_scene(data)
        other_tokens = scene["actor_tokens"][:, 1:]
        batch_size = other_tokens.shape[0]
        y_hat_others = self.scene_encoder.dense_predictor(other_tokens).view(
            batch_size,
            -1,
            self.future_steps,
            2,
        )
        return scene, y_hat_others

    def score_modes(self, target_token: torch.Tensor) -> torch.Tensor:
        prototype_features = self.prototype_encoder(
            self.anchor_prototypes.flatten(start_dim=1)
        )
        batch_size = target_token.shape[0]
        prototype_features = prototype_features.unsqueeze(0).expand(
            batch_size,
            self.num_modes,
            -1,
        )
        target_features = target_token[:, None].expand(
            batch_size,
            self.num_modes,
            -1,
        )
        return self.score_head(
            torch.cat([target_features, prototype_features], dim=-1)
        ).squeeze(-1)

    def training_outputs(
        self,
        data: Dict[str, torch.Tensor],
        residual_state: torch.Tensor,
        time: torch.Tensor,
        matched_mode: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scene, y_hat_others = self.encode_context(data)
        batch_size = residual_state.shape[0]
        references = self.anchor_prototypes.unsqueeze(0).expand(
            batch_size,
            self.num_modes,
            self.future_steps,
            2,
        )
        matched_reference = gather_modes(references, matched_mode)
        predicted_velocity = self.velocity_field(
            residual_state=residual_state[:, None],
            reference=matched_reference[:, None],
            prototypes=matched_reference[:, None],
            time=time,
            scene_tokens=scene["scene_tokens"],
            scene_padding_mask=scene["scene_padding_mask"],
        )[:, 0]
        return {
            "predicted_velocity": predicted_velocity,
            "pi": self.score_modes(scene["target_token"]),
            "y_hat_others": y_hat_others,
        }

    def integrate_residuals(
        self,
        source: torch.Tensor,
        scene: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
        return_states: bool = False,
    ):
        batch_size = source.shape[0]
        references = self.anchor_prototypes.unsqueeze(0).expand(
            batch_size,
            self.num_modes,
            self.future_steps,
            2,
        )

        def field(state, time):
            return self.velocity_field(
                residual_state=state,
                reference=references,
                prototypes=references,
                time=time,
                scene_tokens=scene["scene_tokens"],
                scene_padding_mask=scene["scene_padding_mask"],
            )

        return euler_integrate(
            field,
            source,
            steps=self.integration_steps if steps is None else steps,
            return_states=return_states,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        source_noise: Optional[torch.Tensor] = None,
        integration_steps: Optional[int] = None,
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        scene, y_hat_others = self.encode_context(data)
        batch_size = scene["scene_tokens"].shape[0]
        if source_noise is None:
            if not self.training and "scenario_id" in data:
                source_noise = stable_source_noise(
                    data["scenario_id"],
                    num_modes=self.num_modes,
                    future_steps=self.future_steps,
                    global_seed=self.eval_noise_seed,
                    device=scene["scene_tokens"].device,
                    dtype=scene["scene_tokens"].dtype,
                )
            else:
                source_noise = torch.randn(
                    batch_size,
                    self.num_modes,
                    self.future_steps,
                    2,
                    device=scene["scene_tokens"].device,
                    dtype=scene["scene_tokens"].dtype,
                )
        integrated = self.integrate_residuals(
            source_noise,
            scene,
            steps=integration_steps,
            return_states=return_states,
        )
        if return_states:
            residual, states = integrated
        else:
            residual = integrated
            states = None
        y_hat = (
            self.anchor_prototypes.unsqueeze(0)
            + self.residual_scale.view(1, 1, 1, 2) * residual
        )
        output = {
            "y_hat": y_hat,
            "pi": self.score_modes(scene["target_token"]),
            "y_hat_others": y_hat_others,
            "anchor_prototypes": self.anchor_prototypes,
        }
        if states is not None:
            output["residual_integration_states"] = states
        return output
