import datetime
from pathlib import Path
from typing import Dict, Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2

from .losses import build_matched_flow_batch, compute_stage3_losses
from .model import StaticAnchorFlowModel


class StaticAnchorFlowTrainer(pl.LightningModule):
    def __init__(
        self,
        anchor_paths: Mapping[str, str],
        dim: int = 128,
        historical_steps: int = 50,
        future_steps: int = 60,
        num_modes: int = 6,
        encoder_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: float = 0.2,
        flow_num_heads: int = 8,
        flow_mlp_ratio: float = 2.0,
        flow_path: str = "linear",
        flow_prediction: str = "velocity",
        source_distribution: str = "standard_normal",
        time_distribution: str = "uniform",
        flow_time_max: float = 0.95,
        integration_method: str = "euler",
        integration_steps: int = 5,
        eval_noise_seed: int = 2333,
        residual_scale_min: float = 0.5,
        velocity_output_zero_init: bool = True,
        pretrained_weights: str = None,
        flow_weight: float = 1.0,
        score_weight: float = 1.0,
        other_weight: float = 1.0,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self._validate_algorithm_config(
            flow_path=flow_path,
            flow_prediction=flow_prediction,
            source_distribution=source_distribution,
            time_distribution=time_distribution,
            flow_time_max=flow_time_max,
            integration_method=integration_method,
            residual_scale_min=residual_scale_min,
        )
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.flow_weight = flow_weight
        self.score_weight = score_weight
        self.other_weight = other_weight

        self.net = StaticAnchorFlowModel(
            anchor_paths=anchor_paths,
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
            num_modes=num_modes,
            flow_num_heads=flow_num_heads,
            flow_mlp_ratio=flow_mlp_ratio,
            integration_steps=integration_steps,
            eval_noise_seed=eval_noise_seed,
            velocity_output_zero_init=velocity_output_zero_init,
        )
        if torch.any(
            self.net.residual_scales_by_family < residual_scale_min
        ):
            raise ValueError(
                "anchor artifact residual scale violates residual_scale_min"
            )
        if pretrained_weights is not None:
            self.net.load_scene_encoder_checkpoint(pretrained_weights)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=num_modes),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=num_modes),
                "MR": MR(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.submission_handler = SubmissionAv2()

    @staticmethod
    def _validate_algorithm_config(**config) -> None:
        expected = {
            "flow_path": "linear",
            "flow_prediction": "velocity",
            "source_distribution": "standard_normal",
            "time_distribution": "uniform",
            "integration_method": "euler",
        }
        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(
                    f"Stage 3 requires {key}={value!r}, got {config[key]!r}"
                )
        if config["residual_scale_min"] < 0.5:
            raise ValueError("residual_scale_min must be at least 0.5")
        if not 0.0 < float(config["flow_time_max"]) <= 1.0:
            raise ValueError("flow_time_max must be within (0, 1]")

    def forward(self, data):
        return self.net(data)

    def _shared_loss(
        self,
        data,
        flow_time_max: float,
    ) -> Dict[str, torch.Tensor]:
        targets = data["y"][:, 0].to(torch.float32)
        focal_valid_mask = ~data["x_padding_mask"][:, 0, 50:]
        anchor_selection = self.net.select_anchor_bank(data)
        flow_batch = build_matched_flow_batch(
            anchors=anchor_selection.anchors,
            residual_scale=anchor_selection.residual_scales,
            targets=targets,
            valid_mask=focal_valid_mask,
            flow_time_max=flow_time_max,
        )
        output = self.net.training_outputs(
            data,
            residual_state=flow_batch.flow.state,
            time=flow_batch.flow.time,
            matched_mode=flow_batch.matched_mode,
            anchor_prototypes=anchor_selection.anchors,
        )
        losses = compute_stage3_losses(
            predicted_velocity=output["predicted_velocity"],
            target_velocity=flow_batch.flow.velocity,
            mode_logits=output["pi"],
            matched_mode=flow_batch.matched_mode,
            focal_valid_mask=focal_valid_mask,
            y_hat_others=output["y_hat_others"],
            y_others=data["y"][:, 1:],
            other_valid_mask=~data["x_padding_mask"][:, 1:, 50:],
            flow_weight=self.flow_weight,
            score_weight=self.score_weight,
            other_weight=self.other_weight,
        )
        valid_coordinates = focal_valid_mask.unsqueeze(-1).expand_as(
            flow_batch.flow.velocity
        )
        target_velocity = flow_batch.flow.velocity[valid_coordinates]
        predicted_velocity = output["predicted_velocity"][valid_coordinates]
        losses.update(
            {
                "target_velocity_mean": target_velocity.mean(),
                "target_velocity_std": target_velocity.std(unbiased=False),
                "target_velocity_abs_max": target_velocity.abs().max(),
                "predicted_velocity_mean": predicted_velocity.mean(),
                "predicted_velocity_std": predicted_velocity.std(
                    unbiased=False
                ),
                "predicted_velocity_abs_max": predicted_velocity.abs().max(),
            }
        )
        return losses

    def training_step(self, data, batch_idx):
        del batch_idx
        losses = self._shared_loss(
            data,
            flow_time_max=float(self.hparams.flow_time_max),
        )
        for name, value in losses.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return losses["loss"]

    def validation_step(self, data, batch_idx):
        del batch_idx
        losses = self._shared_loss(data, flow_time_max=1.0)
        self.log(
            "val_loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/flow_loss",
            losses["flow_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        output = self.net(data)
        complete_rows = (~data["x_padding_mask"][:, 0, 50:]).all(dim=-1)
        if complete_rows.any():
            metric_output = {
                "y_hat": output["y_hat"][complete_rows],
                "pi": output["pi"][complete_rows],
            }
            metrics = self.val_metrics(
                metric_output,
                data["y"][complete_rows, 0],
            )
            self.log_dict(
                metrics,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=int(complete_rows.sum()),
                sync_dist=True,
            )

    def predict(self, data):
        with torch.no_grad():
            output = self.net(data)
        return self.submission_handler.format_data(
            data,
            output["y_hat"],
            output["pi"],
            inference=True,
        )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = save_dir / f"anchorflow_static_av2_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(save_dir=run_dir)

    def test_step(self, data, batch_idx) -> None:
        del batch_idx
        output = self.net(data)
        self.submission_handler.format_data(
            data,
            output["y_hat"],
            output["pi"],
        )

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim < 2 or name.endswith("bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
