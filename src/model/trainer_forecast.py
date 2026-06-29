import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2

from .layers.mrgd import ModeDeltaLinear
from .model_forecast import ModelForecast


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        use_mrgd: bool = False,
        mrgd_route_grad: bool = True,
        mrgd_warmup_epochs: int = 5,
        mrgd_residual_scale: float = 1.0,
        mrgd_consensus: str = "mean",
        mrgd_log_interval: int = 50,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_mrgd = use_mrgd
        self.mrgd_route_grad = mrgd_route_grad
        self.mrgd_warmup_epochs = mrgd_warmup_epochs
        self.mrgd_residual_scale = mrgd_residual_scale
        self.mrgd_consensus = mrgd_consensus
        self.mrgd_log_interval = mrgd_log_interval
        self.manual_gradient_clip_val = gradient_clip_val
        self.manual_gradient_clip_algorithm = gradient_clip_algorithm
        if self.use_mrgd:
            self.automatic_optimization = False
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.net = ModelForecast(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
            use_mrgd=use_mrgd,
            mrgd_delta_scale=1.0,
        )

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob

    def cal_loss(self, out, data):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0], device=y_hat.device), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, 50:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        loss = agent_reg_loss + agent_cls_loss + others_reg_loss

        return {
            "loss": loss,
            "agent_reg_loss": agent_reg_loss,
            "agent_cls_loss": agent_cls_loss,
            "others_reg_loss": others_reg_loss,
            "best_mode": best_mode.detach(),
            "l2_norm": l2_norm.detach(),
        }

    def compute_mode_losses(self, out, data, best_mode):
        y_hat = out["y_hat"]
        y = data["y"][:, 0]
        _, num_modes = y_hat.shape[:2]

        mode_losses = []
        mode_counts = []
        for mode_idx in range(num_modes):
            mask = best_mode == mode_idx
            count = int(mask.sum().item())
            mode_counts.append(count)
            if count == 0:
                mode_losses.append(None)
            else:
                mode_losses.append(
                    F.smooth_l1_loss(y_hat[mask, mode_idx, :, :2], y[mask])
                )
        return mode_losses, mode_counts

    def compute_mrgd_grads(self, mode_losses, mode_counts, pairs):
        eps = 1e-12
        valid_modes = [
            mode_idx for mode_idx, loss in enumerate(mode_losses) if loss is not None
        ]
        num_modes = len(mode_losses)

        empty_stats = {
            "valid_modes": 0.0,
            "mean_pair_cos": 0.0,
            "conflict_rate": 0.0,
            "num_pairs": 0.0,
            "share_norm": 0.0,
            "residual_norm": 0.0,
            "residual_share_ratio": 0.0,
        }
        if len(valid_modes) == 0 or len(pairs) == 0:
            return None, None, empty_stats

        shared_params = [pair["shared"] for pair in pairs]
        grads_by_mode = []
        flat_grads_for_cos = []

        for mode_idx in range(num_modes):
            if mode_losses[mode_idx] is None:
                grads_k = [torch.zeros_like(param) for param in shared_params]
            else:
                raw_grads = torch.autograd.grad(
                    mode_losses[mode_idx],
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                grads_k = [
                    grad.detach() if grad is not None else torch.zeros_like(param)
                    for grad, param in zip(raw_grads, shared_params)
                ]

            grads_by_mode.append(grads_k)
            if mode_idx in valid_modes:
                flat_grads_for_cos.append(
                    torch.cat([grad.reshape(-1) for grad in grads_k])
                )

        total_count = max(sum(mode_counts[mode_idx] for mode_idx in valid_modes), 1)
        weights = {
            mode_idx: float(mode_counts[mode_idx]) / float(total_count)
            for mode_idx in valid_modes
        }

        shared_grads = []
        for pair_idx, _ in enumerate(shared_params):
            if self.mrgd_consensus == "normalized_mean":
                normed = []
                norms = []
                for mode_idx in valid_modes:
                    grad = grads_by_mode[mode_idx][pair_idx]
                    norm = torch.norm(grad)
                    norms.append(norm)
                    normed.append(grad / (norm + eps))
                direction = sum(
                    weights[mode_idx] * normed[idx]
                    for idx, mode_idx in enumerate(valid_modes)
                )
                mean_norm = sum(
                    weights[mode_idx] * norms[idx]
                    for idx, mode_idx in enumerate(valid_modes)
                )
                shared_grad = direction * mean_norm
            else:
                shared_grad = sum(
                    weights[mode_idx] * grads_by_mode[mode_idx][pair_idx]
                    for mode_idx in valid_modes
                )
            shared_grads.append(shared_grad.detach())

        private_grads = []
        for pair_idx, pair in enumerate(pairs):
            private_grad = torch.zeros_like(pair["private"])
            for mode_idx in valid_modes:
                residual = grads_by_mode[mode_idx][pair_idx] - shared_grads[pair_idx]
                private_grad[mode_idx] = self.mrgd_residual_scale * residual
            private_grads.append(private_grad.detach())

        mean_pair_cos = 0.0
        conflict_rate = 0.0
        num_pairs = 0
        if len(flat_grads_for_cos) >= 2:
            cos_values = []
            conflicts = []
            for i in range(len(flat_grads_for_cos)):
                for j in range(i + 1, len(flat_grads_for_cos)):
                    gi = flat_grads_for_cos[i]
                    gj = flat_grads_for_cos[j]
                    cos = torch.dot(gi, gj) / (torch.norm(gi) * torch.norm(gj) + eps)
                    cos_values.append(cos)
                    conflicts.append((cos < 0).float())
            if cos_values:
                mean_pair_cos = torch.stack(cos_values).mean().item()
                conflict_rate = torch.stack(conflicts).mean().item()
                num_pairs = len(cos_values)

        share_norm = torch.sqrt(
            sum((grad**2).sum() for grad in shared_grads) + eps
        ).item()
        residual_norm = torch.sqrt(
            sum((grad**2).sum() for grad in private_grads) + eps
        ).item()

        return shared_grads, private_grads, {
            "valid_modes": float(len(valid_modes)),
            "mean_pair_cos": float(mean_pair_cos),
            "conflict_rate": float(conflict_rate),
            "num_pairs": float(num_pairs),
            "share_norm": float(share_norm),
            "residual_norm": float(residual_norm),
            "residual_share_ratio": float(residual_norm / (share_norm + eps)),
        }

    def apply_mrgd_grads(self, pairs, shared_grads, private_grads):
        for pair, shared_grad, private_grad in zip(pairs, shared_grads, private_grads):
            pair["shared"].grad = shared_grad.clone()
            pair["private"].grad = private_grad.clone()

    def compute_diversity_metrics(self, out):
        y_hat = out["y_hat"]
        pi = out["pi"]
        batch_size, num_modes = y_hat.shape[:2]
        if num_modes < 2:
            zero = y_hat.new_tensor(0.0)
            return {
                "pairwise_fde_div": zero,
                "pairwise_ade_div": zero,
                "pi_entropy": zero,
                "effective_modes": zero,
            }

        final_pos = y_hat[:, :, -1, :]
        final_dist = torch.norm(
            final_pos[:, :, None, :] - final_pos[:, None, :, :], dim=-1
        )
        mask = ~torch.eye(num_modes, device=y_hat.device, dtype=torch.bool).unsqueeze(0)
        pairwise_fde_div = final_dist[mask.expand(batch_size, -1, -1)].view(
            batch_size, num_modes * (num_modes - 1)
        ).mean()

        traj_dist = torch.norm(
            y_hat[:, :, None, :, :] - y_hat[:, None, :, :, :], dim=-1
        ).mean(dim=-1)
        pairwise_ade_div = traj_dist[mask.expand(batch_size, -1, -1)].view(
            batch_size, num_modes * (num_modes - 1)
        ).mean()

        prob = torch.softmax(pi, dim=-1)
        entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=-1).mean()
        return {
            "pairwise_fde_div": pairwise_fde_div,
            "pairwise_ade_div": pairwise_ade_div,
            "pi_entropy": entropy,
            "effective_modes": torch.exp(entropy),
        }

    def log_mrgd_delta_norms(self):
        if not hasattr(self.net.decoder, "get_mrgd_layers"):
            return

        for layer_idx, layer in enumerate(self.net.decoder.get_mrgd_layers()):
            per_mode_norm = torch.norm(layer.delta_weight.detach().flatten(1), dim=1)
            for mode_idx, norm in enumerate(per_mode_norm):
                self.log(
                    f"mrgd/delta_weight_norm_l{layer_idx}_m{mode_idx}",
                    norm,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def training_step(self, data, batch_idx):
        if self.use_mrgd:
            return self._training_step_mrgd(data)

        out = self(data)
        losses = self.cal_loss(out, data)

        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, sync_dist=True)
        self.log(
            "train/reg_loss",
            losses["agent_reg_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/cls_loss",
            losses["agent_cls_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/others_reg_loss",
            losses["others_reg_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return losses["loss"]

    def _training_step_mrgd(self, data):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        out = self(data)
        losses = self.cal_loss(out, data)
        total_loss = losses["loss"]

        use_routing_now = (
            self.mrgd_route_grad and self.current_epoch >= self.mrgd_warmup_epochs
        )
        if use_routing_now:
            pairs = self.net.decoder.get_mrgd_param_pairs()
            mode_losses, mode_counts = self.compute_mode_losses(
                out, data, losses["best_mode"]
            )
            shared_grads, private_grads, mrgd_stats = self.compute_mrgd_grads(
                mode_losses, mode_counts, pairs
            )
        else:
            pairs = []
            shared_grads, private_grads = None, None
            mrgd_stats = {
                "valid_modes": 0.0,
                "mean_pair_cos": 0.0,
                "conflict_rate": 0.0,
                "share_norm": 0.0,
                "residual_norm": 0.0,
                "residual_share_ratio": 0.0,
            }

        self.manual_backward(total_loss)
        if use_routing_now and shared_grads is not None:
            self.apply_mrgd_grads(pairs, shared_grads, private_grads)

        if (
            self.manual_gradient_clip_val is not None
            and self.manual_gradient_clip_val > 0
        ):
            self.clip_gradients(
                opt,
                gradient_clip_val=self.manual_gradient_clip_val,
                gradient_clip_algorithm=self.manual_gradient_clip_algorithm,
            )

        opt.step()

        self.log("train/loss", total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log(
            "train/reg_loss",
            losses["agent_reg_loss"].detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/cls_loss",
            losses["agent_cls_loss"].detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/others_reg_loss",
            losses["others_reg_loss"].detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        for name, value in mrgd_stats.items():
            self.log(f"mrgd/{name}", value, on_step=True, on_epoch=True, sync_dist=True)

        best_mode = losses["best_mode"]
        for mode_idx in range(out["y_hat"].shape[1]):
            self.log(
                f"mrgd/best_mode_freq_{mode_idx}",
                (best_mode == mode_idx).float().mean(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        self.log_mrgd_delta_norms()

        return total_loss.detach()

    def on_train_epoch_end(self):
        if not self.use_mrgd:
            return

        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        for scheduler in schedulers:
            if scheduler is None:
                continue
            if isinstance(scheduler, dict):
                scheduler = scheduler.get("scheduler")
            if scheduler is not None:
                scheduler.step()

    def validation_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data["y"][:, 0])
        div_metrics = self.compute_diversity_metrics(out)

        self.log(
            "val/reg_loss",
            losses["agent_reg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        for k, v in div_metrics.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir, filename=f"forecast_mae_{timestamp}"
        )

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            ModeDeltaLinear,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
