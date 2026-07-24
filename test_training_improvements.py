import importlib
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from average_checkpoints import average_checkpoints
from calibrate_temperature import brier_min_fde_at_temperatures


class TrainingImprovementsTest(unittest.TestCase):
    def test_metric_aligned_loss_handles_empty_other_agents(self):
        lightning = types.ModuleType("pytorch_lightning")
        lightning.LightningModule = torch.nn.Module
        torchmetrics = types.ModuleType("torchmetrics")
        torchmetrics.MetricCollection = object
        metrics = types.ModuleType("src.metrics")
        metrics.MR = metrics.minADE = metrics.minFDE = object
        submission = types.ModuleType("src.utils.submission_av2")
        submission.SubmissionAv2 = object
        model_forecast = types.ModuleType("src.model.model_forecast")
        model_forecast.ModelForecast = object

        stubs = {
            "pytorch_lightning": lightning,
            "torchmetrics": torchmetrics,
            "src.metrics": metrics,
            "src.utils.submission_av2": submission,
            "src.model.model_forecast": model_forecast,
        }
        sys.modules.pop("src.model.trainer_forecast", None)
        with mock.patch.dict(sys.modules, stubs):
            trainer_module = importlib.import_module("src.model.trainer_forecast")

        trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
        torch.nn.Module.__init__(trainer)
        trainer.fde_loss_weight = 0.5
        trainer.cls_temperature = 1.0
        trainer.mr_loss_weight = 0.1
        trainer.other_loss_weight = 0.25
        trainer.miss_threshold = 2.0

        y_hat = torch.full((1, 6, 2, 2), 2.0)
        y_hat[:, 0] = 0
        outputs = {
            "y_hat": y_hat,
            "pi": torch.zeros(1, 6),
            "y_hat_others": torch.empty(1, 0, 2, 2),
        }
        data = {
            "y": torch.zeros(1, 1, 2, 2),
            "x_padding_mask": torch.zeros(1, 1, 110, dtype=torch.bool),
        }

        losses = trainer.cal_loss(outputs, data)

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertAlmostEqual(losses["loss"].item(), math.log(6), places=6)
        self.assertEqual(losses["others_reg_loss"], 0)

        trainer.net = torch.nn.Module()
        trainer.net.backbone = torch.nn.Linear(2, 2)
        trainer.net.decoder = torch.nn.Linear(2, 2)
        trainer.net.dense_predictor = torch.nn.Linear(2, 2)
        trainer.lr = 3e-4
        trainer.weight_decay = 1e-4
        trainer.head_lr_scale = 10 / 3
        trainer.warmup_epochs = 10
        trainer.epochs = 60
        optimizers, _ = trainer.configure_optimizers()

        self.assertEqual(
            {group["lr_scale"] for group in optimizers[0].param_groups},
            {1.0, 10 / 3},
        )

    def test_temperature_search_includes_uncalibrated_score(self):
        logits = torch.tensor([[2.0, 0.0]])
        fde = torch.tensor([[0.0, 1.0]])
        temperatures = torch.tensor([0.5, 1.0])

        scores = brier_min_fde_at_temperatures(logits, fde, temperatures)

        self.assertLess(scores[0, 0], scores[1, 0])

    def test_checkpoint_weights_are_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, value in enumerate((1.0, 3.0)):
                path = Path(directory) / f"{index}.ckpt"
                torch.save(
                    {
                        "state_dict": {
                            "weight": torch.tensor([value]),
                            "counter": torch.tensor(index),
                        }
                    },
                    path,
                )
                paths.append(path)

            checkpoint = average_checkpoints(paths)

        torch.testing.assert_close(
            checkpoint["state_dict"]["weight"], torch.tensor([2.0])
        )
        self.assertEqual(checkpoint["state_dict"]["counter"].item(), 1)


if __name__ == "__main__":
    unittest.main()
