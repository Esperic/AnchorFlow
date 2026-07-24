import importlib
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
    def test_sharp_style_loss_handles_empty_other_agents(self):
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
        trainer.winner_fde_weight = 1.0
        trainer.endpoint_reg_weight = 1.0
        trainer.other_loss_weight = 1.0

        y_hat = torch.full((1, 6, 2, 2), 10.0)
        y_hat[:, 0, 0] = 0
        y_hat[:, 0, 1] = torch.tensor([2.0, 0.0])
        y_hat[:, 1, :, 0] = 1
        y_hat[:, 1, :, 1] = 0
        outputs = {
            "y_hat": y_hat,
            "pi": torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]]),
            "y_hat_others": torch.empty(1, 0, 2, 2),
        }
        data = {
            "y": torch.zeros(1, 1, 2, 2),
            "x_padding_mask": torch.zeros(1, 1, 110, dtype=torch.bool),
        }

        losses = trainer.cal_loss(outputs, data)
        expected_cls_loss = torch.nn.functional.cross_entropy(
            outputs["pi"], torch.tensor([1])
        )

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertAlmostEqual(losses["trajectory_reg_loss"], 0.25)
        self.assertAlmostEqual(losses["endpoint_reg_loss"], 0.25)
        self.assertAlmostEqual(losses["reg_loss"], 0.5)
        self.assertAlmostEqual(losses["cls_loss"], expected_cls_loss.item())
        self.assertAlmostEqual(
            losses["loss"].item(), 0.5 + expected_cls_loss.item(), places=6
        )
        self.assertEqual(losses["others_reg_loss"], 0)

        trainer.net = torch.nn.Module()
        trainer.net.backbone = torch.nn.Linear(2, 2)
        trainer.net.decoder = torch.nn.Linear(2, 2)
        trainer.net.dense_predictor = torch.nn.Linear(2, 2)
        trainer.lr = 1e-4
        trainer.weight_decay = 1e-4
        trainer.warmup_epochs = 2
        trainer.epochs = 15
        optimizers, _ = trainer.configure_optimizers()

        self.assertTrue(
            all("lr_scale" not in group for group in optimizers[0].param_groups)
        )
        self.assertEqual(
            len({group["lr"] for group in optimizers[0].param_groups}), 1
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
