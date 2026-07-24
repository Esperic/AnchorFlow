import sys
import types
import unittest

import torch

try:
    from src.metrics import minFDE
except ModuleNotFoundError:
    torchmetrics = types.ModuleType("torchmetrics")

    class Metric:
        def __init__(self, **kwargs):
            pass

        def add_state(self, name, default, dist_reduce_fx):
            setattr(self, name, default)

    torchmetrics.Metric = Metric
    sys.modules["torchmetrics"] = torchmetrics
    from src.metrics import minFDE


class MinFDETest(unittest.TestCase):
    def test_brier_min_fde6_uses_probability_of_best_forecast(self):
        y_hat = torch.zeros(1, 6, 1, 2)
        y_hat[0, :, 0, 0] = torch.arange(6)
        metric = minFDE(k=6, brier=True)

        metric.update(
            {"y_hat": y_hat, "pi": torch.zeros(1, 6)}, torch.zeros(1, 1, 2)
        )

        self.assertAlmostEqual(metric.compute().item(), 25 / 36, places=6)


if __name__ == "__main__":
    unittest.main()
