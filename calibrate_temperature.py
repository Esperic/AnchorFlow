import os
from importlib import import_module

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path


def brier_min_fde_at_temperatures(logits, fde, temperatures):
    min_fde, best_mode = fde.min(-1)
    probability = torch.softmax(
        logits.unsqueeze(0) / temperatures[:, None, None], dim=-1
    )
    best_probability = probability.gather(
        -1,
        best_mode[None, :, None].expand(temperatures.numel(), -1, 1),
    ).squeeze(-1)
    return min_fde.unsqueeze(0) + (1 - best_probability).square()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    checkpoint = to_absolute_path(conf.checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist")

    model_path = conf.model.target._target_
    module = import_module(model_path[: model_path.rfind(".")])
    model_class = getattr(module, model_path[model_path.rfind(".") + 1 :])
    model = model_class.load_from_checkpoint(
        checkpoint,
        probability_temperature=1.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    datamodule = instantiate(conf.datamodule, test=False)
    datamodule.setup("fit")

    temperatures = torch.linspace(
        float(conf.get("temperature_min", 0.5)),
        float(conf.get("temperature_max", 2.0)),
        int(conf.get("temperature_steps", 31)),
        device=device,
    )
    score_sum = torch.zeros_like(temperatures)
    count = 0

    with torch.no_grad():
        for data in datamodule.val_dataloader():
            data = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in data.items()
            }
            out = model(data)
            target = data["y"][:, 0]
            fde = torch.norm(
                out["y_hat"][..., -1, :2] - target[:, None, -1, :2], dim=-1
            )
            score_sum += brier_min_fde_at_temperatures(
                out["pi"], fde, temperatures
            ).sum(-1)
            count += target.shape[0]

    scores = score_sum / count
    best = scores.argmin()
    print(f"probability_temperature={temperatures[best].item():.6g}")
    print(f"val_b-minFDE6={scores[best].item():.6f}")


if __name__ == "__main__":
    main()
