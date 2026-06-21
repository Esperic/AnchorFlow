import torch
import torch.nn.functional as F


def safe_smooth_l1_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction, target)
