import torch


def select_nearest_mode(pred, gt):
    distance = torch.sqrt(torch.pow(pred - gt[:, None, :, :], 2).sum(dim=-1) + 1e-9)
    distance = distance.sum(dim=-1)
    nearest_mode_ids = torch.argmin(distance, dim=-1)
    nearest_mode_bs_ids = torch.arange(gt.shape[0], device=pred.device).long()
    return nearest_mode_ids, nearest_mode_bs_ids


def split_winner_and_others(pred, winner_ids, batch_ids):
    winner = pred[batch_ids, winner_ids].unsqueeze(1)

    batch_size, num_modes, _, coord_dim = pred.shape
    if num_modes == 1:
        others = pred[:, :0]
        return winner, others

    selected_mask = torch.zeros(batch_size, num_modes, device=pred.device, dtype=torch.bool)
    selected_mask.scatter_(1, winner_ids.unsqueeze(1), True)
    others = pred[~selected_mask].view(batch_size, num_modes - 1, pred.size(2), coord_dim)
    return winner, others


def endpoint_diversity_loss(pred, sigma=2.0, eps=1e-6):
    endpoints = pred[:, :, -1, :]
    num_modes = endpoints.size(1)
    if num_modes <= 1:
        return endpoints.new_zeros(())

    pairwise_sq = torch.cdist(endpoints, endpoints).pow(2)
    penalties = torch.exp(-pairwise_sq / max(sigma ** 2, eps))
    eye = torch.eye(num_modes, device=pred.device, dtype=torch.bool).unsqueeze(0)
    penalties = penalties.masked_fill(eye, 0.0)
    normalizer = pred.size(0) * num_modes * (num_modes - 1)
    return penalties.sum() / max(normalizer, 1)
