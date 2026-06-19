import torch
import torch.nn.functional as F


def cdist(x, y, eps=1e-6):
    """
    Batched Euclidean distance.

    Args:
        x: [B, N, S]
        y: [B, M, S]

    Returns:
        dist: [B, N, M]
    """
    xy_dot = torch.einsum("bns,bms->bnm", x, y)
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)
    sq_dist = (x_norm + y_norm - 2.0 * xy_dot).clamp_min(0.0)
    return (sq_dist + eps).sqrt()


def flatten_trajectories(traj):
    """
    Convert trajectories into drift-space points.

    Args:
        traj:
            [B, T, 2]      -> [B, 1, 2T]
            [B, C, T, 2]   -> [B, C, 2T]
            [B, C, S]      -> unchanged

    Returns:
        points: [B, C, S]
    """
    if traj.dim() == 3 and traj.size(-1) == 2:
        return traj.flatten(start_dim=1).unsqueeze(1)
    if traj.dim() == 4 and traj.size(-1) == 2:
        return traj.flatten(start_dim=2)
    if traj.dim() == 3:
        return traj
    raise ValueError(f"Unsupported trajectory/point shape: {tuple(traj.shape)}")


def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.02, 0.1, 0.5),
    eps=1e-8,
    return_info=False,
):
    """
    PyTorch implementation of the official drifting loss core.

    This function operates on sets of points in drift space:
      gen:       [B, C_g, S]
      fixed_pos: [B, C_p, S]
      fixed_neg: [B, C_n, S] or None

    For trajectory prediction, the principled usage is to first flatten an
    entire future trajectory into one point in trajectory space, e.g.
    [B, M, T, 2] -> [B, M, 2T].

    Returns:
        loss: scalar tensor
        info (optional): diagnostics dict
    """
    if gen.dim() != 3 or fixed_pos.dim() != 3:
        raise ValueError("gen and fixed_pos must have shape [B, C, S]")

    if fixed_neg is None:
        fixed_neg = gen[:, :0, :]
    elif fixed_neg.dim() != 3:
        raise ValueError("fixed_neg must have shape [B, C_n, S]")

    batch_size, num_gen, feat_dim = gen.shape
    _, num_pos, pos_dim = fixed_pos.shape
    _, num_neg, neg_dim = fixed_neg.shape

    if pos_dim != feat_dim or neg_dim != feat_dim:
        raise ValueError("gen, fixed_pos, and fixed_neg must share the same feature dimension")

    dtype = gen.dtype
    device = gen.device

    if weight_gen is None:
        weight_gen = torch.ones(batch_size, num_gen, device=device, dtype=dtype)
    else:
        weight_gen = weight_gen.to(device=device, dtype=dtype)

    if weight_pos is None:
        weight_pos = torch.ones(batch_size, num_pos, device=device, dtype=dtype)
    else:
        weight_pos = weight_pos.to(device=device, dtype=dtype)

    if weight_neg is None:
        weight_neg = torch.ones(batch_size, num_neg, device=device, dtype=dtype)
    else:
        weight_neg = weight_neg.to(device=device, dtype=dtype)

    if weight_gen.shape != (batch_size, num_gen):
        raise ValueError("weight_gen must have shape [B, C_g]")
    if weight_pos.shape != (batch_size, num_pos):
        raise ValueError("weight_pos must have shape [B, C_p]")
    if weight_neg.shape != (batch_size, num_neg):
        raise ValueError("weight_neg must have shape [B, C_n]")

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        dist = cdist(old_gen, targets, eps=eps)
        weighted_dist = dist * targets_w[:, None, :]
        scale = (weighted_dist.mean() / targets_w.mean().clamp_min(eps)).clamp_min(1e-3)
        scale_inputs = (scale / (feat_dim ** 0.5)).clamp_min(1e-3)

        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale.clamp_min(1e-3)

        mask = torch.eye(num_gen, device=device, dtype=dtype)[None] * 100.0
        dist_normed[:, :, :num_gen] = dist_normed[:, :, :num_gen] + mask

        total_force = torch.zeros_like(old_gen_scaled)
        info = {"scale": scale.detach()}
        split_idx = num_gen + num_neg

        for R in R_list:
            logits = -dist_normed / float(R)
            affinity = torch.softmax(logits, dim=-1)
            affinity_t = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * affinity_t).clamp_min(1e-8))
            affinity = affinity * targets_w[:, None, :]

            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            coeff_neg = -aff_neg * sum_pos

            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            coeff_pos = aff_pos * sum_neg

            coeff = torch.cat([coeff_neg, coeff_pos], dim=-1)
            force_R = torch.matmul(coeff, targets_scaled)
            coeff_sum = coeff.sum(dim=-1, keepdim=True)
            force_R = force_R - coeff_sum * old_gen_scaled

            force_norm = force_R.pow(2).mean().clamp_min(eps).sqrt()
            total_force = total_force + force_R / force_norm
            info[f"loss_R_{R}"] = force_norm.detach()

        goal_scaled = (old_gen_scaled + total_force).detach()

    gen_scaled = gen / scale_inputs
    loss = F.mse_loss(gen_scaled, goal_scaled)

    if return_info:
        return loss, info
    return loss
