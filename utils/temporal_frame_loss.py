"""Event-point losses for full-frame temporal segmentation."""

import math

import torch
import torch.nn.functional as functional


def generate_gaussian_soft_labels(
    event_x,
    event_y,
    labels,
    target_ids,
    event_batch_indices,
    sigma=2.5,
):
    """Convert per-event binary labels to distance-weighted soft labels.

    For each target instance (grouped by batch_index, target_id), the centroid
    is computed. Every event within the same batch item receives a soft label
    :math:`\\exp(-d_{min}^2 / (2\\sigma^2))` where :math:`d_{min}` is the
    pixel distance to the nearest target centroid. Events whose original binary
    label is 1.0 keep label 1.0 regardless of distance. Background events far
    from all targets stay at 0.
    """
    tensors = (event_x, event_y, labels, target_ids, event_batch_indices)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError('Gaussian soft-label inputs must be flat tensors.')
    if not (
        event_x.shape == event_y.shape == labels.shape
        == target_ids.shape == event_batch_indices.shape
    ):
        raise ValueError('Gaussian soft-label inputs must have matching shapes.')
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError('sigma must be positive.')
    device = labels.device
    event_count = int(labels.numel())
    soft_labels = labels.clone().to(torch.float32)

    if event_count == 0:
        return soft_labels

    event_x_f = event_x.float()
    event_y_f = event_y.float()

    # Collect target centroids per batch item
    valid_target = (
        (labels > 0.5) & (target_ids.long() > 0)
        & (event_batch_indices >= 0)
    )
    if not valid_target.any():
        return soft_labels

    centroids = {}
    for batch_idx in event_batch_indices[valid_target].unique(sorted=True):
        batch_idx = int(batch_idx.item())
        batch_mask = valid_target & (event_batch_indices == batch_idx)
        centroids[batch_idx] = []
        for tid in target_ids[batch_mask].unique(sorted=True):
            tid_mask = batch_mask & (target_ids == tid)
            cx = event_x_f[tid_mask].mean()
            cy = event_y_f[tid_mask].mean()
            centroids[batch_idx].append((cx, cy))

    if not centroids:
        return soft_labels

    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    for batch_idx in event_batch_indices.unique(sorted=True):
        batch_idx = int(batch_idx.item())
        if batch_idx not in centroids:
            continue
        batch_mask = event_batch_indices == batch_idx
        ex = event_x_f[batch_mask]
        ey = event_y_f[batch_mask]
        min_dist_sq = torch.full_like(ex, float('inf'))
        for cx, cy in centroids[batch_idx]:
            dist_sq = (ex - cx).square() + (ey - cy).square()
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
        gaussian_vals = torch.exp(-min_dist_sq * inv_2sigma2)
        # Original positives stay at 1.0
        pos_mask = labels[batch_mask] > 0.5
        gaussian_vals[pos_mask] = 1.0
        soft_labels[batch_mask] = gaussian_vals

    return soft_labels


def quality_focal_loss(
    pred_logits,
    target_soft,
    beta=2.0,
):
    """Quality Focal Loss supporting continuous soft labels.

    :math:`L = |\\sigma(p) - y|^\\beta \\cdot BCE(p, y)`

    When *target_soft* is near 0.5 the modulation factor is small, reducing
    the loss for ambiguous pixels. Hard positives (y≈1) and hard negatives
    (y≈0) that are mis-predicted receive the full focal penalty.
    """
    if pred_logits.shape != target_soft.shape:
        raise ValueError('logits and soft labels must have matching shapes.')
    beta = float(beta)
    if beta < 0.0:
        raise ValueError('beta must be non-negative.')
    target_soft = target_soft.to(dtype=pred_logits.dtype)
    ce = functional.binary_cross_entropy_with_logits(
        pred_logits, target_soft, reduction='none',
    )
    if beta == 0.0:
        return ce.mean()
    pred_sigmoid = torch.sigmoid(pred_logits)
    modulation = torch.abs(pred_sigmoid - target_soft).pow(beta)
    return (modulation * ce).mean()


def frame_balanced_quality_focal_loss(
    logits,
    soft_labels,
    event_batch_indices,
    target_positive_loss_mass=0.20,
    max_positive_weight=16.0,
    beta=2.0,
):
    """Per-frame balanced Quality Focal Loss for soft-label events.

    Same balancing logic as ``frame_balanced_event_bce`` but using QFL
    in place of BCE. Soft labels near 0.5 contribute less modulation,
    naturally down-weighting ambiguous pixels.
    """
    if logits.ndim != 1 or soft_labels.ndim != 1 or event_batch_indices.ndim != 1:
        raise ValueError('logits, soft_labels, and event_batch_indices must be flat.')
    if not (
        logits.shape == soft_labels.shape == event_batch_indices.shape
    ):
        raise ValueError('logits, soft_labels, and event_batch_indices must match.')
    target_positive_loss_mass = float(target_positive_loss_mass)
    max_positive_weight = float(max_positive_weight)
    beta = float(beta)
    if not 0.0 < target_positive_loss_mass < 1.0:
        raise ValueError('target_positive_loss_mass must be in (0, 1).')
    if max_positive_weight < 1.0:
        raise ValueError('max_positive_weight must be at least one.')
    if logits.numel() == 0:
        return logits.sum() * 0.0, {
            'positive_fraction': 0.0,
            'mean_positive_weight': 1.0,
        }

    soft_labels = soft_labels.float()
    point_loss = functional.binary_cross_entropy_with_logits(
        logits, soft_labels, reduction='none',
    )
    if beta > 0.0:
        pred_sigmoid = torch.sigmoid(logits)
        modulation = torch.abs(pred_sigmoid - soft_labels).pow(beta)
        point_loss = modulation * point_loss

    per_view_losses = []
    positive_weights = []
    for batch_index in event_batch_indices.unique(sorted=True):
        sample_mask = event_batch_indices == batch_index
        sample_labels = soft_labels[sample_mask]
        sample_loss = point_loss[sample_mask]
        positive_mask = sample_labels > 0.5
        positive_count = float(positive_mask.sum().item())
        negative_count = float((~positive_mask).sum().item())
        if positive_count == 0.0 or negative_count == 0.0:
            positive_weight = 1.0
        else:
            positive_weight = min(
                max_positive_weight,
                max(
                    1.0,
                    (
                        negative_count
                        / positive_count
                        * target_positive_loss_mass
                        / (1.0 - target_positive_loss_mass)
                    ),
                ),
            )
        weights = torch.ones_like(sample_loss)
        if positive_weight != 1.0:
            weights[positive_mask] = positive_weight
        per_view_losses.append((sample_loss * weights).sum() / weights.sum())
        positive_weights.append(positive_weight)

    return torch.stack(per_view_losses).mean(), {
        'positive_fraction': float((soft_labels > 0.5).float().mean().detach().item()),
        'mean_positive_weight': float(sum(positive_weights) / len(positive_weights)),
    }


def build_target_center_heatmaps(
    event_x,
    event_y,
    labels,
    target_ids,
    event_batch_indices,
    batch_size,
    height,
    width,
    sigma=2.5,
    radius=6,
):
    """Build one soft target-centre map for every labelled time-frame view.

    A target ID identifies a physical target in the centre temporal bin. Its
    positive event centroid is rendered as a clipped Gaussian so sparse
    targets still provide a spatially dense supervisory signal. This helper
    is used only during training; inference receives no target labels.
    """
    tensors = (event_x, event_y, labels, target_ids, event_batch_indices)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError('Target-centre inputs must be flat tensors.')
    if not (
        event_x.shape
        == event_y.shape
        == labels.shape
        == target_ids.shape
        == event_batch_indices.shape
    ):
        raise ValueError('Target-centre inputs must have matching shapes.')
    batch_size = int(batch_size)
    height = int(height)
    width = int(width)
    sigma = float(sigma)
    radius = int(radius)
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError('Target-centre map dimensions must be positive.')
    if sigma <= 0.0:
        raise ValueError('target-centre sigma must be positive.')
    if radius <= 0:
        raise ValueError('target-centre radius must be positive.')

    heatmaps = labels.new_zeros((batch_size, 1, height, width))
    if labels.numel() == 0:
        return heatmaps

    valid = (
        (labels > 0.5)
        & (target_ids.long() > 0)
        & (event_batch_indices >= 0)
        & (event_batch_indices < batch_size)
        & (event_x >= 0)
        & (event_x < width)
        & (event_y >= 0)
        & (event_y < height)
    )
    if not bool(valid.any()):
        return heatmaps

    for batch_index in event_batch_indices[valid].unique(sorted=True):
        sample_mask = valid & (event_batch_indices == batch_index)
        for target_id in target_ids[sample_mask].unique(sorted=True):
            target_mask = sample_mask & (target_ids == target_id)
            center_x = event_x[target_mask].float().mean()
            center_y = event_y[target_mask].float().mean()
            center_x_floor = int(torch.floor(center_x).item())
            center_y_floor = int(torch.floor(center_y).item())
            x_start = max(0, center_x_floor - radius)
            x_end = min(width - 1, center_x_floor + radius)
            y_start = max(0, center_y_floor - radius)
            y_end = min(height - 1, center_y_floor + radius)
            x_coordinates = torch.arange(
                x_start,
                x_end + 1,
                device=labels.device,
                dtype=labels.dtype,
            )
            y_coordinates = torch.arange(
                y_start,
                y_end + 1,
                device=labels.device,
                dtype=labels.dtype,
            )
            squared_distance = (
                (y_coordinates[:, None] - center_y).square()
                + (x_coordinates[None, :] - center_x).square()
            )
            gaussian = torch.exp(
                -squared_distance / (2.0 * sigma * sigma)
            )
            region = heatmaps[
                int(batch_index.item()),
                0,
                y_start:y_end + 1,
                x_start:x_end + 1,
            ]
            heatmaps[
                int(batch_index.item()),
                0,
                y_start:y_end + 1,
                x_start:x_end + 1,
            ] = torch.maximum(region, gaussian)
    return heatmaps


def target_center_heatmap_loss(
    logits,
    target_heatmaps,
    target_positive_loss_mass=0.20,
    max_positive_weight=512.0,
    empty_loss_weight=0.10,
):
    """Balanced BCE for soft target-centre heatmaps.

    The Gaussian target mass is much smaller than a full event frame. Each
    non-empty frame is balanced independently, while target-free views retain
    a small negative-only term so the centre head does not activate globally.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Target-centre logits must have shape [B, 1, H, W].')
    if target_heatmaps.shape != logits.shape:
        raise ValueError('Target-centre logits and heatmaps must match.')
    target_positive_loss_mass = float(target_positive_loss_mass)
    max_positive_weight = float(max_positive_weight)
    empty_loss_weight = float(empty_loss_weight)
    if not 0.0 < target_positive_loss_mass < 1.0:
        raise ValueError('target_positive_loss_mass must be in (0, 1).')
    if max_positive_weight < 1.0:
        raise ValueError('max_positive_weight must be at least one.')
    if not 0.0 <= empty_loss_weight <= 1.0:
        raise ValueError('empty_loss_weight must be in [0, 1].')

    target_heatmaps = target_heatmaps.to(dtype=logits.dtype)
    pixel_loss = functional.binary_cross_entropy_with_logits(
        logits,
        target_heatmaps,
        reduction='none',
    )
    per_view_losses = []
    positive_weights = []
    nonempty_views = 0
    pixel_count = logits.shape[2] * logits.shape[3]
    for batch_index in range(logits.shape[0]):
        sample_target = target_heatmaps[batch_index]
        sample_loss = pixel_loss[batch_index]
        target_mass = float(sample_target.detach().sum().item())
        if target_mass > 0.0:
            nonempty_views += 1
            positive_weight = min(
                max_positive_weight,
                max(
                    1.0,
                    (
                        (pixel_count - target_mass)
                        / target_mass
                        * target_positive_loss_mass
                        / (1.0 - target_positive_loss_mass)
                    ),
                ),
            )
            weights = 1.0 + sample_target * (positive_weight - 1.0)
            per_view_losses.append(
                (sample_loss * weights).sum() / weights.sum()
            )
            positive_weights.append(positive_weight)
        else:
            per_view_losses.append(sample_loss.mean() * empty_loss_weight)
            positive_weights.append(1.0)

    return torch.stack(per_view_losses).mean(), {
        'nonempty_view_fraction': float(nonempty_views / logits.shape[0]),
        'mean_positive_weight': float(
            sum(positive_weights) / len(positive_weights)
        ),
        'mean_target_mass': float(
            target_heatmaps.detach().sum().item() / logits.shape[0]
        ),
    }


def frame_balanced_event_bce(
    logits,
    labels,
    event_batch_indices,
    target_positive_loss_mass=0.20,
    max_positive_weight=16.0,
):
    """Compute BCE at event coordinates with bounded per-frame balancing.

    Each video-time view contributes one equally weighted loss. The positive
    factor is chosen so positives make up at most the configured fraction of
    that view's total loss mass. This increases recall without allowing an
    exceptionally sparse frame to dominate the whole minibatch.
    """
    if logits.ndim != 1 or labels.ndim != 1 or event_batch_indices.ndim != 1:
        raise ValueError('logits, labels, and event_batch_indices must be flat.')
    if not (
        logits.shape == labels.shape == event_batch_indices.shape
    ):
        raise ValueError('logits, labels, and event_batch_indices must match.')
    target_positive_loss_mass = float(target_positive_loss_mass)
    max_positive_weight = float(max_positive_weight)
    if not 0.0 < target_positive_loss_mass < 1.0:
        raise ValueError('target_positive_loss_mass must be in (0, 1).')
    if max_positive_weight < 1.0:
        raise ValueError('max_positive_weight must be at least one.')
    if logits.numel() == 0:
        return logits.sum() * 0.0, {
            'positive_fraction': 0.0,
            'mean_positive_weight': 1.0,
        }

    labels = labels.float()
    point_loss = functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction='none',
    )
    per_view_losses = []
    positive_weights = []
    for batch_index in event_batch_indices.unique(sorted=True):
        sample_mask = event_batch_indices == batch_index
        sample_labels = labels[sample_mask]
        sample_loss = point_loss[sample_mask]
        positive_mask = sample_labels > 0.5
        positive_count = int(positive_mask.sum().item())
        negative_count = int((~positive_mask).sum().item())
        if positive_count == 0 or negative_count == 0:
            positive_weight = 1.0
        else:
            positive_weight = min(
                max_positive_weight,
                max(
                    1.0,
                    (
                        negative_count
                        / float(positive_count)
                        * target_positive_loss_mass
                        / (1.0 - target_positive_loss_mass)
                    ),
                ),
            )
        weights = torch.ones_like(sample_loss)
        if positive_weight != 1.0:
            weights[positive_mask] = positive_weight
        per_view_losses.append((sample_loss * weights).sum() / weights.sum())
        positive_weights.append(positive_weight)

    return torch.stack(per_view_losses).mean(), {
        'positive_fraction': float(labels.mean().detach().item()),
        'mean_positive_weight': float(sum(positive_weights) / len(positive_weights)),
    }


def target_group_coverage_loss(
    logits,
    labels,
    target_ids,
    event_batch_indices,
    score_floor=0.70,
    correct_fraction=0.0001,
):
    """Ensure each labelled target-time group has enough confident events.

    Challenge Pd marks a target group detected once a very small fraction of
    its positive events is classified correctly. This loss uses the kth
    highest positive logit, where k is that fraction rounded up, then applies
    a hinge at a fixed score margin. Already covered targets produce no loss,
    so ordinary point-wise BCE remains responsible for IoU and false alarms.
    """
    if not (
        logits.ndim
        == labels.ndim
        == target_ids.ndim
        == event_batch_indices.ndim
        == 1
    ):
        raise ValueError('Target coverage inputs must be flat tensors.')
    if not (
        logits.shape
        == labels.shape
        == target_ids.shape
        == event_batch_indices.shape
    ):
        raise ValueError('Target coverage inputs must have matching shapes.')
    score_floor = float(score_floor)
    correct_fraction = float(correct_fraction)
    if not 0.0 < score_floor < 1.0:
        raise ValueError('score_floor must be in (0, 1).')
    if not 0.0 < correct_fraction <= 1.0:
        raise ValueError('correct_fraction must be in (0, 1].')
    if logits.numel() == 0:
        return logits.sum() * 0.0, {
            'target_group_count': 0,
            'uncovered_group_count': 0,
        }

    target_ids = target_ids.long()
    labels = labels.float()
    floor_logit = math.log(score_floor / (1.0 - score_floor))
    group_losses = []
    uncovered_group_count = 0
    for batch_index in event_batch_indices.unique(sorted=True):
        batch_mask = event_batch_indices == batch_index
        target_mask = batch_mask & (labels > 0.5) & (target_ids != 0)
        for target_id in target_ids[target_mask].unique(sorted=True):
            group_mask = target_mask & (target_ids == target_id)
            group_logits = logits[group_mask]
            required_count = max(
                1,
                int(math.ceil(group_logits.numel() * correct_fraction)),
            )
            kth_logit = torch.topk(
                group_logits,
                k=required_count,
                largest=True,
                sorted=False,
            ).values.min()
            group_loss = functional.relu(kth_logit.new_tensor(floor_logit) - kth_logit)
            group_losses.append(group_loss)

    if not group_losses:
        return logits.sum() * 0.0, {
            'target_group_count': 0,
            'uncovered_group_count': 0,
        }
    group_loss_tensor = torch.stack(group_losses)
    uncovered_group_count = int(
        (group_loss_tensor.detach() > 0.0).sum().item()
    )
    return group_loss_tensor.mean(), {
        'target_group_count': len(group_losses),
        'uncovered_group_count': uncovered_group_count,
    }


def trajectory_extrapolation_loss(
    logits,
    target_centers,
    min_known_points=3,
    margin_logit=1.0,
):
    """Trajectory extrapolation consistency loss.

    Fits linear trajectories from known target positions and enforces
    high confidence at extrapolated positions in unlabeled time steps.

    Args:
        logits (torch.Tensor): Predicted logits [B, T, 1, H, W]
        target_centers (dict): Target center coordinates
            {batch_idx: {target_id: [(t, x, y), ...]}}
        min_known_points (int): Minimum known points for linear fit (>=3)
        margin_logit (float): Target logit value (~sigmoid(1.0) ~ 0.73)

    Returns:
        torch.Tensor: Extrapolation loss value
        dict: Statistics (extrapolated point count, etc.)
    """
    losses = []
    B, T, _, H, W = logits.shape
    extrapolated_count = 0

    for b in range(B):
        if b not in target_centers:
            continue

        for target_id, observations in target_centers[b].items():
            if len(observations) < min_known_points:
                continue

            t_indices = torch.tensor(
                [obs[0] for obs in observations],
                dtype=torch.float32,
                device=logits.device,
            )
            x_coords = torch.tensor(
                [obs[1] for obs in observations],
                dtype=torch.float32,
                device=logits.device,
            )
            y_coords = torch.tensor(
                [obs[2] for obs in observations],
                dtype=torch.float32,
                device=logits.device,
            )

            ones = torch.ones_like(t_indices)
            A = torch.stack([t_indices, ones], dim=1)

            try:
                sol_x = torch.linalg.lstsq(
                    A, x_coords.unsqueeze(1),
                ).solution.squeeze()
                sol_y = torch.linalg.lstsq(
                    A, y_coords.unsqueeze(1),
                ).solution.squeeze()
            except RuntimeError:
                continue

            known_times = {int(obs[0]) for obs in observations}

            for t_step in range(T):
                if t_step in known_times:
                    continue

                px = sol_x[0] * t_step + sol_x[1]
                py = sol_y[0] * t_step + sol_y[1]

                if not (0 <= px < W and 0 <= py < H):
                    continue

                grid_x = int(px)
                grid_y = int(py)
                sampled_logit = logits[b, t_step, 0, grid_y, grid_x]

                loss = functional.relu(margin_logit - sampled_logit)
                losses.append(loss)
                extrapolated_count += 1

    stats = {
        'extrapolated_points': extrapolated_count,
    }

    if not losses:
        return logits.sum() * 0.0, stats

    return torch.stack(losses).mean(), stats


def trajectory_extrapolation_loss_p23(
    logit_maps,
    video_indices,
    center_bins,
    event_x,
    event_y,
    labels,
    target_ids,
    event_batch_indices,
    min_known_points=3,
    margin_logit=1.0,
):
    """Trajectory extrapolation loss adapted for P23 per-view training.

    Groups per-view predictions by video, sorts them temporally, fits linear
    trajectories to known target positions, and enforces high confidence at
    extrapolated positions in unobserved time steps.

    Args:
        logit_maps: [N, 1, H, W] per-view prediction logits
        video_indices: [N] video index per view
        center_bins: [N] time-bin index per view
        event_x, event_y: flat event coordinates
        labels: flat event labels
        target_ids: flat event target IDs
        event_batch_indices: [E] maps each event to its batch item
        min_known_points: minimum known time steps for linear fit
        margin_logit: target logit value (~sigmoid(1.0) ≈ 0.73)

    Returns:
        loss tensor, stats dict
    """
    N, _, H, W = logit_maps.shape
    device = logit_maps.device

    video_indices_np = video_indices.detach().cpu().numpy()
    center_bins_np = center_bins.detach().cpu().numpy()
    event_x_np = event_x.detach().cpu().numpy()
    event_y_np = event_y.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    target_ids_np = target_ids.detach().cpu().numpy()
    event_batch_indices_np = event_batch_indices.detach().cpu().numpy()

    losses = []
    extrapolated_count = 0
    targets_processed = 0

    unique_videos = sorted(set(int(v) for v in video_indices_np))
    for video_idx in unique_videos:
        view_mask = video_indices_np == video_idx
        view_indices = [i for i, m in enumerate(view_mask) if m]
        if len(view_indices) < 2:
            continue

        sorted_views = sorted(view_indices, key=lambda i: center_bins_np[i])
        local_t_to_batch = {t: bi for t, bi in enumerate(sorted_views)}

        target_centers = {}
        for local_t, batch_i in enumerate(sorted_views):
            event_mask = event_batch_indices_np == batch_i
            if not event_mask.any():
                continue
            for ei in range(len(event_x_np)):
                if not event_mask[ei]:
                    continue
                if labels_np[ei] < 0.5:
                    continue
                tid = int(target_ids_np[ei])
                if tid <= 0:
                    continue
                if tid not in target_centers:
                    target_centers[tid] = []
                target_centers[tid].append((
                    local_t,
                    float(event_x_np[ei]),
                    float(event_y_np[ei]),
                ))

        T = len(sorted_views)
        for tid, observations in target_centers.items():
            if len(observations) < min_known_points:
                continue

            unique_obs = {}
            for t, x, y in observations:
                if t not in unique_obs:
                    unique_obs[t] = ([], [])
                unique_obs[t][0].append(x)
                unique_obs[t][1].append(y)
            merged = [
                (t, sum(xs) / len(xs), sum(ys) / len(ys))
                for t, (xs, ys) in sorted(unique_obs.items())
            ]

            if len(merged) < min_known_points:
                continue

            t_tensor = torch.tensor(
                [m[0] for m in merged], dtype=torch.float32, device=device)
            x_tensor = torch.tensor(
                [m[1] for m in merged], dtype=torch.float32, device=device)
            y_tensor = torch.tensor(
                [m[2] for m in merged], dtype=torch.float32, device=device)

            ones = torch.ones_like(t_tensor)
            A = torch.stack([t_tensor, ones], dim=1)

            try:
                sol_x = torch.linalg.lstsq(
                    A, x_tensor.unsqueeze(1)).solution.squeeze()
                sol_y = torch.linalg.lstsq(
                    A, y_tensor.unsqueeze(1)).solution.squeeze()
            except RuntimeError:
                continue

            known_times = {m[0] for m in merged}
            targets_processed += 1

            for t_step in range(T):
                if t_step in known_times:
                    continue

                px = sol_x[0] * t_step + sol_x[1]
                py = sol_y[0] * t_step + sol_y[1]

                if not (0 <= px < W and 0 <= py < H):
                    continue

                batch_i = local_t_to_batch[t_step]
                grid_x = int(px)
                grid_y = int(py)
                sampled_logit = logit_maps[batch_i, 0, grid_y, grid_x]

                loss = torch.nn.functional.relu(margin_logit - sampled_logit)
                losses.append(loss)
                extrapolated_count += 1

    stats = {
        'extrapolated_points': extrapolated_count,
        'targets_processed': targets_processed,
    }

    if not losses:
        return logit_maps.sum() * 0.0, stats

    return torch.stack(losses).mean(), stats


def trajectory_extrapolation_loss_memory(
    logit_maps,
    event_time_indices,
    event_x,
    event_y,
    labels,
    target_ids,
    min_known_points=3,
    margin_logit=1.0,
):
    """Trajectory extrapolation loss for single-video temporal-memory sequences.

    Groups positive events in one contiguous sequence by target id, fits a
    linear trajectory, and enforces high confidence at extrapolated positions
    in unobserved time steps.

    Args:
        logit_maps: [T, 1, H, W] logits for one sequence
        event_time_indices: [E] sequence time index per event
        event_x, event_y: [E] flat event coordinates
        labels: [E] 0/1 event labels
        target_ids: [E] target id per event (positive events have tid > 0)
        min_known_points: minimum known time steps for linear fit
        margin_logit: target logit value (~sigmoid(1.0) ~ 0.73)

    Returns:
        loss tensor, stats dict
    """
    T, _, H, W = logit_maps.shape
    device = logit_maps.device

    event_time_indices_np = event_time_indices.detach().cpu().numpy()
    event_x_np = event_x.detach().cpu().numpy()
    event_y_np = event_y.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    target_ids_np = target_ids.detach().cpu().numpy()

    losses = []
    extrapolated_count = 0
    targets_processed = 0

    target_centers = {}
    for ei in range(len(event_x_np)):
        if labels_np[ei] < 0.5:
            continue
        tid = int(target_ids_np[ei])
        if tid <= 0:
            continue
        if tid not in target_centers:
            target_centers[tid] = []
        target_centers[tid].append((
            int(event_time_indices_np[ei]),
            float(event_x_np[ei]),
            float(event_y_np[ei]),
        ))

    for tid, observations in target_centers.items():
        if len(observations) < min_known_points:
            continue

        unique_obs = {}
        for t, x, y in observations:
            if t not in unique_obs:
                unique_obs[t] = ([], [])
            unique_obs[t][0].append(x)
            unique_obs[t][1].append(y)
        merged = [
            (t, sum(xs) / len(xs), sum(ys) / len(ys))
            for t, (xs, ys) in sorted(unique_obs.items())
        ]
        if len(merged) < min_known_points:
            continue

        t_tensor = torch.tensor(
            [m[0] for m in merged], dtype=torch.float32, device=device)
        x_tensor = torch.tensor(
            [m[1] for m in merged], dtype=torch.float32, device=device)
        y_tensor = torch.tensor(
            [m[2] for m in merged], dtype=torch.float32, device=device)

        ones = torch.ones_like(t_tensor)
        A = torch.stack([t_tensor, ones], dim=1)

        try:
            sol_x = torch.linalg.lstsq(
                A, x_tensor.unsqueeze(1)).solution.squeeze()
            sol_y = torch.linalg.lstsq(
                A, y_tensor.unsqueeze(1)).solution.squeeze()
        except RuntimeError:
            continue

        known_times = {m[0] for m in merged}
        targets_processed += 1

        for t_step in range(T):
            if t_step in known_times:
                continue

            px = sol_x[0] * t_step + sol_x[1]
            py = sol_y[0] * t_step + sol_y[1]

            if not (0 <= px < W and 0 <= py < H):
                continue

            grid_x = int(px)
            grid_y = int(py)
            sampled_logit = logit_maps[t_step, 0, grid_y, grid_x]

            loss = torch.nn.functional.relu(margin_logit - sampled_logit)
            losses.append(loss)
            extrapolated_count += 1

    stats = {
        'extrapolated_points': extrapolated_count,
        'targets_processed': targets_processed,
    }

    if not losses:
        return logit_maps.sum() * 0.0, stats

    return torch.stack(losses).mean(), stats
