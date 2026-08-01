"""Detection-aware auxiliary loss aligned with Challenge 2 target frames."""

import math

import torch


def _zero_loss(predictions):
    return predictions.reshape(-1).sum() * 0


def target_frame_detection_loss(
    predictions,
    labels,
    target_ids,
    locations,
    prediction_threshold,
    correct_threshold,
    temporal_bin_size,
):
    """Encourage each ground-truth target frame to contain enough detections.

    Challenge 2 treats each ``(batch, target_id, temporal_bin)`` group as
    detected when the proportion of events whose score reaches the official
    prediction threshold is at least ``correct_threshold``. This loss applies
    a hinge penalty to the highest-scoring required events in every such group.
    It is differentiable with respect to the selected event scores and leaves
    background supervision to the existing STC loss.
    """
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    target_ids = target_ids.reshape(-1).long()

    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError('locations must have shape [N, >=4].')
    if not (
        predictions.numel()
        == labels.numel()
        == target_ids.numel()
        == locations.shape[0]
    ):
        raise ValueError('Prediction, label, target-id, and location counts must match.')
    if not 0 < float(prediction_threshold) <= 1:
        raise ValueError('prediction_threshold must be in (0, 1].')
    if not 0 < float(correct_threshold) <= 1:
        raise ValueError('correct_threshold must be in (0, 1].')
    if int(temporal_bin_size) <= 0:
        raise ValueError('temporal_bin_size must be positive.')

    # ``evalute.roc_update`` uses open temporal intervals
    # ``(t > start) & (t < end)``.  Events on a 50-unit boundary are not
    # scored for Pd, so they must not affect this score-aligned loss either.
    event_times = locations[:, 3].long()
    target_mask = (
        (labels > 0.5)
        & (target_ids > 0)
        & (torch.remainder(event_times, int(temporal_bin_size)) != 0)
    )
    if not torch.any(target_mask):
        return _zero_loss(predictions), 0, 0

    scores = torch.clamp(predictions[target_mask], min=0, max=1)
    target_ids = target_ids[target_mask]
    batch_ids = locations[target_mask, 0].long()
    time_bins = torch.div(
        event_times[target_mask],
        int(temporal_bin_size),
        rounding_mode='floor',
    )

    # Build sortable IDs for (batch, target ID, official temporal bin).
    target_stride = int(target_ids.max().item()) + 1
    time_stride = int(time_bins.max().item()) + 1
    group_keys = (
        (batch_ids * target_stride + target_ids) * time_stride + time_bins
    )
    group_keys, order = torch.sort(group_keys)
    scores = scores[order]
    _, counts = torch.unique_consecutive(group_keys, return_counts=True)

    group_count = int(counts.numel())
    max_group_size = int(counts.max().item())
    group_ids = torch.repeat_interleave(
        torch.arange(group_count, device=scores.device, dtype=torch.long),
        counts,
    )
    group_starts = torch.cumsum(counts, dim=0) - counts
    event_starts = torch.repeat_interleave(group_starts, counts)
    within_group_index = (
        torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
        - event_starts
    )

    # One scatter and one top-k avoid a Python loop over target frames.
    flat_indices = group_ids * max_group_size + within_group_index
    padded_scores = scores.new_zeros(group_count * max_group_size).scatter(
        0,
        flat_indices,
        scores,
    )
    grouped_scores = padded_scores.reshape(group_count, max_group_size)

    required_hits = torch.ceil(
        counts.to(dtype=scores.dtype) * float(correct_threshold)
    ).long().clamp(min=1)
    max_required_hits = int(required_hits.max().item())
    top_scores = torch.topk(
        grouped_scores,
        k=max_required_hits,
        dim=1,
        largest=True,
        sorted=False,
    ).values
    required_mask = (
        torch.arange(max_required_hits, device=scores.device).unsqueeze(0)
        < required_hits.unsqueeze(1)
    )
    deficits = torch.relu(float(prediction_threshold) - top_scores)
    group_loss = (
        (deficits * required_mask.to(dtype=scores.dtype)).sum(dim=1)
        / required_hits.to(dtype=scores.dtype)
    )
    missed_groups = int(
        ((deficits > 0) & required_mask).any(dim=1).sum().item()
    )
    return group_loss.mean(), group_count, missed_groups
