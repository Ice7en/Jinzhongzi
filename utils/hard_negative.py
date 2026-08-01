"""Training-only helpers for conservative background hard-negative mining."""

import math

import torch


def top_ratio_background_bce(predictions, labels, ratio, eps=1e-5):
    """Return BCE against zero for the highest-scoring background events.

    ``ratio`` is applied only to events with a background label. Selecting by
    rank keeps the loss focused on the predictions most likely to become
    false alarms while keeping its cost independent of the total event count.
    """
    if not 0 < ratio <= 1:
        raise ValueError('hard-negative ratio must be in (0, 1].')

    background_predictions = predictions[labels < 0.5]
    if background_predictions.numel() == 0:
        return predictions.sum() * 0, 0

    hard_count = max(1, int(math.ceil(background_predictions.numel() * ratio)))
    hard_predictions = torch.topk(
        background_predictions,
        k=hard_count,
        largest=True,
        sorted=False,
    ).values
    hard_predictions = torch.clamp(hard_predictions, min=0, max=1)
    loss = -torch.log(1 - hard_predictions + eps).mean()
    return loss, hard_count
