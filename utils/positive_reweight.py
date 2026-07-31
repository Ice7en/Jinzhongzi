"""Training-only positive-event loss reweighting helpers."""

import torch


def validate_positive_stc_floor(floor):
    floor = float(floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError('positive STC floor must be in [0, 1].')
    return floor


def apply_positive_stc_floor(stc_weights, labels, floor):
    """Raise weak-STC positive events to a minimum loss weight.

    The original STC loss can almost ignore sparse or fast target events when
    their local support is low. This helper only changes positive-event
    weights; background weights remain unchanged.
    """
    floor = validate_positive_stc_floor(floor)
    low_support_positive = (labels > 0.5) & (stc_weights < floor)
    # PyTorch 1.9 CUDA can fail on boolean-indexed in-place assignment.
    positive_weights = torch.where(
        low_support_positive,
        torch.full_like(stc_weights, floor),
        stc_weights,
    )
    return positive_weights, int(low_support_positive.sum().item())
