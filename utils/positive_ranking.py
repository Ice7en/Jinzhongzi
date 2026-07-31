"""Hard positive/background ranking loss for event confidence separation."""

import math

import torch


def positive_hard_ranking_loss(predictions, labels, ratio, margin):
    """Separate low-confidence positives from high-confidence backgrounds.

    The loss selects only the bottom positive and top background tails. Using
    the two tail means keeps memory linear in the event count, including for
    full-stream training, while directly improving the decision boundary.
    """
    if not 0 < float(ratio) <= 1:
        raise ValueError('ratio must be in (0, 1].')
    if float(margin) < 0:
        raise ValueError('margin must be non-negative.')

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    if predictions.numel() != labels.numel():
        raise ValueError('predictions and labels must have the same length.')

    positive_scores = predictions[labels > 0.5]
    background_scores = predictions[labels <= 0.5]
    if positive_scores.numel() == 0 or background_scores.numel() == 0:
        return predictions.sum() * 0, 0, 0

    positive_count = max(1, int(math.ceil(positive_scores.numel() * float(ratio))))
    background_count = max(1, int(math.ceil(background_scores.numel() * float(ratio))))
    hard_positive_scores = torch.topk(
        positive_scores,
        k=min(positive_count, positive_scores.numel()),
        largest=False,
        sorted=False,
    ).values
    hard_background_scores = torch.topk(
        background_scores,
        k=min(background_count, background_scores.numel()),
        largest=True,
        sorted=False,
    ).values
    loss = torch.relu(
        float(margin)
        + hard_background_scores.mean()
        - hard_positive_scores.mean()
    )
    return loss, int(hard_positive_scores.numel()), int(hard_background_scores.numel())
