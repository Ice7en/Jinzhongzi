"""
Confidence calibration head module.
Explicitly predicts detection reliability to reduce false alarms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    """Confidence prediction head for estimating prediction reliability.

    Zero-initialized design ensures initial output is neutral (sigmoid=0.5).
    """

    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=1),
        )

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, decoder_features):
        """Args:
            decoder_features: Decoder output features [B, C, H, W]

        Returns:
            torch.Tensor: Confidence logit [B, 1, H, W]
        """
        return self.layers(decoder_features)


def confidence_calibration_loss(
    confidence_logits,
    event_logits,
    labels_soft,
    hard_target=True,
):
    """Confidence calibration loss.

    With hard_target=True (default) the head predicts binary correctness:
    confidence approaches 1 at positive events and 0 at negative events,
    using a class-balanced BCE.  At inference the score is recalibrated as
    ``prob * sigmoid(conf)``, so keeping confidence near 1 for true positives
    (rather than tying it to the probability) avoids squashing predictions.

    With hard_target=False it predicts the softer reliability
    ``1 - |predicted_prob - true_label|`` via MSE.

    Args:
        confidence_logits: Confidence prediction logits [E]
        event_logits: Event prediction logits [E]
        labels_soft: Ground truth labels [E]
        hard_target: Whether to supervise hard correctness instead of soft
            reliability.

    Returns:
        torch.Tensor: Calibration loss value
    """
    if hard_target:
        positive = labels_soft > 0.5
        positive_count = positive.sum().clamp(min=1).float()
        negative_count = (~positive).sum().clamp(min=1).float()
        # Class-balanced weights summing to total count, so the loss is
        # scale-invariant in E and stays ~0.69 at init regardless of the
        # positive/negative ratio.
        total = positive_count + negative_count
        weight = torch.where(
            positive,
            0.5 * total / positive_count,
            0.5 * total / negative_count,
        )
        return F.binary_cross_entropy_with_logits(
            confidence_logits,
            labels_soft,
            weight=weight,
        )
    with torch.no_grad():
        prediction_error = (torch.sigmoid(event_logits) - labels_soft).abs()
        target_confidence = 1.0 - prediction_error

    return F.mse_loss(torch.sigmoid(confidence_logits), target_confidence)
