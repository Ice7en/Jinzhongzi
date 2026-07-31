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


def confidence_calibration_loss(confidence_logits, event_logits, labels_soft):
    """Confidence calibration loss.

    Target confidence = 1 - |predicted_prob - true_label|.
    The more accurate the prediction, the higher the confidence should be.

    Args:
        confidence_logits: Confidence prediction logits [B, 1, H, W]
        event_logits: Event prediction logits [B, 1, H, W]
        labels_soft: Ground truth labels [B, 1, H, W]

    Returns:
        torch.Tensor: Calibration loss value
    """
    with torch.no_grad():
        prediction_error = (torch.sigmoid(event_logits) - labels_soft).abs()
        target_confidence = 1.0 - prediction_error

    return F.mse_loss(torch.sigmoid(confidence_logits), target_confidence)
