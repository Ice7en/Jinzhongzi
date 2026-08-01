"""Lightweight event-frame construction for the optional P3-Lite branch."""

import numpy as np


def build_event_frame(
    locations,
    polarities,
    width,
    height,
    temporal_bins,
    temporal_size,
):
    """Build log-count polarity frames from an unlabelled event sequence.

    Channels are ordered as ``[bin0-negative, bin0-positive, ...]``.  The
    caller supplies full events before any training-only point subsampling,
    so the 2D branch sees the same raw observation at train and inference.
    """
    locations = np.asarray(locations)
    polarities = np.asarray(polarities).reshape(-1)
    width = int(width)
    height = int(height)
    temporal_bins = int(temporal_bins)
    temporal_size = int(temporal_size)

    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError("locations must have shape [N, 3+] ordered as x, y, t.")
    if locations.shape[0] != polarities.shape[0]:
        raise ValueError("locations and polarities must contain the same event count.")
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive.")
    if temporal_bins <= 0 or temporal_size <= 0:
        raise ValueError("temporal_bins and temporal_size must be positive.")

    frame = np.zeros((temporal_bins * 2, height, width), dtype=np.float32)
    if locations.shape[0] == 0:
        return frame

    coordinates = locations[:, :3].astype(np.int64, copy=False)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    t = coordinates[:, 2]
    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
        & (t >= 0)
    )
    if not valid.any():
        return frame

    x = x[valid]
    y = y[valid]
    temporal_bin = np.minimum(
        (t[valid] * temporal_bins) // temporal_size,
        temporal_bins - 1,
    )
    polarity = (polarities[valid] > 0.5).astype(np.int64, copy=False)
    channel = temporal_bin * 2 + polarity
    flat_indices = channel * (height * width) + y * width + x
    np.add.at(frame.reshape(-1), flat_indices, 1.0)
    np.log1p(frame, out=frame)
    return frame
