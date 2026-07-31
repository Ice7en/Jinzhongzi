"""Label-free per-event features computed from the raw event stream."""

import numpy as np


def build_local_activity_feature(
    locations,
    width,
    height,
    temporal_size,
    temporal_radius,
):
    """Return standardized same-pixel temporal activity for every event.

    For each event, the feature counts observations at the same image pixel in
    a symmetric temporal window.  It is calculated from ``x/y/t`` only, before
    any training-only sampling, so exactly the same representation is
    available for training, validation, and hidden-test inference.
    """
    locations = np.asarray(locations)
    width = int(width)
    height = int(height)
    temporal_size = int(temporal_size)
    temporal_radius = int(temporal_radius)

    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError("locations must have shape [N, 3+] ordered as x, y, t.")
    if width <= 0 or height <= 0 or temporal_size <= 0:
        raise ValueError("width, height, and temporal_size must be positive.")
    if temporal_radius < 0:
        raise ValueError("temporal_radius must be non-negative.")

    activity = np.zeros(locations.shape[0], dtype=np.float32)
    if locations.shape[0] == 0:
        return activity

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
        & (t < temporal_size)
    )
    if not valid.any():
        return activity

    x = x[valid]
    y = y[valid]
    t = t[valid]
    pixel_key = y * width + x
    encoded_events = np.sort(pixel_key * temporal_size + t)
    lower_bound = pixel_key * temporal_size + np.maximum(t - temporal_radius, 0)
    upper_bound = pixel_key * temporal_size + np.minimum(
        t + temporal_radius,
        temporal_size - 1,
    )
    counts = (
        np.searchsorted(encoded_events, upper_bound, side="right")
        - np.searchsorted(encoded_events, lower_bound, side="left")
    )
    values = np.log1p(counts).astype(np.float32, copy=False)
    standard_deviation = float(values.std())
    if standard_deviation > 1e-6:
        values = (values - values.mean()) / standard_deviation
    else:
        values.fill(0.0)
    activity[valid] = values
    return activity


def build_local_spatiotemporal_density_feature(
    locations,
    width,
    height,
    temporal_size,
    spatial_cell_size=3,
    temporal_cell_size=50,
    neighborhood_radius=1,
):
    """Return standardized local 3D event-density for every event.

    The feature is derived from the complete raw event stream before any
    training-only sampling. Each event receives the number of raw events in
    its neighboring spatial-temporal cells, followed by ``log1p`` and
    per-video standardization. This preserves local context for dense videos
    even when their model input is later sampled or partitioned.
    """
    locations = np.asarray(locations)
    width = int(width)
    height = int(height)
    temporal_size = int(temporal_size)
    spatial_cell_size = int(spatial_cell_size)
    temporal_cell_size = int(temporal_cell_size)
    neighborhood_radius = int(neighborhood_radius)

    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError("locations must have shape [N, 3+] ordered as x, y, t.")
    if width <= 0 or height <= 0 or temporal_size <= 0:
        raise ValueError("width, height, and temporal_size must be positive.")
    if spatial_cell_size <= 0 or temporal_cell_size <= 0:
        raise ValueError("spatial_cell_size and temporal_cell_size must be positive.")
    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative.")

    feature = np.zeros(locations.shape[0], dtype=np.float32)
    if locations.shape[0] == 0:
        return feature

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
        & (t < temporal_size)
    )
    if not valid.any():
        return feature

    x = x[valid]
    y = y[valid]
    t = t[valid]
    cells_x = (width + spatial_cell_size - 1) // spatial_cell_size
    cells_y = (height + spatial_cell_size - 1) // spatial_cell_size
    cells_t = (temporal_size + temporal_cell_size - 1) // temporal_cell_size
    cell_x = x // spatial_cell_size
    cell_y = y // spatial_cell_size
    cell_t = t // temporal_cell_size

    flat_cells = (cell_t * cells_y + cell_y) * cells_x + cell_x
    cell_counts = np.bincount(
        flat_cells,
        minlength=cells_t * cells_y * cells_x,
    ).reshape(cells_t, cells_y, cells_x)

    # A summed-volume table computes every (2r + 1)^3 cell neighborhood in
    # linear time without materializing a large sliding-window tensor.
    radius = neighborhood_radius
    window = 2 * radius + 1
    padded_counts = np.pad(
        cell_counts,
        ((radius, radius), (radius, radius), (radius, radius)),
    )
    prefix = np.zeros(
        tuple(np.asarray(padded_counts.shape) + 1),
        dtype=np.int64,
    )
    prefix[1:, 1:, 1:] = padded_counts.cumsum(0).cumsum(1).cumsum(2)
    local_counts = (
        prefix[window:, window:, window:]
        - prefix[:-window, window:, window:]
        - prefix[window:, :-window, window:]
        - prefix[window:, window:, :-window]
        + prefix[:-window, :-window, window:]
        + prefix[:-window, window:, :-window]
        + prefix[window:, :-window, :-window]
        - prefix[:-window, :-window, :-window]
    )

    values = np.log1p(local_counts[cell_t, cell_y, cell_x]).astype(
        np.float32,
        copy=False,
    )
    standard_deviation = float(values.std())
    if standard_deviation > 1e-6:
        values = (values - values.mean()) / standard_deviation
    else:
        values.fill(0.0)
    feature[valid] = values
    return feature
