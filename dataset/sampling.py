"""Training-only event sampling strategies."""

import numpy as np


def target_context_mask(
    labels,
    locations,
    width,
    height,
    temporal_size,
    spatial_cell_size=3,
    temporal_cell_size=50,
    spatial_radius_cells=1,
    temporal_radius_cells=1,
):
    """Mark events in coarse spatiotemporal cells near labeled targets.

    This training-only diagnostic primitive does not select samples.  It
    identifies the event context that would be available around target cells
    before a budgeted sampler chooses background events.
    """
    labels = np.asarray(labels).reshape(-1)
    locations = np.asarray(locations)
    width = int(width)
    height = int(height)
    temporal_size = int(temporal_size)
    spatial_cell_size = int(spatial_cell_size)
    temporal_cell_size = int(temporal_cell_size)
    spatial_radius_cells = int(spatial_radius_cells)
    temporal_radius_cells = int(temporal_radius_cells)

    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError('locations must have shape [N, 3+] ordered as x, y, t.')
    if labels.shape[0] != locations.shape[0]:
        raise ValueError('labels and locations must have the same length.')
    if width <= 0 or height <= 0 or temporal_size <= 0:
        raise ValueError('width, height, and temporal_size must be positive.')
    if spatial_cell_size <= 0 or temporal_cell_size <= 0:
        raise ValueError('cell sizes must be positive.')
    if spatial_radius_cells < 0 or temporal_radius_cells < 0:
        raise ValueError('cell radii must be non-negative.')

    event_count = labels.shape[0]
    context = np.zeros(event_count, dtype=bool)
    if event_count == 0:
        return context

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
    positive = valid & (labels > 0.5)
    if not np.any(positive):
        return context

    cells_x = (width + spatial_cell_size - 1) // spatial_cell_size
    cells_y = (height + spatial_cell_size - 1) // spatial_cell_size
    cells_t = (temporal_size + temporal_cell_size - 1) // temporal_cell_size
    cell_x = x[valid] // spatial_cell_size
    cell_y = y[valid] // spatial_cell_size
    cell_t = t[valid] // temporal_cell_size

    positive_cells = np.zeros((cells_t, cells_y, cells_x), dtype=bool)
    positive_cells[
        t[positive] // temporal_cell_size,
        y[positive] // spatial_cell_size,
        x[positive] // spatial_cell_size,
    ] = True
    context_cells = np.zeros_like(positive_cells)

    for delta_t in range(-temporal_radius_cells, temporal_radius_cells + 1):
        source_t_start = max(0, -delta_t)
        source_t_end = cells_t - max(0, delta_t)
        destination_t_start = max(0, delta_t)
        destination_t_end = cells_t - max(0, -delta_t)
        for delta_y in range(-spatial_radius_cells, spatial_radius_cells + 1):
            source_y_start = max(0, -delta_y)
            source_y_end = cells_y - max(0, delta_y)
            destination_y_start = max(0, delta_y)
            destination_y_end = cells_y - max(0, -delta_y)
            for delta_x in range(-spatial_radius_cells, spatial_radius_cells + 1):
                source_x_start = max(0, -delta_x)
                source_x_end = cells_x - max(0, delta_x)
                destination_x_start = max(0, delta_x)
                destination_x_end = cells_x - max(0, -delta_x)
                context_cells[
                    destination_t_start:destination_t_end,
                    destination_y_start:destination_y_end,
                    destination_x_start:destination_x_end,
                ] |= positive_cells[
                    source_t_start:source_t_end,
                    source_y_start:source_y_end,
                    source_x_start:source_x_end,
                ]

    context[valid] = context_cells[cell_t, cell_y, cell_x]
    return context


def density_dual_view_modes(
    event_count,
    event_count_cutoff,
    density_dual_view_enabled=False,
):
    """Return the training views required for one source video.

    A dense video contributes its existing target-preserving view and one
    uniform view.  The latter exposes the model to the natural foreground
    ratio it will encounter during label-free chunked inference.
    """
    event_count = int(event_count)
    event_count_cutoff = int(event_count_cutoff)
    if event_count < 0:
        raise ValueError('event_count must be non-negative.')
    if event_count_cutoff <= 0:
        raise ValueError('event_count_cutoff must be positive.')

    if density_dual_view_enabled and event_count > event_count_cutoff:
        return ('target_preserving', 'uniform')
    return ('standard',)


def dense_target_oversample_repeats(
    event_count,
    event_count_cutoff,
    dense_target_oversampling_enabled=False,
    factor=1,
):
    """Return how many target-preserving training views a video contributes."""
    event_count = int(event_count)
    event_count_cutoff = int(event_count_cutoff)
    factor = int(factor)
    if event_count < 0:
        raise ValueError('event_count must be non-negative.')
    if event_count_cutoff <= 0:
        raise ValueError('event_count_cutoff must be positive.')
    if factor <= 0:
        raise ValueError('factor must be positive.')

    if dense_target_oversampling_enabled and event_count > event_count_cutoff:
        return factor
    return 1


def dense_specialist_view_count(
    event_count,
    event_count_cutoff,
    dense_specialist_enabled=False,
    views_per_video=1,
):
    """Return uniform P8-matched views for a dense-scene specialist.

    Unlike P9, this mode is intended for a separate model and exposes no
    low-density source videos.  Each returned view is sampled uniformly, so
    its foreground/background ratio matches a random P8 inference partition.
    """
    event_count = int(event_count)
    event_count_cutoff = int(event_count_cutoff)
    views_per_video = int(views_per_video)
    if event_count < 0:
        raise ValueError('event_count must be non-negative.')
    if event_count_cutoff <= 0:
        raise ValueError('event_count_cutoff must be positive.')
    if views_per_video <= 0:
        raise ValueError('views_per_video must be positive.')
    if dense_specialist_enabled and event_count > event_count_cutoff:
        return views_per_video
    return 0


def dense_specialist_training_view(target_preserving_enabled=False):
    """Choose the label-aware sampling policy for a dense-only expert."""
    return 'target_preserving' if target_preserving_enabled else 'uniform'


def select_training_event_indices(
    labels,
    max_events_num,
    target_preserving_enabled=False,
    rng=None,
):
    """Select a fixed-size subset of events for one training video.

    With target preservation disabled, this is the original uniform random
    sampling behavior. With it enabled, all positive training events are kept
    whenever they fit in the budget and only background events are subsampled.
    This function must only be called for the training split.
    """
    labels = np.asarray(labels).reshape(-1)
    event_count = len(labels)
    max_events_num = int(max_events_num)
    if max_events_num <= 0:
        raise ValueError('max_events_num must be positive.')
    # No sampling is needed when the complete stream fits the budget.  Keeping
    # its original ordering matters for temporal-chunk training.
    if event_count <= max_events_num:
        return np.arange(event_count, dtype=np.int64)

    rng = np.random if rng is None else rng
    if not target_preserving_enabled:
        return rng.choice(event_count, max_events_num, replace=False)

    positive_indices = np.flatnonzero(labels > 0.5)
    if len(positive_indices) >= max_events_num:
        return rng.choice(positive_indices, max_events_num, replace=False)

    background_indices = np.flatnonzero(labels <= 0.5)
    background_budget = max_events_num - len(positive_indices)
    sampled_background = rng.choice(
        background_indices,
        background_budget,
        replace=False,
    )
    selected_indices = np.concatenate((positive_indices, sampled_background))
    rng.shuffle(selected_indices)
    return selected_indices
