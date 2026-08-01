"""Utilities for memory-bounded, time-contiguous training samples."""


def partition_event_indices(event_count, max_events_num):
    """Partition time-sorted events into consecutive slices within the budget.

    EV-UAV event files are ordered by time.  Keeping each slice contiguous
    preserves local spatiotemporal event density, unlike global random event
    subsampling.  Every original event appears in exactly one returned slice.
    """
    event_count = int(event_count)
    max_events_num = int(max_events_num)
    if event_count < 0:
        raise ValueError('event_count must be non-negative.')
    if max_events_num <= 0:
        raise ValueError('max_events_num must be positive.')

    return [
        (start, min(start + max_events_num, event_count))
        for start in range(0, event_count, max_events_num)
    ]
