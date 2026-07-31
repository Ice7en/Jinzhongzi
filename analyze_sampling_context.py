"""Measure target-neighborhood context lost by budgeted event sampling."""

import argparse
from pathlib import Path

import numpy as np

from dataset.sampling import target_context_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit target-adjacent event context on the training split.'
    )
    parser.add_argument(
        '--train-root',
        type=Path,
        default=Path('dataset/训练集、验证集/train'),
    )
    parser.add_argument('--max-events', type=int, default=100000)
    parser.add_argument('--width', type=int, default=346)
    parser.add_argument('--height', type=int, default=260)
    parser.add_argument('--temporal-size', type=int, default=8000)
    parser.add_argument('--spatial-cell-size', type=int, default=3)
    parser.add_argument('--temporal-cell-size', type=int, default=50)
    parser.add_argument('--spatial-radius-cells', type=int, default=1)
    parser.add_argument('--temporal-radius-cells', type=int, default=1)
    return parser.parse_args()


def expected_background_retention(event_count, positive_count, max_events):
    background_count = event_count - positive_count
    if background_count <= 0:
        return 1.0
    background_budget = max(0, int(max_events) - positive_count)
    return min(1.0, background_budget / background_count)


def main():
    args = parse_args()
    paths = sorted(args.train_root.glob('*.npz'))
    if not paths:
        raise FileNotFoundError('No training npz files found in: {}'.format(args.train_root))

    dense_rows = []
    for path in paths:
        with np.load(path) as archive:
            locations = archive['ev_loc']
            labels = archive['evs_norm'][:, 4]
        event_count = len(labels)
        positive_count = int(np.count_nonzero(labels > 0.5))
        if event_count <= args.max_events:
            continue
        context = target_context_mask(
            labels,
            locations,
            width=args.width,
            height=args.height,
            temporal_size=args.temporal_size,
            spatial_cell_size=args.spatial_cell_size,
            temporal_cell_size=args.temporal_cell_size,
            spatial_radius_cells=args.spatial_radius_cells,
            temporal_radius_cells=args.temporal_radius_cells,
        )
        background = labels <= 0.5
        context_background = int(np.count_nonzero(context & background))
        background_count = int(np.count_nonzero(background))
        retention = expected_background_retention(
            event_count,
            positive_count,
            args.max_events,
        )
        dense_rows.append((
            path.stem,
            event_count,
            positive_count,
            context_background,
            background_count,
            retention,
        ))

    if not dense_rows:
        print('No training videos exceed max_events={}.'.format(args.max_events))
        return

    print(
        'target context audit: {} dense videos, max_events={}, '
        'cells={}x{}x{}, radii={}x{}'.format(
            len(dense_rows),
            args.max_events,
            args.spatial_cell_size,
            args.spatial_cell_size,
            args.temporal_cell_size,
            args.spatial_radius_cells,
            args.temporal_radius_cells,
        )
    )
    print(
        'video       events  positive  context_bg  context_share  '
        'uniform_keep  context_lost'
    )
    for row in sorted(dense_rows, key=lambda item: item[1], reverse=True):
        name, events, positive, context_background, background, retention = row
        share = context_background / background if background else 0.0
        lost = context_background * (1.0 - retention)
        print(
            '{:<10} {:>7} {:>9} {:>11} {:>13.3%} {:>12.3%} {:>13.0f}'.format(
                name,
                events,
                positive,
                context_background,
                share,
                retention,
                lost,
            )
        )

    total_context = sum(row[3] for row in dense_rows)
    expected_lost = sum(row[3] * (1.0 - row[5]) for row in dense_rows)
    print(
        '\naggregate context background events: {}; expected P1b uniform loss: {} ({:.2%})'.format(
            total_context,
            round(expected_lost),
            expected_lost / total_context if total_context else 0.0,
        )
    )


if __name__ == '__main__':
    main()
