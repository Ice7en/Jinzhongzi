"""Sequence views for the bidirectional full-stream temporal memory model."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.temporal_frame import (
    build_temporal_context_frame,
    load_temporal_frame_video,
)


def temporal_sequence_start(center_bin, bin_count, sequence_length):
    """Choose a fixed-length sequence centred on an observed time bin."""
    center_bin = int(center_bin)
    bin_count = int(bin_count)
    sequence_length = int(sequence_length)
    if bin_count <= 0 or sequence_length <= 0:
        raise ValueError('bin_count and sequence_length must be positive.')
    if sequence_length > bin_count:
        raise ValueError('sequence_length must not exceed bin_count.')
    if center_bin < 0 or center_bin >= bin_count:
        raise ValueError('center_bin is outside the available range.')
    return min(
        max(center_bin - sequence_length // 2, 0),
        bin_count - sequence_length,
    )


class TemporalMemoryTrainDataset(Dataset):
    """Sample contiguous full-stream frame sequences without validation data."""

    def __init__(
        self,
        root,
        whole_t,
        temporal_bin_size,
        context_bins,
        sequence_length,
        width,
        height,
        views_per_video,
        positive_frame_probability,
        random_seed,
        log_count_clip=4.0,
        cache_all_videos=True,
        cache_video_count=16,
    ):
        self.root = Path(root)
        self.whole_t = int(whole_t)
        self.temporal_bin_size = int(temporal_bin_size)
        self.context_bins = int(context_bins)
        self.sequence_length = int(sequence_length)
        self.width = int(width)
        self.height = int(height)
        self.views_per_video = int(views_per_video)
        self.positive_frame_probability = float(positive_frame_probability)
        self.random_seed = int(random_seed)
        self.log_count_clip = float(log_count_clip)
        self.cache_all_videos = bool(cache_all_videos)
        self.cache_video_count = int(cache_video_count)
        self.current_epoch = 0

        self.file_paths = sorted(self.root.glob('*.npz'))
        if not self.file_paths:
            raise RuntimeError('No npz files found in {}'.format(self.root))
        if self.context_bins < 1 or self.context_bins % 2 == 0:
            raise ValueError('context_bins must be a positive odd integer.')
        if self.sequence_length <= 0:
            raise ValueError('sequence_length must be positive.')
        if self.views_per_video <= 0:
            raise ValueError('views_per_video must be positive.')
        if not 0.0 <= self.positive_frame_probability <= 1.0:
            raise ValueError('positive_frame_probability must be in [0, 1].')
        if self.cache_video_count <= 0:
            raise ValueError('cache_video_count must be positive.')

        self._videos = {}
        self._lru = OrderedDict()
        if self.cache_all_videos:
            for video_index in range(len(self.file_paths)):
                video = self._load_video(video_index)
                if self.sequence_length > len(video.event_indices_by_bin):
                    raise ValueError(
                        'sequence_length exceeds the available temporal bins.'
                    )
                self._videos[video_index] = video

    def _load_video(self, video_index):
        return load_temporal_frame_video(
            self.file_paths[video_index],
            self.temporal_bin_size,
            self.whole_t,
        )

    def _video(self, video_index):
        cached = self._videos.get(video_index)
        if cached is not None:
            return cached
        cached = self._lru.pop(video_index, None)
        if cached is not None:
            self._lru[video_index] = cached
            return cached
        cached = self._load_video(video_index)
        if self.sequence_length > len(cached.event_indices_by_bin):
            raise ValueError(
                'sequence_length exceeds the available temporal bins.'
            )
        self._lru[video_index] = cached
        while len(self._lru) > self.cache_video_count:
            self._lru.popitem(last=False)
        return cached

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def __len__(self):
        return len(self.file_paths) * self.views_per_video

    def _sample_center_bin(self, video_index, view_index, video):
        seed = (
            self.random_seed
            + 1000003 * self.current_epoch
            + 1009 * video_index
            + view_index
        )
        rng = np.random.default_rng(seed)
        use_positive = (
            video.positive_bins.size > 0
            and rng.random() < self.positive_frame_probability
        )
        candidates = video.positive_bins if use_positive else video.occupied_bins
        if candidates.size == 0:
            raise RuntimeError('{} contains no valid event-time bins.'.format(video.name))
        return int(candidates[rng.integers(candidates.size)])

    def __getitem__(self, index):
        index = int(index)
        video_index = index // self.views_per_video
        view_index = index % self.views_per_video
        video = self._video(video_index)
        center_bin = self._sample_center_bin(video_index, view_index, video)
        start_bin = temporal_sequence_start(
            center_bin,
            len(video.event_indices_by_bin),
            self.sequence_length,
        )

        frames = []
        event_time_indices = []
        event_x = []
        event_y = []
        labels = []
        target_ids = []
        for sequence_index, temporal_bin in enumerate(
            range(start_bin, start_bin + self.sequence_length)
        ):
            frames.append(
                build_temporal_context_frame(
                    video,
                    temporal_bin,
                    self.context_bins,
                    self.width,
                    self.height,
                    self.log_count_clip,
                )
            )
            event_indices = video.event_indices_by_bin[temporal_bin]
            if event_indices.size == 0:
                continue
            locations = video.locations[event_indices]
            event_time_indices.append(
                np.full(event_indices.shape, sequence_index, dtype=np.int64)
            )
            event_x.append(locations[:, 0].astype(np.int64, copy=False))
            event_y.append(locations[:, 1].astype(np.int64, copy=False))
            labels.append(video.labels[event_indices].astype(np.float32, copy=False))
            target_ids.append(
                video.target_ids[event_indices].astype(np.int64, copy=False)
            )

        if not event_time_indices:
            raise RuntimeError('Sampled sequence contains no events.')
        return {
            'frames': np.stack(frames, axis=0),
            'event_time_indices': np.concatenate(event_time_indices),
            'event_x': np.concatenate(event_x),
            'event_y': np.concatenate(event_y),
            'labels': np.concatenate(labels),
            'target_ids': np.concatenate(target_ids),
        }


def temporal_memory_collate(samples):
    """Keep one variable-event sequence per GPU step for predictable memory."""
    if len(samples) != 1:
        raise ValueError('Temporal-memory training requires batch_size=1.')
    sample = samples[0]
    return {
        'frames': torch.from_numpy(sample['frames']).float(),
        'event_time_indices': torch.from_numpy(
            sample['event_time_indices']
        ).long(),
        'event_x': torch.from_numpy(sample['event_x']).long(),
        'event_y': torch.from_numpy(sample['event_y']).long(),
        'labels': torch.from_numpy(sample['labels']).float(),
        'target_ids': torch.from_numpy(sample['target_ids']).long(),
    }
