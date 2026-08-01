import os
import torch
import numpy as np
from dataset.basedataset import BaseDataLoader
from dataset.event_frame import build_event_frame
from dataset.event_features import (
    build_local_activity_feature,
    build_local_spatiotemporal_density_feature,
)
from dataset.sampling import (
    dense_specialist_training_view,
    dense_specialist_view_count,
    dense_target_oversample_repeats,
    density_dual_view_modes,
    select_training_event_indices,
)
from dataset.temporal_chunks import partition_event_indices
from utils.spatial_tta import horizontal_flip_event_inputs, padded_feature_width

class EvUAV(BaseDataLoader):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)

        self.mode = mode
        self.root = os.path.join(self.root,mode)
        self.file_list = os.listdir(self.root)
        self.source_video_count = len(self.file_list)
        self.p11_local_activity_enabled = bool(
            getattr(self.configs, 'p11_local_activity_enabled', False)
        )
        self.p11_local_activity_radius = int(
            getattr(self.configs, 'p11_local_activity_radius', 50)
        )
        self._p11_activity_cache = {}
        self.p12_local_density_enabled = bool(
            getattr(self.configs, 'p12_local_density_enabled', False)
        )
        self.p12_spatial_cell_size = int(
            getattr(self.configs, 'p12_spatial_cell_size', 3)
        )
        self.p12_temporal_cell_size = int(
            getattr(self.configs, 'p12_temporal_cell_size', 50)
        )
        self.p12_neighborhood_radius = int(
            getattr(self.configs, 'p12_neighborhood_radius', 1)
        )
        self._p12_density_cache = {}
        self.temporal_chunk_enabled = bool(
            mode == 'train'
            and getattr(self.configs, 'temporal_chunk_enabled', False)
        )
        self.density_dual_view_enabled = bool(
            mode == 'train'
            and getattr(self.configs, 'density_dual_view_enabled', False)
        )
        self.density_dual_view_event_count_cutoff = int(
            getattr(
                self.configs,
                'density_dual_view_event_count_cutoff',
                self.configs.max_events_num,
            )
        )
        self.dense_target_oversampling_enabled = bool(
            mode == 'train'
            and getattr(self.configs, 'dense_target_oversampling_enabled', False)
        )
        self.dense_target_oversampling_event_count_cutoff = int(
            getattr(
                self.configs,
                'dense_target_oversampling_event_count_cutoff',
                self.configs.max_events_num,
            )
        )
        self.dense_target_oversampling_factor = int(
            getattr(self.configs, 'dense_target_oversampling_factor', 1)
        )
        self.dense_specialist_enabled = bool(
            mode == 'train'
            and getattr(self.configs, 'dense_specialist_enabled', False)
        )
        self.dense_specialist_event_count_cutoff = int(
            getattr(
                self.configs,
                'dense_specialist_event_count_cutoff',
                self.configs.max_events_num,
            )
        )
        self.dense_specialist_views_per_video = int(
            getattr(self.configs, 'dense_specialist_views_per_video', 1)
        )
        self.dense_specialist_target_preserving_enabled = bool(
            getattr(
                self.configs,
                'dense_specialist_target_preserving_enabled',
                False,
            )
        )
        self.horizontal_flip_augmentation_enabled = bool(
            mode == 'train'
            and getattr(self.configs, 'p15_horizontal_flip_augmentation_enabled', False)
        )
        self.horizontal_flip_augmentation_probability = float(
            getattr(self.configs, 'p15_horizontal_flip_augmentation_probability', 0.5)
        )
        if not 0.0 <= self.horizontal_flip_augmentation_probability <= 1.0:
            raise ValueError(
                'p15_horizontal_flip_augmentation_probability must be in [0, 1].'
            )
        if self.horizontal_flip_augmentation_enabled and getattr(
            self.configs,
            'p3_lite_enabled',
            False,
        ):
            raise ValueError(
                'p15_horizontal_flip_augmentation does not support P3-Lite event frames.'
            )
        self.density_dual_view_source_video_count = 0
        self.density_dual_view_extra_sample_count = 0
        self.dense_target_oversampling_source_video_count = 0
        self.dense_target_oversampling_extra_sample_count = 0
        self.dense_specialist_source_video_count = 0
        self.dense_specialist_sample_count = 0
        enabled_sampling_strategies = sum((
            self.temporal_chunk_enabled,
            self.density_dual_view_enabled,
            self.dense_target_oversampling_enabled,
            self.dense_specialist_enabled,
        ))
        if enabled_sampling_strategies > 1:
            raise ValueError(
                'temporal_chunk_enabled, density_dual_view_enabled, '
                'dense_target_oversampling_enabled, and '
                'dense_specialist_enabled are mutually exclusive training '
                'strategies.'
            )
        if self.density_dual_view_enabled:
            if not getattr(self.configs, 'target_preserving_enabled', False):
                raise ValueError(
                    'density_dual_view_enabled requires '
                    'target_preserving_enabled=true.'
                )
            if self.density_dual_view_event_count_cutoff < self.configs.max_events_num:
                raise ValueError(
                    'density_dual_view_event_count_cutoff must be at least '
                    'max_events_num.'
                )
        if self.dense_target_oversampling_enabled:
            if not getattr(self.configs, 'target_preserving_enabled', False):
                raise ValueError(
                    'dense_target_oversampling_enabled requires '
                    'target_preserving_enabled=true.'
                )
            if self.dense_target_oversampling_factor < 2:
                raise ValueError(
                    'dense_target_oversampling_factor must be at least 2 '
                    'when the strategy is enabled.'
                )
        if self.dense_specialist_enabled:
            if (
                self.dense_specialist_event_count_cutoff
                < self.configs.max_events_num
            ):
                raise ValueError(
                    'dense_specialist_event_count_cutoff must be at least '
                    'max_events_num.'
                )
            if self.dense_specialist_views_per_video < 1:
                raise ValueError(
                    'dense_specialist_views_per_video must be positive when '
                    'the dense specialist is enabled.'
                )
        self.sample_specs = None
        if self.temporal_chunk_enabled:
            self.sample_specs = self._build_temporal_chunk_specs()
        elif self.density_dual_view_enabled:
            self.sample_specs = self._build_density_dual_view_specs()
        elif self.dense_target_oversampling_enabled:
            self.sample_specs = self._build_dense_target_oversampling_specs()
        elif self.dense_specialist_enabled:
            self.sample_specs = self._build_dense_specialist_specs()

    def _get_p11_local_activity(self, file_name, locations):
        """Cache raw-stream activity so later epochs do not recompute it."""
        activity = self._p11_activity_cache.get(file_name)
        if activity is None:
            activity = build_local_activity_feature(
                locations,
                width=self.res[0],
                height=self.res[1],
                temporal_size=self.whole_t,
                temporal_radius=self.p11_local_activity_radius,
            )
            self._p11_activity_cache[file_name] = activity
        return activity

    def _get_p12_local_density(self, file_name, locations):
        """Cache complete-stream density before any training-only sampling."""
        density = self._p12_density_cache.get(file_name)
        if density is None:
            density = build_local_spatiotemporal_density_feature(
                locations,
                width=self.res[0],
                height=self.res[1],
                temporal_size=self.whole_t,
                spatial_cell_size=self.p12_spatial_cell_size,
                temporal_cell_size=self.p12_temporal_cell_size,
                neighborhood_radius=self.p12_neighborhood_radius,
            )
            self._p12_density_cache[file_name] = density
        return density

    def _build_temporal_chunk_specs(self):
        """Split only oversized training videos into event-time-contiguous chunks."""
        sample_specs = []
        for file_name in self.file_list:
            file_path = os.path.join(self.root, file_name)
            with np.load(file_path) as events:
                event_count = len(events['ev_loc'])
            for start, end in partition_event_indices(
                event_count,
                self.configs.max_events_num,
            ):
                sample_specs.append((file_name, start, end))
        return sample_specs

    def _build_density_dual_view_specs(self):
        """Add one uniform complement only for oversized training videos."""
        sample_specs = []
        dense_video_count = 0
        for file_name in self.file_list:
            file_path = os.path.join(self.root, file_name)
            with np.load(file_path) as events:
                event_count = len(events['ev_loc'])
            view_modes = density_dual_view_modes(
                event_count,
                self.density_dual_view_event_count_cutoff,
                density_dual_view_enabled=True,
            )
            sample_specs.extend((file_name, view_mode) for view_mode in view_modes)
            if len(view_modes) > 1:
                dense_video_count += 1

        self.density_dual_view_source_video_count = dense_video_count
        self.density_dual_view_extra_sample_count = dense_video_count
        return sample_specs

    def _build_dense_target_oversampling_specs(self):
        """Repeat only dense target-preserving source videos during training."""
        sample_specs = []
        dense_video_count = 0
        extra_sample_count = 0
        for file_name in self.file_list:
            file_path = os.path.join(self.root, file_name)
            with np.load(file_path) as events:
                event_count = len(events['ev_loc'])
            repeats = dense_target_oversample_repeats(
                event_count,
                self.dense_target_oversampling_event_count_cutoff,
                dense_target_oversampling_enabled=True,
                factor=self.dense_target_oversampling_factor,
            )
            sample_specs.extend((file_name, 'standard') for _ in range(repeats))
            if repeats > 1:
                dense_video_count += 1
                extra_sample_count += repeats - 1

        self.dense_target_oversampling_source_video_count = dense_video_count
        self.dense_target_oversampling_extra_sample_count = extra_sample_count
        return sample_specs

    def _build_dense_specialist_specs(self):
        """Use only configured views from oversized source videos.

        The source event-count cutoff is the same observable used later for
        dense-only expert inference.  Repeated views expose the specialist to
        independent random contexts without changing the frozen base model.
        """
        sample_specs = []
        dense_video_count = 0
        training_view = dense_specialist_training_view(
            self.dense_specialist_target_preserving_enabled
        )
        for file_name in self.file_list:
            file_path = os.path.join(self.root, file_name)
            with np.load(file_path) as events:
                event_count = len(events['ev_loc'])
            view_count = dense_specialist_view_count(
                event_count,
                self.dense_specialist_event_count_cutoff,
                dense_specialist_enabled=True,
                views_per_video=self.dense_specialist_views_per_video,
            )
            if view_count:
                dense_video_count += 1
                sample_specs.extend(
                    (file_name, training_view) for _ in range(view_count)
                )

        if not sample_specs:
            raise ValueError(
                'dense_specialist_enabled found no training videos above its '
                'event-count cutoff.'
            )
        self.dense_specialist_source_video_count = dense_video_count
        self.dense_specialist_sample_count = len(sample_specs)
        return sample_specs

    def __getitem__(self, num):
        if self.sample_specs is None:
            file_name = self.file_list[num]
            event_start = None
            event_end = None
            training_view = 'standard'
        elif self.temporal_chunk_enabled:
            file_name, event_start, event_end = self.sample_specs[num]
            training_view = 'standard'
        else:
            file_name, training_view = self.sample_specs[num]
            event_start = None
            event_end = None

        with np.load(os.path.join(self.root, file_name)) as events:
            evs_norm = events['evs_norm']
            ev_loc = events['ev_loc']
        local_activity = None
        if self.p11_local_activity_enabled:
            local_activity = self._get_p11_local_activity(file_name, ev_loc)
        local_density = None
        if self.p12_local_density_enabled:
            local_density = self._get_p12_local_density(file_name, ev_loc)
        if event_start is not None:
            evs_norm = evs_norm[event_start:event_end]
            ev_loc = ev_loc[event_start:event_end]
            if local_activity is not None:
                local_activity = local_activity[event_start:event_end]
            if local_density is not None:
                local_density = local_density[event_start:event_end]

        seg_label = evs_norm[:, 4]
        idx = evs_norm[:, 5]
        evs_norm = evs_norm[:, 0:4]
        if local_activity is not None:
            evs_norm = np.concatenate(
                (evs_norm, local_activity.reshape(-1, 1)),
                axis=1,
            )
        if local_density is not None:
            evs_norm = np.concatenate(
                (evs_norm, local_density.reshape(-1, 1)),
                axis=1,
            )
        event_frame = None
        if getattr(self.configs, 'p3_lite_enabled', False):
            event_frame = build_event_frame(
                ev_loc,
                evs_norm[:, 3],
                width=self.res[0],
                height=self.res[1],
                temporal_bins=getattr(self.configs, 'p3_lite_temporal_bins', 4),
                temporal_size=self.whole_t,
            )


        if self.mode=='train':
            num_events = ev_loc.shape[0]
            if num_events > self.configs.max_events_num:
                target_preserving_enabled = getattr(
                    self.configs,
                    'target_preserving_enabled',
                    False,
                )
                if training_view == 'target_preserving':
                    target_preserving_enabled = True
                elif training_view == 'uniform':
                    target_preserving_enabled = False
                dowmsample_idx = select_training_event_indices(
                    seg_label,
                    self.configs.max_events_num,
                    target_preserving_enabled=target_preserving_enabled,
                )
                ev_loc = ev_loc[dowmsample_idx]
                evs_norm=evs_norm[dowmsample_idx]
                seg_label = seg_label[dowmsample_idx]
                idx = idx[dowmsample_idx]

            if (
                self.horizontal_flip_augmentation_enabled
                and np.random.random() < self.horizontal_flip_augmentation_probability
            ):
                ev_loc, evs_norm = horizontal_flip_event_inputs(
                    ev_loc,
                    evs_norm,
                    image_width=self.res[0],
                    feature_width=padded_feature_width(self.res[0]),
                )


        out={}
        out['ev_loc']=ev_loc
        out['evs_norm']=evs_norm
        out['seg_label']=seg_label
        out['idx'] = idx
        if event_frame is not None:
            out['event_frame'] = event_frame

        return out


    def __len__(self):
        if self.sample_specs is not None:
            return len(self.sample_specs)
        return len(self.file_list)
