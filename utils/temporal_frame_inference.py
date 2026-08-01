"""Inference helpers for the full-stream temporal event-frame model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dataset.temporal_frame import (
    build_temporal_context_frame,
    temporal_frame_video_from_events,
)
from model.temporal_frame_net import (
    TemporalFrameNet,
    append_local_contrast_channels,
    build_motion_persistence_channels,
)
from utils.multiscale_motion import (
    build_multiscale_motion_persistence_channels,
    multiscale_motion_channel_count,
)


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'on'}:
            return True
        if normalized in {'false', '0', 'no', 'off'}:
            return False
        raise ValueError('Expected a boolean value, got {!r}.'.format(value))
    return bool(value)


@dataclass(frozen=True)
class TemporalFrameInferenceConfig:
    """Optional full-stream expert blended with sparse event scores."""

    enabled: bool = False
    model_path: str = ''
    sparse_weight: float = 0.5
    local_contrast_enabled: bool = False
    local_contrast_kernel_size: int = 9
    motion_persistence_enabled: bool = False
    motion_persistence_radius_per_bin: int = 4
    fine_detail_enabled: bool = False
    fine_temporal_bin_size: int = 25
    fine_context_bins: int = 9
    target_center_enabled: bool = False
    confidence_head_enabled: bool = False
    density_calibration_enabled: bool = False

    def __post_init__(self):
        if not 0.0 <= self.sparse_weight <= 1.0:
            raise ValueError('temporal-frame sparse_weight must be in [0, 1].')
        if (
            int(self.local_contrast_kernel_size) <= 0
            or int(self.local_contrast_kernel_size) % 2 == 0
        ):
            raise ValueError(
                'temporal-frame local_contrast_kernel_size must be a '
                'positive odd integer.'
            )
        if int(self.motion_persistence_radius_per_bin) < 0:
            raise ValueError(
                'motion_persistence_radius_per_bin must be non-negative.'
            )
        if self.fine_detail_enabled:
            if int(self.fine_temporal_bin_size) <= 0:
                raise ValueError(
                    'fine_temporal_bin_size must be positive.'
                )
            if (
                int(self.fine_context_bins) <= 0
                or int(self.fine_context_bins) % 2 == 0
            ):
                raise ValueError(
                    'fine_context_bins must be a positive odd integer.'
                )
        if self.enabled and not self.model_path:
            raise ValueError(
                'TEMPORAL_FRAME.temporal_frame_model_path is required when '
                'the temporal-frame expert is enabled.'
            )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, 'temporal_frame_enabled', False)),
            model_path=str(getattr(cfg, 'temporal_frame_model_path', '')),
            sparse_weight=float(
                getattr(cfg, 'temporal_frame_sparse_weight', 0.5)
            ),
            local_contrast_enabled=_as_bool(
                getattr(cfg, 'temporal_frame_local_contrast_enabled', False)
            ),
            local_contrast_kernel_size=int(
                getattr(cfg, 'temporal_frame_local_contrast_kernel_size', 9)
            ),
            motion_persistence_enabled=_as_bool(
                getattr(cfg, 'temporal_frame_motion_persistence_enabled', False)
            ),
            motion_persistence_radius_per_bin=int(
                getattr(
                    cfg,
                    'temporal_frame_motion_persistence_radius_per_bin',
                    4,
                )
            ),
            fine_detail_enabled=_as_bool(
                getattr(cfg, 'temporal_frame_fine_detail_enabled', False)
            ),
            fine_temporal_bin_size=int(
                getattr(cfg, 'temporal_frame_fine_temporal_bin_size', 25)
            ),
            fine_context_bins=int(
                getattr(cfg, 'temporal_frame_fine_context_bins', 9)
            ),
            target_center_enabled=_as_bool(
                getattr(cfg, 'temporal_frame_target_center_enabled', False)
            ),
            confidence_head_enabled=_as_bool(
                getattr(
                    cfg, 'temporal_frame_confidence_head_enabled', False,
                )
            ),
            density_calibration_enabled=_as_bool(
                getattr(
                    cfg, 'temporal_frame_density_calibration_enabled', False,
                )
            ),
        )

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return (
            'enabled (sparse_weight={:.3f}, temporal_frame_weight={:.3f}, '
            'local_contrast={}, local_contrast_kernel_size={}, '
            'motion_persistence={}, motion_radius_per_bin={}, '
            'fine_detail={}, fine_temporal_bin_size={}, fine_context_bins={}, '
            'target_center={}, '
            'model={})'
        ).format(
            self.sparse_weight,
            1.0 - self.sparse_weight,
            self.local_contrast_enabled,
            self.local_contrast_kernel_size,
            self.motion_persistence_enabled,
            self.motion_persistence_radius_per_bin,
            self.fine_detail_enabled,
            self.fine_temporal_bin_size,
            self.fine_context_bins,
            self.target_center_enabled,
            self.model_path,
        )

    @property
    def frame_only(self):
        """Whether inference can skip all sparse-model computation."""
        return self.enabled and self.sparse_weight == 0.0


def blend_temporal_frame_scores(sparse_scores, frame_scores, sparse_weight):
    """Blend aligned sparse and full-stream score vectors before thresholding."""
    if sparse_scores.shape != frame_scores.shape:
        raise ValueError(
            'Sparse and temporal-frame prediction shapes do not match: {} and {}.'
            .format(tuple(sparse_scores.shape), tuple(frame_scores.shape))
        )
    sparse_weight = float(sparse_weight)
    if not 0.0 <= sparse_weight <= 1.0:
        raise ValueError('sparse_weight must be in [0, 1].')
    return (
        sparse_scores * sparse_weight
        + frame_scores * (1.0 - sparse_weight)
    )


def temporal_frame_video_from_sample(sample, temporal_bin_size, whole_t):
    """Build label-free full-stream frame input from an EvUAV sample."""
    if 'ev_loc' not in sample or 'evs_norm' not in sample:
        raise KeyError('Temporal-frame inference requires ev_loc and evs_norm.')
    locations = np.asarray(sample['ev_loc'])
    event_features = np.asarray(sample['evs_norm'])
    if event_features.ndim != 2 or event_features.shape[1] < 4:
        raise ValueError('evs_norm must have at least four feature columns.')
    if locations.shape[0] != event_features.shape[0]:
        raise ValueError('ev_loc and evs_norm must have matching event counts.')
    return temporal_frame_video_from_events(
        name=str(sample.get('file_name', 'sample')),
        locations=locations,
        polarities=event_features[:, 3],
        temporal_bin_size=temporal_bin_size,
        whole_t=whole_t,
    )


def load_temporal_frame_model(
    checkpoint_path,
    device,
    context_bins,
    width,
    local_contrast_enabled=False,
    local_contrast_kernel_size=9,
    motion_persistence_enabled=False,
    motion_persistence_radius_per_bin=4,
    fine_detail_enabled=False,
    fine_temporal_bin_size=25,
    fine_context_bins=9,
    target_center_enabled=False,
    confidence_head_enabled=False,
    density_calibration_enabled=False,
):
    """Load a checkpoint and validate the architecture stored with it."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Temporal-frame checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    saved_config = checkpoint.get('temporal_frame', {})
    saved_context_bins = saved_config.get('context_bins')
    saved_width = saved_config.get('width')
    saved_local_contrast_enabled = _as_bool(
        saved_config.get('local_contrast_enabled', False)
    )
    saved_local_contrast_kernel_size = int(
        saved_config.get('local_contrast_kernel_size', 9)
    )
    saved_motion_persistence_enabled = _as_bool(
        saved_config.get('motion_persistence_enabled', False)
    )
    saved_motion_persistence_radius_per_bin = int(
        saved_config.get('motion_persistence_radius_per_bin', 4)
    )
    saved_fine_detail_enabled = _as_bool(
        saved_config.get('fine_detail_enabled', False)
    )
    saved_fine_temporal_bin_size = int(
        saved_config.get('fine_temporal_bin_size', 25)
    )
    saved_fine_context_bins = int(
        saved_config.get('fine_context_bins', 9)
    )
    saved_target_center_enabled = _as_bool(
        saved_config.get('target_center_enabled', False)
    )
    saved_confidence_head_enabled = _as_bool(
        saved_config.get('confidence_head_enabled', False)
    )
    saved_density_calibration_enabled = _as_bool(
        saved_config.get('density_calibration_enabled', False)
    )
    if saved_context_bins is not None and int(saved_context_bins) != int(context_bins):
        raise ValueError(
            'Checkpoint context_bins={} does not match configured {}.'.format(
                saved_context_bins,
                context_bins,
            )
        )
    if saved_width is not None and int(saved_width) != int(width):
        raise ValueError(
            'Checkpoint width={} does not match configured {}.'.format(
                saved_width,
                width,
            )
        )
    local_contrast_enabled = _as_bool(local_contrast_enabled)
    local_contrast_kernel_size = int(local_contrast_kernel_size)
    if (
        local_contrast_kernel_size <= 0
        or local_contrast_kernel_size % 2 == 0
    ):
        raise ValueError('local_contrast_kernel_size must be a positive odd integer.')
    if saved_local_contrast_enabled != local_contrast_enabled:
        raise ValueError(
            'Checkpoint local_contrast_enabled={} does not match configured {}.'
            .format(
                saved_local_contrast_enabled,
                local_contrast_enabled,
            )
        )
    if (
        local_contrast_enabled
        and saved_local_contrast_kernel_size != local_contrast_kernel_size
    ):
        raise ValueError(
            'Checkpoint local_contrast_kernel_size={} does not match configured {}.'
            .format(
                saved_local_contrast_kernel_size,
                local_contrast_kernel_size,
            )
        )
    motion_persistence_enabled = _as_bool(motion_persistence_enabled)
    motion_persistence_radius_per_bin = int(motion_persistence_radius_per_bin)
    if motion_persistence_radius_per_bin < 0:
        raise ValueError('motion_persistence_radius_per_bin must be non-negative.')
    if saved_motion_persistence_enabled != motion_persistence_enabled:
        raise ValueError(
            'Checkpoint motion_persistence_enabled={} does not match configured {}.'
            .format(
                saved_motion_persistence_enabled,
                motion_persistence_enabled,
            )
        )
    if (
        motion_persistence_enabled
        and saved_motion_persistence_radius_per_bin
        != motion_persistence_radius_per_bin
    ):
        raise ValueError(
            'Checkpoint motion_persistence_radius_per_bin={} does not match '
            'configured {}.'.format(
                saved_motion_persistence_radius_per_bin,
                motion_persistence_radius_per_bin,
            )
        )
    fine_detail_enabled = _as_bool(fine_detail_enabled)
    fine_temporal_bin_size = int(fine_temporal_bin_size)
    fine_context_bins = int(fine_context_bins)
    if fine_detail_enabled:
        if fine_temporal_bin_size <= 0:
            raise ValueError('fine_temporal_bin_size must be positive.')
        if fine_context_bins <= 0 or fine_context_bins % 2 == 0:
            raise ValueError(
                'fine_context_bins must be a positive odd integer.'
            )
    if saved_fine_detail_enabled != fine_detail_enabled:
        raise ValueError(
            'Checkpoint fine_detail_enabled={} does not match configured {}.'
            .format(saved_fine_detail_enabled, fine_detail_enabled)
        )
    if (
        fine_detail_enabled
        and saved_fine_temporal_bin_size != fine_temporal_bin_size
    ):
        raise ValueError(
            'Checkpoint fine_temporal_bin_size={} does not match configured {}.'
            .format(saved_fine_temporal_bin_size, fine_temporal_bin_size)
        )
    if fine_detail_enabled and saved_fine_context_bins != fine_context_bins:
        raise ValueError(
            'Checkpoint fine_context_bins={} does not match configured {}.'
            .format(saved_fine_context_bins, fine_context_bins)
        )
    target_center_enabled = _as_bool(target_center_enabled)
    if saved_target_center_enabled != target_center_enabled:
        raise ValueError(
            'Checkpoint target_center_enabled={} does not match configured {}.'
            .format(saved_target_center_enabled, target_center_enabled)
        )
    confidence_head_enabled = _as_bool(confidence_head_enabled)
    if saved_confidence_head_enabled != confidence_head_enabled:
        raise ValueError(
            'Checkpoint confidence_head_enabled={} does not match configured {}.'
            .format(saved_confidence_head_enabled, confidence_head_enabled)
        )
    density_calibration_enabled = _as_bool(density_calibration_enabled)
    if saved_density_calibration_enabled != density_calibration_enabled:
        raise ValueError(
            'Checkpoint density_calibration_enabled={} does not match '
            'configured {}.'.format(
                saved_density_calibration_enabled,
                density_calibration_enabled,
            )
        )
    model = TemporalFrameNet(
        input_channels=int(context_bins) * 2,
        width=int(width),
        local_contrast_channels=(
            int(context_bins) * 2 if local_contrast_enabled else 0
        ),
        motion_persistence_channels=(
            multiscale_motion_channel_count(context_bins)
            if motion_persistence_enabled else 0
        ),
        fine_detail_channels=(
            int(fine_context_bins) * 2 if fine_detail_enabled else 0
        ),
        target_center_enabled=target_center_enabled,
        confidence_head_enabled=confidence_head_enabled,
        density_calibration_enabled=density_calibration_enabled,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def predict_temporal_frame_scores(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
    local_contrast_enabled=False,
    local_contrast_kernel_size=9,
    motion_persistence_enabled=False,
    motion_persistence_radius_per_bin=4,
    fine_detail_enabled=False,
    fine_detail_video=None,
    fine_context_bins=9,
    fine_detail_bin_ratio=2,
):
    """Return one probability per source event from complete-frame inference.

    When fine detail is active, each 25-unit fine bin receives its own
    high-resolution context while retaining the P23 coarse frame associated
    with its enclosing 50-unit bin. Event indices remain in original order.
    """
    context_bins = int(context_bins)
    width = int(width)
    height = int(height)
    inference_batch_size = int(inference_batch_size)
    if inference_batch_size <= 0:
        raise ValueError('inference_batch_size must be positive.')
    fine_detail_enabled = _as_bool(fine_detail_enabled)
    fine_context_bins = int(fine_context_bins)
    fine_detail_bin_ratio = int(fine_detail_bin_ratio)
    if fine_detail_enabled:
        if fine_detail_video is None:
            raise ValueError(
                'fine_detail_video is required when fine detail is enabled.'
            )
        if fine_detail_video.locations.shape[0] != video.locations.shape[0]:
            raise ValueError(
                'Fine-detail and coarse videos must have matching event counts.'
            )
        if fine_context_bins <= 0 or fine_context_bins % 2 == 0:
            raise ValueError(
                'fine_context_bins must be a positive odd integer.'
            )
        if fine_detail_bin_ratio <= 0:
            raise ValueError('fine_detail_bin_ratio must be positive.')
    event_count = video.locations.shape[0]
    scores = np.empty(event_count, dtype=np.float32)
    pending_frames = []
    pending_fine_detail_frames = []
    pending_event_indices = []

    def flush():
        if not pending_frames:
            return
        raw_frame_tensor = torch.from_numpy(
            np.stack(pending_frames, axis=0)
        ).float().to(device)
        frame_tensor = raw_frame_tensor
        if local_contrast_enabled:
            frame_tensor = append_local_contrast_channels(
                raw_frame_tensor,
                local_contrast_kernel_size,
            )
        if motion_persistence_enabled:
            motion_channels = build_multiscale_motion_persistence_channels(
                raw_frame_tensor,
                context_bins,
            )
            frame_tensor = torch.cat((frame_tensor, motion_channels), dim=1)
        if fine_detail_enabled:
            fine_detail_tensor = torch.from_numpy(
                np.stack(pending_fine_detail_frames, axis=0)
            ).float().to(device)
            frame_tensor = torch.cat((frame_tensor, fine_detail_tensor), dim=1)
        with torch.no_grad():
            logits = model(frame_tensor).squeeze(1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        for batch_index, event_indices in enumerate(pending_event_indices):
            locations = video.locations[event_indices]
            scores[event_indices] = probabilities[
                batch_index,
                locations[:, 1],
                locations[:, 0],
            ]
        pending_frames.clear()
        pending_fine_detail_frames.clear()
        pending_event_indices.clear()

    event_indices_by_bin = (
        fine_detail_video.event_indices_by_bin
        if fine_detail_enabled else video.event_indices_by_bin
    )
    for center_bin, event_indices in enumerate(event_indices_by_bin):
        if event_indices.size == 0:
            continue
        coarse_center_bin = (
            center_bin // fine_detail_bin_ratio
            if fine_detail_enabled else center_bin
        )
        if coarse_center_bin >= len(video.event_indices_by_bin):
            raise ValueError(
                'Fine-detail temporal bins do not align with coarse bins.'
            )
        pending_frames.append(
            build_temporal_context_frame(
                video,
                coarse_center_bin,
                context_bins,
                width,
                height,
                log_count_clip,
            )
        )
        if fine_detail_enabled:
            pending_fine_detail_frames.append(
                build_temporal_context_frame(
                    fine_detail_video,
                    center_bin,
                    fine_context_bins,
                    width,
                    height,
                    log_count_clip,
                )
            )
        pending_event_indices.append(event_indices)
        if len(pending_frames) >= inference_batch_size:
            flush()
    flush()
    if not np.isfinite(scores).all():
        raise RuntimeError('Temporal-frame inference produced non-finite scores.')
    return torch.from_numpy(scores)
