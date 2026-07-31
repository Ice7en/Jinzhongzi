"""Inference helpers for bidirectional full-stream temporal memory."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dataset.temporal_frame import build_temporal_context_frame
from model.temporal_memory_net import BidirectionalTemporalMemoryNet


def _as_bool(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'true', '1', 'yes', 'on'}:
            return True
        if value in {'false', '0', 'no', 'off'}:
            return False
        raise ValueError('Expected a boolean value, got {!r}.'.format(value))
    return bool(value)


@dataclass(frozen=True)
class TemporalMemoryInferenceConfig:
    """Configuration for a full-stream bidirectional temporal-memory expert."""

    enabled: bool = False
    model_path: str = ''
    sparse_weight: float = 0.5

    def __post_init__(self):
        if self.enabled and not self.model_path:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_model_path is required when '
                'temporal memory is enabled.'
            )

    @property
    def memory_only(self):
        return self.enabled and self.sparse_weight == 0.0

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, 'temporal_memory_enabled', False)),
            model_path=str(getattr(cfg, 'temporal_memory_model_path', '')),
            sparse_weight=float(getattr(cfg, 'temporal_memory_sparse_weight', 0.5)),
        )

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return 'enabled (sparse_weight={:.3f}, memory_weight={:.3f}, model={})'.format(
            self.sparse_weight,
            1.0 - self.sparse_weight,
            self.model_path,
        )


def load_temporal_memory_model(
    checkpoint_path,
    device,
    context_bins,
    width,
    sequence_length,
):
    """Load a temporal-memory checkpoint and validate its saved architecture."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Temporal-memory checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved = checkpoint.get('temporal_memory', {})
    saved_context_bins = saved.get('context_bins')
    saved_width = saved.get('width')
    saved_sequence_length = saved.get('sequence_length')
    saved_density_calibration = bool(
        saved.get('density_calibration_enabled', False)
    )
    if saved_context_bins is not None and int(saved_context_bins) != int(context_bins):
        raise ValueError(
            'Checkpoint context_bins={} does not match configured {}.'.format(
                saved_context_bins, context_bins
            )
        )
    if saved_width is not None and int(saved_width) != int(width):
        raise ValueError(
            'Checkpoint width={} does not match configured {}.'.format(
                saved_width, width
            )
        )
    if (
        saved_sequence_length is not None
        and int(saved_sequence_length) != int(sequence_length)
    ):
        raise ValueError(
            'Checkpoint sequence_length={} does not match configured {}.'.format(
                saved_sequence_length, sequence_length
            )
        )
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(context_bins) * 2,
        width=int(width),
        density_calibration_enabled=saved_density_calibration,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def _frame_tensor(video, temporal_bins, context_bins, width, height, log_count_clip, device):
    frames = np.stack(
        [
            build_temporal_context_frame(
                video,
                temporal_bin,
                context_bins,
                width,
                height,
                log_count_clip,
            )
            for temporal_bin in temporal_bins
        ],
        axis=0,
    )
    return torch.from_numpy(frames).float().to(device)


def predict_temporal_memory_scores(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
):
    """Return one probability per event using bidirectional full-stream memory.

    Bottleneck maps are kept for all temporal bins, while skip features are
    recomputed in a second pass. This keeps inference within the 4GB budget.
    """
    context_bins = int(context_bins)
    width = int(width)
    height = int(height)
    inference_batch_size = int(inference_batch_size)
    if context_bins < 1 or context_bins % 2 == 0:
        raise ValueError('context_bins must be a positive odd integer.')
    if inference_batch_size <= 0:
        raise ValueError('inference_batch_size must be positive.')
    temporal_bin_count = len(video.event_indices_by_bin)
    if temporal_bin_count <= 0:
        raise ValueError('video must contain temporal bins.')

    bottlenecks = []
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            temporal_bins = list(
                range(start, min(start + inference_batch_size, temporal_bin_count))
            )
            frames = _frame_tensor(
                video,
                temporal_bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
            )
            bottlenecks.append(model.encode_bottleneck(frames))
        residuals = model.temporal_residual(torch.cat(bottlenecks, dim=0))

    scores = np.empty(video.locations.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            temporal_bins = list(
                range(start, min(start + inference_batch_size, temporal_bin_count))
            )
            frames = _frame_tensor(
                video,
                temporal_bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
            )
            probabilities = torch.sigmoid(
                model.decode_with_residual(frames, residuals[start:start + len(temporal_bins)])
            ).squeeze(1).cpu().numpy()
            for local_index, temporal_bin in enumerate(temporal_bins):
                event_indices = video.event_indices_by_bin[temporal_bin]
                if event_indices.size == 0:
                    continue
                locations = video.locations[event_indices]
                scores[event_indices] = probabilities[
                    local_index,
                    locations[:, 1],
                    locations[:, 0],
                ]
    if not np.isfinite(scores).all():
        raise RuntimeError('Temporal-memory inference produced non-finite scores.')
    return torch.from_numpy(scores)
