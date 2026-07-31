"""Memory-bounded random-partition inference for dense event videos.

The 4 GB training configuration samples at most ``max_events_num`` events
from an oversized video.  This module optionally applies the same observable,
label-free input budget at inference time, while restoring one score for every
original event before thresholding and postprocessing.
"""

from dataclasses import dataclass

import numpy as np


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError("Expected a boolean value, got {!r}.".format(value))
    return bool(value)


def _parse_random_seeds(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("p8_random_seeds must be a non-empty YAML list.")
    seeds = tuple(int(value) for value in values)
    if len(set(seeds)) != len(seeds):
        raise ValueError("p8_random_seeds must not contain duplicates.")
    return seeds


@dataclass(frozen=True)
class InferenceChunkConfig:
    """Optional random-partition inference configuration."""

    enabled: bool = False
    event_count_cutoff: int = 100000
    chunk_size: int = 100000
    random_seeds: tuple = (37,)

    def __post_init__(self):
        if self.event_count_cutoff <= 0:
            raise ValueError("p8_event_count_cutoff must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("p8_chunk_size must be positive.")
        object.__setattr__(self, "random_seeds", _parse_random_seeds(self.random_seeds))

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, "p8_enabled", False)),
            event_count_cutoff=int(getattr(cfg, "p8_event_count_cutoff", 100000)),
            chunk_size=int(getattr(cfg, "p8_chunk_size", 100000)),
            random_seeds=getattr(cfg, "p8_random_seeds", [37]),
        )

    def should_partition(self, event_count):
        return self.enabled and int(event_count) > self.event_count_cutoff

    def describe(self):
        if not self.enabled:
            return "disabled"
        return "enabled (event_count > {}, chunk_size={}, random_seeds={})".format(
            self.event_count_cutoff,
            self.chunk_size,
            ",".join(str(seed) for seed in self.random_seeds),
        )


def random_partition_indices(event_count, chunk_size, random_seed):
    """Return a deterministic, label-free partition covering every event once."""
    event_count = int(event_count)
    chunk_size = int(chunk_size)
    if event_count <= 0:
        raise ValueError("event_count must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    order = np.random.default_rng(int(random_seed)).permutation(event_count)
    return tuple(
        order[start:min(start + chunk_size, event_count)]
        for start in range(0, event_count, chunk_size)
    )


def subset_inference_sample(sample, indices):
    """Subset only event inputs while preserving a source-order mapping."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("A non-empty one-dimensional event-index array is required.")
    if "event_frame" in sample:
        raise ValueError("P8 random chunk inference does not support P3-Lite event frames.")
    return {
        "ev_loc": sample["ev_loc"][indices],
        "evs_norm": sample["evs_norm"][indices],
        # custom_collate needs these fields, but partitioning and prediction
        # never inspect their values.
        "seg_label": sample["seg_label"][indices],
        "idx": sample["idx"][indices],
    }


def evaluation_batch_from_sample(sample):
    """Construct evaluator fields without materializing a full sparse tensor."""
    import torch

    event_count = len(sample["ev_loc"])
    locations = np.column_stack((
        np.zeros(event_count, dtype=np.int64),
        sample["ev_loc"],
    ))
    return {
        "seg_label": torch.from_numpy(sample["seg_label"]),
        "locs": torch.from_numpy(locations).to(torch.int64).contiguous(),
        "idx_label": sample["idx"].copy(),
    }


def predict_full_event_scores(
    predictor,
    dataset,
    sample,
    device,
    source_event_count=None,
):
    """Run one ordinary single-video forward pass and return CPU scores."""
    import torch

    sparse_batch = dataset.custom_collate([sample])
    with torch.no_grad():
        scores = predictor.predict_event_scores(
            sparse_batch["voxel_ev"],
            sparse_batch["p2v_map"].long().to(device),
            event_frame=sparse_batch.get("event_frame"),
            source_event_count=(
                len(sample["ev_loc"])
                if source_event_count is None else int(source_event_count)
            ),
        ).detach().cpu().reshape(-1).clone()
    if scores.numel() != len(sample["ev_loc"]):
        raise RuntimeError("Inference scores do not match the source event count.")
    del sparse_batch
    torch.cuda.empty_cache()
    return scores


def _predict_one_random_partition(predictor, dataset, sample, device, chunk_size, random_seed):
    """Infer each event once from one deterministic random partition."""
    import torch

    event_count = len(sample["ev_loc"])
    chunks = random_partition_indices(event_count, chunk_size, random_seed)
    scores = None
    covered = np.zeros(event_count, dtype=bool)

    for indices in chunks:
        if indices.min() < 0 or indices.max() >= event_count:
            raise RuntimeError("A partition index is outside the source event range.")
        if covered[indices].any():
            raise RuntimeError("Inference partition overlaps source events.")
        chunk_scores = predict_full_event_scores(
            predictor,
            dataset,
            subset_inference_sample(sample, indices),
            device,
            source_event_count=event_count,
        )
        if scores is None:
            scores = torch.empty(event_count, dtype=chunk_scores.dtype)
        scores[torch.from_numpy(indices)] = chunk_scores
        covered[indices] = True

    if scores is None or not covered.all():
        raise RuntimeError("Inference partition did not cover every source event.")
    return scores, len(chunks)


def predict_random_chunk_scores(predictor, dataset, sample, device, config):
    """Average one or more random partitions without reading labels for decisions."""
    if not isinstance(config, InferenceChunkConfig):
        raise TypeError("config must be an InferenceChunkConfig.")
    if not config.should_partition(len(sample["ev_loc"])):
        raise ValueError("P8 was requested for a video that does not exceed its cutoff.")

    averaged_scores = None
    chunk_count = 0
    for random_seed in config.random_seeds:
        scores, current_chunk_count = _predict_one_random_partition(
            predictor,
            dataset,
            sample,
            device,
            config.chunk_size,
            random_seed,
        )
        if averaged_scores is None:
            averaged_scores = scores
        else:
            averaged_scores += scores
        chunk_count += current_chunk_count

    return averaged_scores / len(config.random_seeds), chunk_count
