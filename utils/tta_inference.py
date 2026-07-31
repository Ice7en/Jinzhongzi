"""Shared sample-level inference for optional label-free spatial TTA."""

from utils.inference_chunks import (
    predict_full_event_scores,
    predict_random_chunk_scores,
)
from utils.spatial_tta import (
    HorizontalFlipTTAConfig,
    horizontal_flip_sample,
    padded_feature_width,
)


def _predict_one_sample(predictor, dataset, sample, device, chunk_config):
    if chunk_config.should_partition(len(sample["ev_loc"])):
        return predict_random_chunk_scores(
            predictor,
            dataset,
            sample,
            device,
            chunk_config,
        )
    return predict_full_event_scores(predictor, dataset, sample, device), 0


def predict_sample_scores(
    predictor,
    dataset,
    sample,
    device,
    chunk_config,
    tta_config,
):
    """Predict one sample, optionally averaging its horizontal mirror."""
    if not isinstance(tta_config, HorizontalFlipTTAConfig):
        raise TypeError("tta_config must be a HorizontalFlipTTAConfig.")

    original_scores, original_chunk_count = _predict_one_sample(
        predictor,
        dataset,
        sample,
        device,
        chunk_config,
    )
    if not tta_config.enabled:
        return original_scores, original_chunk_count

    flipped_sample = horizontal_flip_sample(
        sample,
        image_width=int(dataset.res[0]),
        feature_width=padded_feature_width(int(dataset.res[0])),
    )
    flipped_scores, flipped_chunk_count = _predict_one_sample(
        predictor,
        dataset,
        flipped_sample,
        device,
        chunk_config,
    )
    if original_scores.shape != flipped_scores.shape:
        raise RuntimeError("Original and flipped prediction shapes do not match.")
    scores = (
        original_scores * tta_config.original_weight
        + flipped_scores * (1.0 - tta_config.original_weight)
    )
    return scores, original_chunk_count + flipped_chunk_count
