"""Diagnose horizontal-flip test-time augmentation on Challenge 2 validation.

The script mirrors only observable event coordinates and their normalized x
feature, restores scores to the original event order, and tests probability
averages before the existing P0/P6 evaluation path.  It never writes files or
changes the normal inference and submission paths.
"""

from dataclasses import dataclass

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.challenge_eval import add_batch_to_evaluator
from utils.density_threshold import (
    ChallengeCountTotals,
    DensityAdaptiveThresholdConfig,
    aggregate_challenge_counts,
)
from utils.ensemble import ChallengePredictor
from utils.eval import evalute
from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
    predict_full_event_scores,
    predict_random_chunk_scores,
)
from utils.postprocess import ChallengePostprocessor
from utils.spatial_tta import horizontal_flip_sample, padded_feature_width


@dataclass(frozen=True)
class CachedVideo:
    file_name: str
    event_count: int
    batch: dict
    original_scores: torch.Tensor
    flipped_scores: torch.Tensor


def parse_weights(values):
    """Read original-score weights from a diagnostic-only YAML list."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('HORIZONTAL_FLIP_TTA_SWEEP.original_weights must be a non-empty YAML list.')
    weights = tuple(sorted({float(value) for value in values}))
    if any(value < 0.0 or value > 1.0 for value in weights):
        raise ValueError('HORIZONTAL_FLIP_TTA_SWEEP.original_weights must be in [0, 1].')
    return weights


def predict_scores(predictor, dataset, sample, device, chunk_config):
    """Use the configured P8 path or one normal forward pass."""
    if chunk_config.should_partition(len(sample['ev_loc'])):
        return predict_random_chunk_scores(
            predictor,
            dataset,
            sample,
            device,
            chunk_config,
        )
    return predict_full_event_scores(predictor, dataset, sample, device), 0


def evaluate_counts(cached_video, scores, threshold):
    """Return exact Challenge 2 count contributions for one video."""
    evaluator = evalute(cfg)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
    predictions, _ = postprocessor.apply(scores, cached_video.batch['locs'])
    add_batch_to_evaluator(
        evaluator,
        cached_video.batch,
        predictions,
        sample_number=0,
        prediction_threshold=threshold,
        collect_roc=True,
    )
    labels = cached_video.batch['seg_label'].reshape(-1)
    binary_predictions = predictions.reshape(-1) >= threshold
    positive_mask = labels > 0.5
    return ChallengeCountTotals(
        true_positive_events=int((binary_predictions & positive_mask).sum().item()),
        false_positive_events=int((binary_predictions & ~positive_mask).sum().item()),
        positive_events=int(positive_mask.sum().item()),
        detected_target_frames=int(evaluator.correct_num),
        target_frames=int(evaluator.obj_num),
        false_components=int(evaluator.false_num),
        frame_count=int(evaluator.frame_num),
    )


def format_metrics(metrics):
    return 'Score={:.10f}, Pd={:.8f}, IoU={:.8f}, Acc={:.8f}, Fa={:.8e}'.format(
        metrics.score,
        metrics.pd,
        metrics.iou,
        metrics.acc,
        metrics.fa,
    )


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this sparse-convolution model.')
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval: True and TEST.roc: True in the config.')
    if cfg.batch_size != 1:
        raise ValueError('sweep_horizontal_flip_tta.py requires batch_size=1.')
    if getattr(cfg, 'p3_lite_enabled', False):
        raise ValueError('Horizontal flip TTA does not support P3-Lite event frames.')

    original_weights = parse_weights(getattr(cfg, 'horizontal_flip_original_weights'))
    chunk_config = InferenceChunkConfig.from_cfg(cfg)
    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    device = torch.device('cuda:0')
    predictor = ChallengePredictor(cfg, device, evspsegnet)
    dataset = EvUAV(cfg, mode='val')
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError('No validation files found in: {}'.format(dataset.root))

    image_width = int(cfg.res[0])
    feature_width = padded_feature_width(image_width)
    print('dict load:', predictor.primary_model_path)
    print('model ensemble:', predictor.describe())
    print('validation root:', dataset.root)
    print('validation videos:', len(dataset.file_list))
    print('P6 threshold policy:', threshold_policy.describe(float(cfg.prediction_threshold)))
    print('P8 random chunk inference:', chunk_config.describe())
    print('horizontal flip: image_width={}, feature_width={}'.format(
        image_width,
        feature_width,
    ))
    print('original-score weights:', ', '.join('{:.3f}'.format(value) for value in original_weights))
    print('postprocessor:', ChallengePostprocessor.from_cfg(
        cfg,
        float(cfg.prediction_threshold),
    ).describe())

    cached_videos = []
    p8_chunk_forwards = 0
    progress = tqdm.tqdm(total=len(dataset), desc='TTA inference', unit='video')
    for video_index, file_name in enumerate(dataset.file_list):
        sample = dataset[video_index]
        original_scores, original_chunk_count = predict_scores(
            predictor,
            dataset,
            sample,
            device,
            chunk_config,
        )
        flipped_sample = horizontal_flip_sample(
            sample,
            image_width=image_width,
            feature_width=feature_width,
        )
        flipped_scores, flipped_chunk_count = predict_scores(
            predictor,
            dataset,
            flipped_sample,
            device,
            chunk_config,
        )
        if original_scores.shape != flipped_scores.shape:
            raise RuntimeError('Original and flipped score shapes do not match.')
        cached_videos.append(CachedVideo(
            file_name=file_name,
            event_count=len(sample['ev_loc']),
            batch=evaluation_batch_from_sample(sample),
            original_scores=original_scores,
            flipped_scores=flipped_scores,
        ))
        p8_chunk_forwards += original_chunk_count + flipped_chunk_count
        progress.update(1)
    progress.close()
    if chunk_config.enabled:
        print('P8 chunk forwards across original and flip paths:', p8_chunk_forwards)

    results = []
    for original_weight in original_weights:
        counts = []
        for video in cached_videos:
            scores = (
                video.original_scores * original_weight
                + video.flipped_scores * (1.0 - original_weight)
            )
            threshold = threshold_policy.threshold_for_event_count(
                video.event_count,
                float(cfg.prediction_threshold),
            )
            counts.append(evaluate_counts(video, scores, threshold))
        metrics = aggregate_challenge_counts(counts)
        results.append((metrics, original_weight))
        print('original_weight={:.3f}: {}'.format(
            original_weight,
            format_metrics(metrics),
        ))

    results.sort(key=lambda item: item[0].score, reverse=True)
    best_metrics, best_original_weight = results[0]
    print('\nbest horizontal-flip TTA: original_weight={:.3f}, flipped_weight={:.3f}'.format(
        best_original_weight,
        1.0 - best_original_weight,
    ))
    print('best result:', format_metrics(best_metrics))
