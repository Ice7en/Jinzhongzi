"""Sweep density-aware decision thresholds on cached Challenge 2 predictions.

The script is diagnostic-only.  It evaluates thresholds selected from each
video's event count, an inference-time observable that does not use labels.
"""

from itertools import product

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.challenge_eval import add_batch_to_evaluator
from utils.density_threshold import (
    ChallengeCountTotals,
    aggregate_challenge_counts,
    select_density_threshold,
)
from utils.ensemble import ChallengePredictor
from utils.eval import evalute
from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
)
from utils.postprocess import ChallengePostprocessor
from utils.spatial_tta import HorizontalFlipTTAConfig
from utils.tta_inference import predict_sample_scores


def parse_candidates(values, name, cast):
    """Read a non-empty YAML list of typed sweep candidates."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('{} must be a non-empty YAML list.'.format(name))
    return tuple(sorted({cast(value) for value in values}))


def cache_validation_scores(predictor, dataset, device, chunk_config, tta_config):
    """Run the same sample-level inference path as test2.py once."""
    cached_batches = []
    p8_partitioned_videos = 0
    p8_chunk_count = 0
    progress = tqdm.tqdm(total=len(dataset), desc='model inference', unit='video')
    for video_index in range(len(dataset)):
        sample = dataset[video_index]
        event_count = len(sample['ev_loc'])
        batch = evaluation_batch_from_sample(sample)
        scores, chunk_count = predict_sample_scores(
            predictor,
            dataset,
            sample,
            device,
            chunk_config,
            tta_config,
        )
        if chunk_config.should_partition(event_count):
            p8_partitioned_videos += 1
            p8_chunk_count += chunk_count

        cached_batches.append({
            'file_name': dataset.file_list[video_index],
            'event_count': event_count,
            'seg_label': batch['seg_label'].detach().cpu().clone(),
            'locs': batch['locs'].detach().cpu().clone(),
            'idx_label': batch['idx_label'].copy(),
            'scores': scores,
        })
        progress.update(1)
    progress.close()
    return cached_batches, p8_partitioned_videos, p8_chunk_count


def evaluate_cached_video(cached_batch, threshold):
    """Return exact global-metric contributions for one thresholded video."""
    evaluator = evalute(cfg)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
    predictions, _ = postprocessor.apply(
        cached_batch['scores'],
        cached_batch['locs'],
    )
    add_batch_to_evaluator(
        evaluator,
        cached_batch,
        predictions,
        sample_number=0,
        prediction_threshold=threshold,
        collect_roc=True,
    )
    labels = cached_batch['seg_label'].reshape(-1)
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


def evaluate_rule(per_video_counts, cutoff, low_threshold, high_threshold):
    """Evaluate one binary density rule from cached per-video statistics."""
    selected_counts = []
    high_density_videos = 0
    for item in per_video_counts:
        threshold = select_density_threshold(
            item['event_count'],
            cutoff,
            low_threshold,
            high_threshold,
        )
        if threshold == high_threshold:
            high_density_videos += 1
        selected_counts.append(item['counts_by_threshold'][threshold])
    return aggregate_challenge_counts(selected_counts), high_density_videos


def thresholds_needed_for_video(
    event_count,
    cutoffs,
    low_thresholds,
    high_thresholds,
    baseline_threshold,
):
    """Return only thresholds this video can use across the candidate rules."""
    thresholds = {float(baseline_threshold)}
    for cutoff in cutoffs:
        if int(event_count) > cutoff:
            thresholds.update(high_thresholds)
        else:
            thresholds.update(low_thresholds)
    return tuple(sorted(thresholds))


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this sparse-convolution model.')
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval: True and TEST.roc: True in the config.')
    if cfg.batch_size != 1:
        raise ValueError('sweep_density_thresholds.py requires batch_size=1.')

    cutoffs = parse_candidates(
        getattr(cfg, 'event_count_cutoffs', []),
        'DENSITY_SWEEP.event_count_cutoffs',
        int,
    )
    low_thresholds = parse_candidates(
        getattr(cfg, 'low_thresholds', []),
        'DENSITY_SWEEP.low_thresholds',
        float,
    )
    high_thresholds = parse_candidates(
        getattr(cfg, 'high_thresholds', []),
        'DENSITY_SWEEP.high_thresholds',
        float,
    )
    if any(value <= 0.0 or value >= 1.0 for value in low_thresholds + high_thresholds):
        raise ValueError('All density-sweep thresholds must be in (0, 1).')

    baseline_threshold = float(cfg.prediction_threshold)
    device = torch.device('cuda:0')
    predictor = ChallengePredictor(cfg, device, evspsegnet)
    chunk_config = InferenceChunkConfig.from_cfg(cfg)
    tta_config = HorizontalFlipTTAConfig.from_cfg(cfg)
    if chunk_config.enabled and getattr(cfg, 'p3_lite_enabled', False):
        raise ValueError('P8 random chunk inference does not support P3-Lite event frames.')
    if tta_config.enabled and getattr(cfg, 'p3_lite_enabled', False):
        raise ValueError('P14 horizontal-flip TTA does not support P3-Lite event frames.')
    if predictor.dense_expert_config.enabled and cfg.batch_size != 1:
        raise ValueError('P20 dense-expert inference requires batch_size=1.')
    dataset = EvUAV(cfg, mode='val')
    dataset.file_list = sorted(dataset.file_list)

    print('dict load:', predictor.primary_model_path)
    print('model ensemble:', predictor.describe())
    print('P8 random chunk inference:', chunk_config.describe())
    print('P14 horizontal-flip TTA:', tta_config.describe())
    print('event-count cutoffs:', ', '.join(str(value) for value in cutoffs))
    print('low-density thresholds:', ', '.join('{:.3f}'.format(value) for value in low_thresholds))
    print('high-density thresholds:', ', '.join('{:.3f}'.format(value) for value in high_thresholds))
    print('baseline threshold:', '{:.3f}'.format(baseline_threshold))

    cached_batches, p8_partitioned_videos, p8_chunk_count = cache_validation_scores(
        predictor,
        dataset,
        device,
        chunk_config,
        tta_config,
    )
    if chunk_config.enabled:
        print(
            'P8 random chunk cache: {} high-density videos, {} chunk forwards'.format(
                p8_partitioned_videos,
                p8_chunk_count,
            )
        )
    per_video_counts = []
    thresholds_by_video = [
        thresholds_needed_for_video(
            cached_batch['event_count'],
            cutoffs,
            low_thresholds,
            high_thresholds,
            baseline_threshold,
        )
        for cached_batch in cached_batches
    ]
    progress = tqdm.tqdm(
        total=sum(len(thresholds) for thresholds in thresholds_by_video),
        desc='postprocess cache',
        unit='video-threshold',
    )
    for cached_batch, thresholds in zip(cached_batches, thresholds_by_video):
        counts_by_threshold = {}
        for threshold in thresholds:
            counts_by_threshold[threshold] = evaluate_cached_video(cached_batch, threshold)
            progress.update(1)
        per_video_counts.append({
            'file_name': cached_batch['file_name'],
            'event_count': cached_batch['event_count'],
            'counts_by_threshold': counts_by_threshold,
        })
    progress.close()

    baseline = aggregate_challenge_counts(
        item['counts_by_threshold'][baseline_threshold]
        for item in per_video_counts
    )
    print('\nstatic threshold={:.3f}: Score={:.10f}, Pd={:.8f}, IoU={:.8f}, Acc={:.8f}, Fa={:.8e}'.format(
        baseline_threshold,
        baseline.score,
        baseline.pd,
        baseline.iou,
        baseline.acc,
        baseline.fa,
    ))

    results = []
    for cutoff, low_threshold, high_threshold in product(cutoffs, low_thresholds, high_thresholds):
        metrics, high_density_videos = evaluate_rule(
            per_video_counts,
            cutoff,
            low_threshold,
            high_threshold,
        )
        results.append((metrics, cutoff, low_threshold, high_threshold, high_density_videos))
    results.sort(key=lambda item: item[0].score, reverse=True)

    print('\nDensity-aware threshold sweep (sorted by Score)')
    print('Score         Pd          IoU         Acc          Fa            cutoff   low    high   dense_videos')
    for metrics, cutoff, low_threshold, high_threshold, high_density_videos in results:
        print(
            '{:.10f}  {:.8f}  {:.8f}  {:.8f}  {:.8e}  {:>7}  {:.3f}  {:.3f}  {:>12}'.format(
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
                cutoff,
                low_threshold,
                high_threshold,
                high_density_videos,
            )
        )

    best_metrics, best_cutoff, best_low, best_high, best_video_count = results[0]
    print('\nbest density rule: event_count > {} -> {:.3f}, otherwise {:.3f}'.format(
        best_cutoff,
        best_high,
        best_low,
    ))
    print('best high-density videos:', best_video_count)
    print('Score improvement over static {:.3f}: {:+.10f}'.format(
        baseline_threshold,
        best_metrics.score - baseline.score,
    ))
