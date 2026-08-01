"""Check whether high-density videos benefit from inference-time partitioning.

The P1b models are trained with at most ``max_events_num`` events per video,
whereas validation normally feeds an entire event stream to the sparse model.
This diagnostic keeps every event and only changes which unlabelled events
share an inference input.  Scores are restored to their original event order
before the normal P0/P6 evaluation path runs.
"""

from dataclasses import dataclass

import numpy as np
import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from dataset.temporal_chunks import partition_event_indices
from model.evspsegnet import evspsegnet
from utils.challenge_eval import add_batch_to_evaluator
from utils.density_threshold import (
    ChallengeCountTotals,
    DensityAdaptiveThresholdConfig,
    aggregate_challenge_counts,
)
from utils.ensemble import ChallengePredictor
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


@dataclass(frozen=True)
class CachedVideo:
    """Validation labels and coordinates that do not require a sparse tensor."""

    file_name: str
    event_count: int
    batch: dict
    sample: dict


def parse_chunk_sizes(values):
    """Read positive chunk sizes from the diagnostic-only YAML list."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('INFERENCE_CHUNK_SWEEP.chunk_sizes must be a non-empty YAML list.')
    sizes = tuple(sorted({int(value) for value in values}))
    if any(value <= 0 for value in sizes):
        raise ValueError('INFERENCE_CHUNK_SWEEP.chunk_sizes must be positive.')
    return sizes


def parse_strategies(values):
    """Return supported partition strategies in stable order."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('INFERENCE_CHUNK_SWEEP.chunk_strategies must be a non-empty YAML list.')
    strategies = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    supported = {'temporal', 'random'}
    invalid = [value for value in strategies if value not in supported]
    if invalid:
        raise ValueError(
            'Unsupported INFERENCE_CHUNK_SWEEP.chunk_strategies: {}.'.format(
                ', '.join(invalid)
            )
        )
    return strategies


def partition_indices(event_count, chunk_size, strategy, random_seed):
    """Partition every event exactly once without consulting labels."""
    event_count = int(event_count)
    chunk_size = int(chunk_size)
    if event_count <= 0:
        raise ValueError('event_count must be positive.')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive.')

    if strategy == 'temporal':
        return tuple(
            np.arange(start, end, dtype=np.int64)
            for start, end in partition_event_indices(event_count, chunk_size)
        )
    if strategy == 'random':
        order = np.random.default_rng(int(random_seed)).permutation(event_count)
        return tuple(
            order[start:end]
            for start, end in partition_event_indices(event_count, chunk_size)
        )
    raise ValueError('Unknown partition strategy: {}.'.format(strategy))


def subset_sample(sample, indices):
    """Build one inference item while retaining the source event order mapping."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError('A non-empty one-dimensional event-index array is required.')
    if 'event_frame' in sample:
        raise ValueError('Inference chunk diagnostics do not support P3-Lite event frames.')
    return {
        'ev_loc': sample['ev_loc'][indices],
        'evs_norm': sample['evs_norm'][indices],
        'seg_label': sample['seg_label'][indices],
        'idx': sample['idx'][indices],
    }


def evaluation_batch(sample):
    """Build evaluator-only fields without materializing a full sparse tensor."""
    event_count = len(sample['ev_loc'])
    locations = np.column_stack((
        np.zeros(event_count, dtype=np.int64),
        sample['ev_loc'],
    ))
    return {
        'seg_label': torch.from_numpy(sample['seg_label']),
        'locs': torch.from_numpy(locations).to(torch.int64).contiguous(),
        'idx_label': sample['idx'].copy(),
    }


def predict_one_sample(predictor, dataset, sample, device):
    """Run a full single-sample sparse forward pass and return CPU event scores."""
    sparse_batch = dataset.custom_collate([sample])
    with torch.no_grad():
        scores = predictor.predict_event_scores(
            sparse_batch['voxel_ev'],
            sparse_batch['p2v_map'].long().to(device),
            event_frame=sparse_batch.get('event_frame'),
        ).detach().cpu().reshape(-1).clone()
    if scores.numel() != len(sample['ev_loc']):
        raise RuntimeError('Inference scores do not match the source event count.')
    del sparse_batch
    torch.cuda.empty_cache()
    return scores


def predict_partitioned_scores(
    predictor,
    dataset,
    sample,
    device,
    chunk_size,
    strategy,
    random_seed,
):
    """Predict every event once from an unlabelled input partition."""
    event_count = len(sample['ev_loc'])
    chunks = partition_indices(event_count, chunk_size, strategy, random_seed)
    scores = None
    covered = np.zeros(event_count, dtype=bool)

    for indices in chunks:
        if indices.max() >= event_count or indices.min() < 0:
            raise RuntimeError('A partition index is outside the source event range.')
        if covered[indices].any():
            raise RuntimeError('Inference partition overlaps source events.')
        chunk_scores = predict_one_sample(
            predictor,
            dataset,
            subset_sample(sample, indices),
            device,
        )
        if scores is None:
            scores = torch.empty(event_count, dtype=chunk_scores.dtype)
        scores[torch.from_numpy(indices)] = chunk_scores
        covered[indices] = True

    if scores is None or not covered.all():
        raise RuntimeError('Inference partition did not cover every source event.')
    return scores, len(chunks)


def evaluate_counts(cached_video, scores, threshold):
    """Return exact global-metric count contributions for one video."""
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
        raise ValueError('sweep_inference_chunks.py requires batch_size=1.')
    if getattr(cfg, 'p3_lite_enabled', False):
        raise ValueError('Disable P3-Lite before running inference chunk diagnostics.')

    event_count_cutoff = int(getattr(cfg, 'chunk_event_count_cutoff'))
    chunk_sizes = parse_chunk_sizes(getattr(cfg, 'chunk_sizes'))
    strategies = parse_strategies(getattr(cfg, 'chunk_strategies'))
    random_seed = int(getattr(cfg, 'chunk_random_seed'))
    if event_count_cutoff <= 0:
        raise ValueError('INFERENCE_CHUNK_SWEEP.chunk_event_count_cutoff must be positive.')

    device = torch.device('cuda:0')
    predictor = ChallengePredictor(cfg, device, evspsegnet)
    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    dataset = EvUAV(cfg, mode='val')
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError('No validation files found in: {}'.format(dataset.root))

    print('dict load:', predictor.primary_model_path)
    print('model ensemble:', predictor.describe())
    print('validation root:', dataset.root)
    print('validation videos:', len(dataset.file_list))
    print('P6 threshold policy:', threshold_policy.describe(float(cfg.prediction_threshold)))
    print('high-density chunk cutoff:', event_count_cutoff)
    print('chunk sizes:', ', '.join(str(value) for value in chunk_sizes))
    print('chunk strategies:', ', '.join(strategies))
    print('postprocessor:', ChallengePostprocessor.from_cfg(
        cfg, float(cfg.prediction_threshold)
    ).describe())

    cached_videos = []
    full_scores = {}
    progress = tqdm.tqdm(total=len(dataset.file_list), desc='full inference', unit='video')
    for video_index, file_name in enumerate(dataset.file_list):
        sample = dataset[video_index]
        cached_video = CachedVideo(
            file_name=file_name,
            event_count=len(sample['ev_loc']),
            batch=evaluation_batch(sample),
            sample=sample,
        )
        cached_videos.append(cached_video)
        full_scores[file_name] = predict_one_sample(predictor, dataset, sample, device)
        progress.update(1)
    progress.close()

    thresholds = {
        video.file_name: threshold_policy.threshold_for_event_count(
            video.event_count,
            float(cfg.prediction_threshold),
        )
        for video in cached_videos
    }
    baseline_counts = [
        evaluate_counts(video, full_scores[video.file_name], thresholds[video.file_name])
        for video in cached_videos
    ]
    baseline = aggregate_challenge_counts(baseline_counts)
    high_density_videos = [
        video for video in cached_videos
        if video.event_count > event_count_cutoff
    ]
    print('\nfull-input baseline:', format_metrics(baseline))
    print('high-density videos ({}): {}'.format(
        len(high_density_videos),
        ', '.join(
            '{} ({} events)'.format(video.file_name, video.event_count)
            for video in high_density_videos
        ),
    ))
    if not high_density_videos:
        raise RuntimeError('No validation videos exceed the configured chunk cutoff.')

    results = []
    for strategy in strategies:
        for chunk_size in chunk_sizes:
            replacement_scores = {}
            chunk_counts = {}
            for video in high_density_videos:
                scores, chunks = predict_partitioned_scores(
                    predictor,
                    dataset,
                    video.sample,
                    device,
                    chunk_size,
                    strategy,
                    random_seed,
                )
                replacement_scores[video.file_name] = scores
                chunk_counts[video.file_name] = chunks

            candidate_counts = []
            for video in cached_videos:
                scores = replacement_scores.get(
                    video.file_name,
                    full_scores[video.file_name],
                )
                candidate_counts.append(
                    evaluate_counts(video, scores, thresholds[video.file_name])
                )
            metrics = aggregate_challenge_counts(candidate_counts)
            results.append((metrics, strategy, chunk_size, replacement_scores, chunk_counts))
            print(
                '{} chunk_size={}: {} ({:+.10f})'.format(
                    strategy,
                    chunk_size,
                    format_metrics(metrics),
                    metrics.score - baseline.score,
                )
            )

    results.sort(key=lambda item: item[0].score, reverse=True)
    best_metrics, best_strategy, best_chunk_size, best_scores, best_chunk_counts = results[0]
    print('\nbest partitioned inference: strategy={}, chunk_size={}'.format(
        best_strategy,
        best_chunk_size,
    ))
    print('best global result:', format_metrics(best_metrics))
    print('global improvement over full input: {:+.10f}'.format(
        best_metrics.score - baseline.score,
    ))
    print('\nHigh-density per-video comparison')
    print('video       chunks  full Score/Pd/IoU/Acc/Fa                 partitioned Score/Pd/IoU/Acc/Fa')
    for video in high_density_videos:
        threshold = thresholds[video.file_name]
        full_metrics = aggregate_challenge_counts([
            evaluate_counts(video, full_scores[video.file_name], threshold)
        ])
        chunked_metrics = aggregate_challenge_counts([
            evaluate_counts(video, best_scores[video.file_name], threshold)
        ])
        print(
            '{:<10} {:>6}  {}  {}'.format(
                video.file_name.replace('.npz', ''),
                best_chunk_counts[video.file_name],
                format_metrics(full_metrics),
                format_metrics(chunked_metrics),
            )
        )
