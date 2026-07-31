"""Diagnose density-conditioned E1 ensemble weights on Challenge 2 validation.

The script runs both checkpoints once, caches their per-event CPU scores, then
evaluates low- and high-density ensemble weights with the same P6 threshold
policy and postprocessor used by ``test2.py``.  It never writes predictions,
changes checkpoints, or changes the normal inference path.
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
    DensityAdaptiveThresholdConfig,
    aggregate_challenge_counts,
)
from utils.ensemble import ChallengePredictor, weighted_average
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


def parse_weights(values, name):
    """Read a non-empty set of valid primary-model ensemble weights."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('{} must be a non-empty YAML list.'.format(name))

    weights = tuple(sorted({float(value) for value in values}))
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError('{} values must be in [0, 1].'.format(name))
    return weights


def evaluate_cached_video(cached_batch, scores, threshold):
    """Return exact Challenge 2 count contributions for one video."""
    evaluator = evalute(cfg)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
    predictions, _ = postprocessor.apply(scores, cached_batch['locs'])
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


def cache_validation_score_pairs(predictor, dataloader, device, file_names):
    """Infer both checkpoint scores once and retain evaluator inputs on CPU."""
    cached_batches = []
    progress = tqdm.tqdm(total=len(dataloader), desc='model inference', unit='video')
    for video_index, batch in enumerate(dataloader):
        with torch.no_grad():
            primary_scores, secondary_scores = predictor.predict_event_score_pair(
                batch['voxel_ev'],
                batch['p2v_map'].long().to(device),
                event_frame=batch.get('event_frame'),
            )
        if secondary_scores is None:
            raise ValueError(
                'Density ensemble sweep requires ENSEMBLE.ensemble_enabled=true.'
            )
        if primary_scores.shape != secondary_scores.shape:
            raise RuntimeError('Primary and secondary score shapes do not match.')

        cached_batches.append({
            'file_name': file_names[video_index],
            'event_count': int(primary_scores.numel()),
            'seg_label': batch['seg_label'].detach().cpu().clone(),
            'locs': batch['locs'].detach().cpu().clone(),
            'idx_label': batch['idx_label'].copy(),
            'primary_scores': primary_scores.detach().cpu().reshape(-1).clone(),
            'secondary_scores': secondary_scores.detach().cpu().reshape(-1).clone(),
        })
        progress.update(1)
    progress.close()
    return cached_batches


def density_group(event_count, event_count_cutoff):
    """Classify using only observable input event count."""
    return 'high' if int(event_count) > int(event_count_cutoff) else 'low'


def cache_group_counts(
    cached_batches,
    event_count_cutoff,
    low_weights,
    high_weights,
    threshold_policy,
    fallback_threshold,
):
    """Cache exact metric counts separately for each density group and weight."""
    group_counts = {
        'low': {weight: [] for weight in low_weights},
        'high': {weight: [] for weight in high_weights},
    }
    total_evaluations = sum(
        len(high_weights if density_group(item['event_count'], event_count_cutoff) == 'high' else low_weights)
        for item in cached_batches
    )
    progress = tqdm.tqdm(
        total=total_evaluations,
        desc='postprocess weight cache',
        unit='video-weight',
    )

    for cached_batch in cached_batches:
        group = density_group(cached_batch['event_count'], event_count_cutoff)
        threshold = threshold_policy.threshold_for_event_count(
            cached_batch['event_count'],
            fallback_threshold,
        )
        for primary_weight in group_counts[group]:
            scores = weighted_average(
                cached_batch['primary_scores'],
                cached_batch['secondary_scores'],
                primary_weight,
            )
            group_counts[group][primary_weight].append(
                evaluate_cached_video(cached_batch, scores, threshold)
            )
            progress.update(1)

    progress.close()
    if not group_counts['low'] or not group_counts['high']:
        raise RuntimeError('Both density groups must have at least one candidate weight.')
    if not next(iter(group_counts['low'].values())):
        raise RuntimeError('No low-density validation videos matched the cutoff.')
    if not next(iter(group_counts['high'].values())):
        raise RuntimeError('No high-density validation videos matched the cutoff.')
    return group_counts


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this sparse-convolution model.')
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval: True and TEST.roc: True in the config.')
    if cfg.batch_size != 1:
        raise ValueError('sweep_density_ensemble_weights.py requires batch_size=1.')

    device = torch.device('cuda:0')
    predictor = ChallengePredictor(cfg, device, evspsegnet)
    if not predictor.config.enabled:
        raise ValueError(
            'Set ENSEMBLE.ensemble_enabled=true and provide the secondary checkpoint.'
        )

    event_count_cutoff = int(getattr(cfg, 'ensemble_event_count_cutoff'))
    low_weights = parse_weights(
        getattr(cfg, 'ensemble_low_density_primary_weights'),
        'DENSITY_ENSEMBLE_SWEEP.low_density_primary_weights',
    )
    high_weights = parse_weights(
        getattr(cfg, 'ensemble_high_density_primary_weights'),
        'DENSITY_ENSEMBLE_SWEEP.high_density_primary_weights',
    )
    baseline_weight = float(predictor.config.primary_weight)
    low_weights = tuple(sorted(set(low_weights + (baseline_weight,))))
    high_weights = tuple(sorted(set(high_weights + (baseline_weight,))))

    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    dataset = EvUAV(cfg, mode='val')
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError('No validation files found in: {}'.format(dataset.root))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset.custom_collate,
        shuffle=False,
    )

    print('dict load:', predictor.primary_model_path)
    print('model ensemble:', predictor.describe())
    print('validation root:', dataset.root)
    print('validation videos:', len(dataset.file_list))
    print('P6 threshold policy:', threshold_policy.describe(float(cfg.prediction_threshold)))
    print('event-count cutoff:', event_count_cutoff)
    print('low-density primary-weight candidates:', ', '.join('{:.3f}'.format(value) for value in low_weights))
    print('high-density primary-weight candidates:', ', '.join('{:.3f}'.format(value) for value in high_weights))
    print('postprocessor:', ChallengePostprocessor.from_cfg(cfg, float(cfg.prediction_threshold)).describe())

    cached_batches = cache_validation_score_pairs(
        predictor,
        dataloader,
        device,
        dataset.file_list,
    )
    high_density_files = [
        item['file_name']
        for item in cached_batches
        if density_group(item['event_count'], event_count_cutoff) == 'high'
    ]
    print('high-density validation videos ({}): {}'.format(
        len(high_density_files),
        ', '.join(high_density_files),
    ))

    group_counts = cache_group_counts(
        cached_batches,
        event_count_cutoff,
        low_weights,
        high_weights,
        threshold_policy,
        float(cfg.prediction_threshold),
    )
    baseline = aggregate_challenge_counts(
        group_counts['low'][baseline_weight] + group_counts['high'][baseline_weight]
    )
    print('\nstatic ensemble primary_weight={:.3f}: Score={:.10f}, Pd={:.8f}, IoU={:.8f}, Acc={:.8f}, Fa={:.8e}'.format(
        baseline_weight,
        baseline.score,
        baseline.pd,
        baseline.iou,
        baseline.acc,
        baseline.fa,
    ))

    results = []
    for low_weight, high_weight in product(low_weights, high_weights):
        metrics = aggregate_challenge_counts(
            group_counts['low'][low_weight] + group_counts['high'][high_weight]
        )
        results.append((metrics, low_weight, high_weight))
    results.sort(key=lambda item: item[0].score, reverse=True)

    print('\nDensity-conditioned ensemble sweep (top 20 by Score)')
    print('Score         Pd          IoU         Acc          Fa            low_weight  high_weight')
    for metrics, low_weight, high_weight in results[:20]:
        print(
            '{:.10f}  {:.8f}  {:.8f}  {:.8f}  {:.8e}  {:>10.3f}  {:>11.3f}'.format(
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
                low_weight,
                high_weight,
            )
        )

    best_metrics, best_low_weight, best_high_weight = results[0]
    print('\nbest density-conditioned ensemble: event_count > {} -> primary_weight={:.3f}, otherwise {:.3f}'.format(
        event_count_cutoff,
        best_high_weight,
        best_low_weight,
    ))
    print('Score improvement over static ensemble: {:+.10f}'.format(
        best_metrics.score - baseline.score,
    ))
