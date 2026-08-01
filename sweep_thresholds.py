"""Evaluate one Challenge 2 predictor across configured decision thresholds."""

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.density_threshold import DensityAdaptiveThresholdConfig
from utils.ensemble import ChallengePredictor
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor


def parse_thresholds(values):
    """Return sorted unique thresholds from the SWEEP configuration list."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('SWEEP.thresholds must be a non-empty YAML list.')

    thresholds = sorted({float(value) for value in values})
    if any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise ValueError('Every SWEEP.thresholds value must be in (0, 1).')
    return thresholds


def cache_validation_scores(predictor, dataloader, device):
    """Run the model once and retain only CPU data needed by the evaluator."""
    cached_batches = []
    pbar = tqdm.tqdm(
        total=len(dataloader),
        desc='model inference',
        unit='video',
        unit_scale=True,
        position=0,
        leave=True,
    )
    for batch in dataloader:
        with torch.no_grad():
            scores = predictor.predict_event_scores(
                batch['voxel_ev'],
                batch['p2v_map'].long().to(device),
                event_frame=batch.get('event_frame'),
            ).detach().cpu().reshape(-1).clone()

        cached_batches.append({
            'seg_label': batch['seg_label'].detach().cpu().clone(),
            'locs': batch['locs'].detach().cpu().clone(),
            'idx_label': batch['idx_label'].copy(),
            'scores': scores,
        })
        pbar.update(1)
    pbar.close()
    return cached_batches


def evaluate_threshold(cached_batches, threshold):
    """Apply the configured postprocessor and score one decision threshold."""
    evaluator = evalute(cfg)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
    postprocess_stats = postprocessor.new_stats()
    sample_number = 0

    for cached_batch in cached_batches:
        predictions, batch_stats = postprocessor.apply(
            cached_batch['scores'],
            cached_batch['locs'],
        )
        postprocess_stats.merge(batch_stats)
        sample_number = add_batch_to_evaluator(
            evaluator,
            cached_batch,
            predictions,
            sample_number,
            threshold,
            collect_roc=True,
        )

    return evaluate_challenge_metrics(evaluator, threshold), postprocess_stats


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this sparse-convolution model.')
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval: True and TEST.roc: True in the config.')
    if DensityAdaptiveThresholdConfig.from_cfg(cfg).enabled:
        raise ValueError(
            'P6 density-adaptive threshold is enabled. Use '
            'sweep_density_thresholds.py instead of a global threshold sweep.'
        )

    thresholds = parse_thresholds(getattr(cfg, 'thresholds', []))
    device = torch.device('cuda:0')
    predictor = ChallengePredictor(cfg, device, evspsegnet)
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
    print('threshold candidates:', ', '.join('{:.3f}'.format(value) for value in thresholds))
    print('postprocessor:', ChallengePostprocessor.from_cfg(cfg, thresholds[0]).describe())

    cached_batches = cache_validation_scores(predictor, dataloader, device)
    print('cached validation predictions: {} videos'.format(len(cached_batches)))

    results = []
    for threshold in thresholds:
        metrics, postprocess_stats = evaluate_threshold(cached_batches, threshold)
        results.append((threshold, metrics, postprocess_stats))
        print(
            'threshold={:.3f}: Score={:.10f}, Pd={:.10f}, IoU={:.10f}, '
            'Acc={:.10f}, Fa={:.10e}'.format(
                threshold,
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
            )
        )

    results.sort(key=lambda item: item[1].score, reverse=True)
    best_threshold, best_metrics, best_postprocess_stats = results[0]
    print('\nChallenge 2 threshold sweep (sorted by Score)')
    print('threshold      Score         Pd          IoU         Acc          Fa')
    for threshold, metrics, _ in results:
        print(
            '{:>8.3f}  {:>12.10f}  {:>10.8f}  {:>10.8f}  {:>10.8f}  {:.8e}'.format(
                threshold,
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
            )
        )
    print('\nbest threshold: {:.3f}'.format(best_threshold))
    print('best postprocess result:', best_postprocess_stats.summary())
    print(
        'use the same value for evaluation and submission: '
        'TEST.prediction_threshold={:.3f}'.format(best_threshold)
    )
