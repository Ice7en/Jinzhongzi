"""Report Challenge 2 metrics for each validation video.

This is a diagnostic-only entry point.  It uses the same model ensemble,
decision threshold, and postprocessor as ``test2.py`` but does not write
submission files or change predictions.
"""

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.density_threshold import DensityAdaptiveThresholdConfig
from utils.ensemble import ChallengePredictor
from utils.eval import evalute
from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
)
from utils.postprocess import ChallengePostprocessor
from utils.spatial_tta import HorizontalFlipTTAConfig
from utils.tta_inference import predict_sample_scores


def evaluate_video(batch, scores, postprocessor, prediction_threshold):
    """Evaluate one validation video using the shared Challenge 2 path."""
    evaluator = evalute(cfg)
    predictions, postprocess_stats = postprocessor.apply(scores, batch['locs'])
    add_batch_to_evaluator(
        evaluator,
        batch,
        predictions,
        sample_number=0,
        prediction_threshold=prediction_threshold,
        collect_roc=True,
    )
    return evaluate_challenge_metrics(evaluator, prediction_threshold), evaluator, postprocess_stats


def print_table(results, sort_key, heading):
    print('\n{}'.format(heading))
    print('video       events   targets  hits     Pd       IoU      Acc      pos(in/out)       false   Fa')
    for result in sorted(results, key=sort_key):
        stats = result['postprocess_stats']
        print(
            '{:<10} {:>8} {:>7} {:>5}  {:.6f}  {:.6f}  {:.6f}  {:>7}/{:<7} {:>7}  {:.3e}'.format(
                result['file_name'].replace('.npz', ''),
                result['event_count'],
                result['targets'],
                result['hits'],
                result['metrics'].pd,
                result['metrics'].iou,
                result['metrics'].acc,
                stats.input_positive_events,
                stats.output_positive_events,
                result['false_components'],
                result['metrics'].fa,
            )
        )


def configured_thresholds(prediction_threshold):
    """Return diagnostic candidates while always including the active threshold."""
    values = getattr(cfg, 'thresholds', [])
    if not isinstance(values, (list, tuple)):
        raise ValueError('SWEEP.thresholds must be a YAML list.')
    thresholds = {float(value) for value in values}
    thresholds.add(float(prediction_threshold))
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError('All diagnostic thresholds must be in (0, 1).')
    return sorted(thresholds)


def print_threshold_table(results):
    """Show per-video threshold potential without changing the global pipeline."""
    print('\nPer-video threshold diagnostic (validation-only; not a submission rule)')
    print('video       base_t  base_score  best_t  best_score  delta     Pd(base->best)  Fa(base->best)')
    rows = []
    for result in results:
        threshold_results = result['threshold_results']
        base_threshold = result['base_threshold']
        base_metrics = threshold_results[base_threshold]
        best_threshold, best_metrics = max(
            threshold_results.items(),
            key=lambda item: item[1].score,
        )
        rows.append((
            best_metrics.score - base_metrics.score,
            result['file_name'],
            base_threshold,
            base_metrics,
            best_threshold,
            best_metrics,
        ))

    for delta, file_name, base_threshold, base_metrics, best_threshold, best_metrics in sorted(
        rows,
        key=lambda item: (-item[0], item[1]),
    ):
        print(
            '{:<10} {:.3f}   {:.6f}    {:.3f}   {:.6f}   {:+.6f}  {:.3f}->{:.3f}    {:.2e}->{:.2e}'.format(
                file_name.replace('.npz', ''),
                base_threshold,
                base_metrics.score,
                best_threshold,
                best_metrics.score,
                delta,
                base_metrics.pd,
                best_metrics.pd,
                base_metrics.fa,
                best_metrics.fa,
            )
        )


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this sparse-convolution model.')
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval: True and TEST.roc: True in the config.')
    if cfg.batch_size != 1:
        raise ValueError('analyze_validation.py requires TEST/TRAIN batch_size=1.')

    device = torch.device('cuda:0')
    prediction_threshold = float(cfg.prediction_threshold)
    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    chunk_config = InferenceChunkConfig.from_cfg(cfg)
    tta_config = HorizontalFlipTTAConfig.from_cfg(cfg)
    if chunk_config.enabled and getattr(cfg, 'p3_lite_enabled', False):
        raise ValueError('P8 random chunk inference does not support P3-Lite event frames.')
    if tta_config.enabled and getattr(cfg, 'p3_lite_enabled', False):
        raise ValueError('P14 horizontal-flip TTA does not support P3-Lite event frames.')
    threshold_candidates = configured_thresholds(prediction_threshold)
    predictor = ChallengePredictor(cfg, device, evspsegnet)
    dataset = EvUAV(cfg, mode='val')
    dataset.file_list = sorted(dataset.file_list)
    dataloader = None
    if not chunk_config.enabled:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.custom_collate,
            shuffle=False,
        )

    print('dict load:', predictor.primary_model_path)
    print('model ensemble:', predictor.describe())
    print('prediction threshold:', prediction_threshold)
    print('threshold policy:', threshold_policy.describe(prediction_threshold))
    print('P8 random chunk inference:', chunk_config.describe())
    print('P14 horizontal-flip TTA:', tta_config.describe())
    print('diagnostic thresholds:', ', '.join('{:.3f}'.format(value) for value in threshold_candidates))
    print('postprocessor:', ChallengePostprocessor.from_cfg(
        cfg,
        prediction_threshold,
    ).describe())

    results = []
    p8_partitioned_videos = 0
    p8_chunk_count = 0
    progress = tqdm.tqdm(total=len(dataset), desc='video', unit='video')
    for video_index, file_name in enumerate(dataset.file_list):
        sample = dataset[video_index]
        event_count = len(sample['ev_loc'])
        if chunk_config.enabled or tta_config.enabled:
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
        else:
            sparse_batch = dataset.custom_collate([sample])
            with torch.no_grad():
                scores = predictor.predict_event_scores(
                    sparse_batch['voxel_ev'],
                    sparse_batch['p2v_map'].long().to(device),
                    event_frame=sparse_batch.get('event_frame'),
                )
            batch = {
                'seg_label': sparse_batch['seg_label'].detach().cpu().clone(),
                'locs': sparse_batch['locs'].detach().cpu().clone(),
                'idx_label': sparse_batch['idx_label'].copy(),
            }
            del sparse_batch

        base_threshold = threshold_policy.threshold_for_event_count(
            event_count,
            prediction_threshold,
        )
        video_threshold_candidates = sorted(
            set(threshold_candidates + [base_threshold])
        )
        threshold_results = {}
        default_evaluator = None
        default_postprocess_stats = None
        for threshold in video_threshold_candidates:
            metrics, evaluator, postprocess_stats = evaluate_video(
                batch,
                scores,
                ChallengePostprocessor.from_cfg(cfg, threshold),
                threshold,
            )
            threshold_results[threshold] = metrics
            if threshold == base_threshold:
                default_evaluator = evaluator
                default_postprocess_stats = postprocess_stats

        if default_evaluator is None or default_postprocess_stats is None:
            raise RuntimeError('The active prediction threshold was not evaluated.')
        results.append({
            'file_name': file_name,
            'metrics': threshold_results[base_threshold],
            'event_count': event_count,
            'base_threshold': base_threshold,
            'targets': default_evaluator.obj_num,
            'hits': default_evaluator.correct_num,
            'false_components': default_evaluator.false_num,
            'postprocess_stats': default_postprocess_stats,
            'threshold_results': threshold_results,
        })
        progress.update(1)
    progress.close()
    if chunk_config.enabled:
        print(
            'P8 random chunk result: {} high-density videos, {} chunk forwards'.format(
                p8_partitioned_videos,
                p8_chunk_count,
            )
        )

    print_table(results, lambda item: (item['metrics'].pd, item['file_name']), 'Ranked by lowest Pd')
    print_table(
        results,
        lambda item: (-item['false_components'], item['file_name']),
        'Ranked by false components',
    )
    print_threshold_table(results)
