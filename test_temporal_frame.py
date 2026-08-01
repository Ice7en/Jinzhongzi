"""Evaluate a temporal-frame checkpoint on Challenge 2 validation videos."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from configs.configs import cfg
from dataset.temporal_frame import load_temporal_frame_video
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.eval import evalute
from utils.postprocess import ChallengePostprocessor
from utils.temporal_frame_inference import (
    load_temporal_frame_model,
    predict_temporal_frame_scores,
)


def evaluation_batch(video):
    event_count = video.locations.shape[0]
    locations = torch.from_numpy(
        np.column_stack((
            np.zeros(event_count, dtype=np.int64),
            video.locations,
        ))
    ).long()
    return {
        'seg_label': torch.from_numpy(video.labels).float(),
        'locs': locations,
        'idx_label': video.target_ids.copy(),
    }


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for temporal-frame evaluation.')
    if not cfg.temporal_frame_model_path:
        raise ValueError(
            'Set TEMPORAL_FRAME.temporal_frame_model_path to a checkpoint.'
        )
    if bool(cfg.p6_density_threshold_enabled):
        raise ValueError(
            'test_temporal_frame.py evaluates its fixed threshold list. '
            'Disable P6 for this standalone calibration pass.'
        )

    device = torch.device('cuda:0')
    fine_detail_enabled = bool(cfg.temporal_frame_fine_detail_enabled)
    fine_detail_bin_ratio = 1
    if fine_detail_enabled:
        if (
            int(cfg.temporal_frame_fine_temporal_bin_size)
            > int(cfg.temporal_frame_bin_size)
            or int(cfg.temporal_frame_bin_size)
            % int(cfg.temporal_frame_fine_temporal_bin_size) != 0
        ):
            raise ValueError(
                'TEMPORAL_FRAME.fine_temporal_bin_size must be a positive '
                'divisor no greater than temporal_frame_bin_size.'
            )
        fine_detail_bin_ratio = (
            int(cfg.temporal_frame_bin_size)
            // int(cfg.temporal_frame_fine_temporal_bin_size)
        )
    model, checkpoint = load_temporal_frame_model(
        cfg.temporal_frame_model_path,
        device,
        cfg.temporal_frame_context_bins,
        cfg.temporal_frame_width,
        cfg.temporal_frame_local_contrast_enabled,
        cfg.temporal_frame_local_contrast_kernel_size,
        cfg.temporal_frame_motion_persistence_enabled,
        cfg.temporal_frame_motion_persistence_radius_per_bin,
        fine_detail_enabled,
        cfg.temporal_frame_fine_temporal_bin_size,
        cfg.temporal_frame_fine_context_bins,
        cfg.temporal_frame_target_center_enabled,
        getattr(cfg, 'temporal_frame_confidence_head_enabled', False),
        getattr(cfg, 'temporal_frame_density_calibration_enabled', False),
    )
    validation_root = Path(cfg.root) / 'val'
    file_paths = sorted(validation_root.glob('*.npz'))
    if not file_paths:
        raise RuntimeError('No validation npz files found in {}'.format(
            validation_root
        ))
    thresholds = tuple(float(value) for value in cfg.temporal_frame_eval_thresholds)
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError('TEMPORAL_FRAME.temporal_frame_eval_thresholds is invalid.')

    print('dict load:', cfg.temporal_frame_model_path)
    print(
        'temporal-frame model: context_bins={}, width={}, bin_size={}, '
        'fine_detail={}, fine_bin_size={}, fine_context_bins={}, '
        'target_center={}'.format(
            cfg.temporal_frame_context_bins,
            cfg.temporal_frame_width,
            cfg.temporal_frame_bin_size,
            fine_detail_enabled,
            cfg.temporal_frame_fine_temporal_bin_size,
            cfg.temporal_frame_fine_context_bins,
            cfg.temporal_frame_target_center_enabled,
        )
    )
    print('validation root:', validation_root)
    print('validation videos:', len(file_paths))
    print(
        'threshold candidates:',
        ', '.join('{:.3f}'.format(value) for value in thresholds),
    )
    print(
        'postprocessor:',
        ChallengePostprocessor.from_cfg(cfg, thresholds[0]).describe(),
    )

    samples = []
    pbar = tqdm.tqdm(file_paths, desc='video', unit='video')
    for path in pbar:
        video = load_temporal_frame_video(
            path,
            cfg.temporal_frame_bin_size,
            cfg.whole_t,
        )
        fine_detail_video = (
            load_temporal_frame_video(
                path,
                cfg.temporal_frame_fine_temporal_bin_size,
                cfg.whole_t,
            )
            if fine_detail_enabled else None
        )
        scores = predict_temporal_frame_scores(
            model,
            video,
            device,
            cfg.temporal_frame_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_frame_inference_batch_size,
            cfg.temporal_frame_log_count_clip,
            cfg.temporal_frame_local_contrast_enabled,
            cfg.temporal_frame_local_contrast_kernel_size,
            cfg.temporal_frame_motion_persistence_enabled,
            cfg.temporal_frame_motion_persistence_radius_per_bin,
            fine_detail_enabled,
            fine_detail_video,
            cfg.temporal_frame_fine_context_bins,
            fine_detail_bin_ratio,
        )
        samples.append((evaluation_batch(video), scores))
    pbar.close()

    results = []
    for threshold in thresholds:
        evaluator = evalute(cfg)
        postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
        postprocess_stats = postprocessor.new_stats()
        sample_number = 0
        for batch, scores in samples:
            predictions, stats = postprocessor.apply(scores, batch['locs'])
            postprocess_stats.merge(stats)
            sample_number = add_batch_to_evaluator(
                evaluator,
                batch,
                predictions,
                sample_number,
                threshold,
            )
        metrics = evaluate_challenge_metrics(evaluator, threshold)
        results.append((threshold, metrics, postprocess_stats))
        print(
            'threshold={:.3f}: Score={:.10f}, Pd={:.8f}, IoU={:.8f}, '
            'Acc={:.8f}, Fa={:.10e}'.format(
                threshold,
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
            )
        )

    print('\nTemporal-frame threshold sweep (sorted by Score)')
    print('threshold      Score         Pd          IoU         Acc          Fa')
    for threshold, metrics, _ in sorted(
        results,
        key=lambda item: item[1].score,
        reverse=True,
    ):
        print(
            '{:8.3f}  {:.10f}  {:.8f}  {:.8f}  {:.8f}  {:.8e}'.format(
                threshold,
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
            )
        )
    best_threshold, best_metrics, best_stats = max(
        results,
        key=lambda item: item[1].score,
    )
    print('\nbest threshold: {:.3f}'.format(best_threshold))
    print('best postprocess result:', best_stats.summary())
    print('best Score: {:.10f}'.format(best_metrics.score))
