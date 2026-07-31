"""Short validation sweep for multiple full-stream temporal-frame experts.

The script intentionally keeps the post-processing policy global and fixed.
It is a diagnostic for model complementarity, not a per-video rule search.
Model paths and feature flags are supplied through environment variables so the
normal config parser can continue to own ``--config`` and ``--set`` handling.
"""

import os
from pathlib import Path

import numpy as np
import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.eval import evalute
from utils.inference_chunks import evaluation_batch_from_sample
from utils.postprocess import ChallengePostprocessor
from utils.temporal_frame_inference import (
    predict_temporal_frame_scores,
    load_temporal_frame_model,
    temporal_frame_video_from_sample,
)


MODEL_ENV_NAMES = ('P23', 'P24', 'P25', 'P27')
DEFAULT_MODEL_ENV = {
    'P23': '',
    'P24': '',
    'P25': '',
    'P27': '',
}


def _required_path(name):
    value = os.environ.get(name, DEFAULT_MODEL_ENV[name]).strip()
    if not value:
        raise ValueError('Set ${} to a temporal-frame checkpoint path.'.format(name))
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError('{} checkpoint not found: {}'.format(name, path))
    return str(path)


def _model_specs():
    return [
        {
            'name': 'P23',
            'path': _required_path('P23'),
            'local_contrast': False,
            'motion_persistence': False,
            'fine_detail': False,
        },
        {
            'name': 'P24',
            'path': _required_path('P24'),
            'local_contrast': True,
            'motion_persistence': False,
            'fine_detail': False,
        },
        {
            'name': 'P25',
            'path': _required_path('P25'),
            'local_contrast': False,
            'motion_persistence': True,
            'fine_detail': False,
        },
        {
            'name': 'P27',
            'path': _required_path('P27'),
            'local_contrast': False,
            'motion_persistence': False,
            'fine_detail': True,
        },
    ]


def _load_scores(spec, dataset, device):
    model = load_temporal_frame_model(
        spec['path'],
        device,
        cfg.temporal_frame_context_bins,
        cfg.temporal_frame_width,
        spec['local_contrast'],
        cfg.temporal_frame_local_contrast_kernel_size,
        spec['motion_persistence'],
        cfg.temporal_frame_motion_persistence_radius_per_bin,
        spec['fine_detail'],
        cfg.temporal_frame_fine_temporal_bin_size,
        cfg.temporal_frame_fine_context_bins,
    )[0]
    scores = []
    batches = []
    for video_index in tqdm.trange(
        len(dataset), desc='{} inference'.format(spec['name']), unit='video'
    ):
        sample = dataset[video_index]
        frame_video = temporal_frame_video_from_sample(
            sample,
            cfg.temporal_frame_bin_size,
            cfg.whole_t,
        )
        fine_video = None
        fine_ratio = 1
        if spec['fine_detail']:
            fine_video = temporal_frame_video_from_sample(
                sample,
                cfg.temporal_frame_fine_temporal_bin_size,
                cfg.whole_t,
            )
            fine_ratio = (
                int(cfg.temporal_frame_bin_size)
                // int(cfg.temporal_frame_fine_temporal_bin_size)
            )
        event_scores = predict_temporal_frame_scores(
            model,
            frame_video,
            device,
            cfg.temporal_frame_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_frame_inference_batch_size,
            cfg.temporal_frame_log_count_clip,
            spec['local_contrast'],
            cfg.temporal_frame_local_contrast_kernel_size,
            spec['motion_persistence'],
            cfg.temporal_frame_motion_persistence_radius_per_bin,
            spec['fine_detail'],
            fine_video,
            cfg.temporal_frame_fine_context_bins,
            fine_ratio,
        )
        scores.append(event_scores)
        batches.append(evaluation_batch_from_sample(sample))
    del model
    torch.cuda.empty_cache()
    return scores, batches


def _weight_grid():
    # The first row reproduces the current P23/P24/P25 reference. The other
    # rows add P27 only as a controlled small expert, then test a few broader
    # mixtures without selecting weights per video.
    rows = [
        (0.650, 0.250, 0.100, 0.000),
        (0.625, 0.240, 0.095, 0.040),
        (0.600, 0.230, 0.090, 0.080),
        (0.575, 0.220, 0.085, 0.120),
        (0.550, 0.210, 0.080, 0.160),
        (0.500, 0.200, 0.075, 0.225),
        (0.450, 0.180, 0.070, 0.300),
        (0.000, 0.000, 0.000, 1.000),
    ]
    return rows


def _evaluate(weight, score_vectors, batches, threshold):
    evaluator = evalute(cfg)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, threshold)
    sample_number = 0
    for video_index, batch in enumerate(batches):
        predictions = sum(
            float(weight[model_index]) * torch.as_tensor(
                score_vectors[model_index][video_index]
            )
            for model_index in range(len(score_vectors))
        ).float()
        predictions, _ = postprocessor.apply(predictions, batch['locs'])
        sample_number = add_batch_to_evaluator(
            evaluator,
            batch,
            predictions,
            sample_number,
            threshold,
        )
    return evaluate_challenge_metrics(evaluator, threshold)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for temporal-frame inference.')
    if not cfg.eval or not cfg.roc:
        raise ValueError('Set TEST.eval=true and TEST.roc=true in the config.')
    threshold = float(cfg.prediction_threshold)
    device = torch.device('cuda:0')
    dataset = EvUAV(cfg, mode='val')
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError('No validation files found: {}'.format(dataset.root))
    specs = _model_specs()
    print('validation root:', dataset.root)
    print('validation videos:', len(dataset.file_list))
    print('fixed threshold:', threshold)
    print('postprocessor:', ChallengePostprocessor.from_cfg(cfg, threshold).describe())
    print('models:', ', '.join('{}={}'.format(s['name'], s['path']) for s in specs))

    score_vectors = []
    batches = None
    for spec in specs:
        current_scores, current_batches = _load_scores(spec, dataset, device)
        if batches is None:
            batches = current_batches
        score_vectors.append(current_scores)

    print('\nweights (P23, P24, P25, P27)     Score          Pd          IoU         Acc          Fa')
    results = []
    for weight in _weight_grid():
        metrics = _evaluate(weight, score_vectors, batches, threshold)
        results.append((metrics.score, weight, metrics))
        print(
            '({:.3f}, {:.3f}, {:.3f}, {:.3f})  {:.10f}  {:.8f}  {:.8f}  {:.8f}  {:.8e}'.format(
                *weight,
                metrics.score,
                metrics.pd,
                metrics.iou,
                metrics.acc,
                metrics.fa,
            )
        )
    best_score, best_weight, best_metrics = max(results, key=lambda item: item[0])
    print('\nbest fixed mixture:', best_weight)
    print('best score: {:.10f}'.format(best_score))
