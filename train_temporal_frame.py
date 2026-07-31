"""Train the full-stream temporal event-frame auxiliary model."""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import tqdm
import yaml

from configs.configs import cfg
from dataset.temporal_frame import (
    TemporalFrameTrainDataset,
    temporal_frame_collate,
)
from model.modules.confidence_head import confidence_calibration_loss
from model.temporal_frame_net import (
    TemporalFrameNet,
    append_local_contrast_channels,
    build_motion_persistence_channels,
    gather_event_logits,
)
from utils.multiscale_motion import (
    build_multiscale_motion_persistence_channels,
    multiscale_motion_channel_count,
)
from utils.temporal_frame_loss import (
    build_target_center_heatmaps,
    frame_balanced_event_bce,
    frame_balanced_quality_focal_loss,
    generate_gaussian_soft_labels,
    target_center_heatmap_loss,
    target_group_coverage_loss,
    trajectory_extrapolation_loss_p23,
)


def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_directory(config):
    started_at = datetime.now().astimezone()
    run_name = '{}_seed{}_pid{}'.format(
        started_at.strftime('%Y%m%d-%H%M%S'),
        int(config.seed),
        os.getpid(),
    )
    run_dir = Path(config.model_save_root) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / 'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(
            config.resolved_config,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    return run_dir, started_at


def save_checkpoint(checkpoint, path):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def build_scheduler(optimizer, config):
    scheduler_name = str(config.scheduler).lower()
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.epochs),
            eta_min=float(config.scheduler_min_lr),
        )
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.scheduler_step_size),
            gamma=float(config.scheduler_gamma),
        )
    raise ValueError('Unsupported scheduler for temporal frames: {}'.format(
        config.scheduler
    ))


def _checkpoint_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes', 'on'}
    return bool(value)


def load_initial_temporal_frame_weights(model, config):
    """Load a compatible expert while leaving newly added adapters neutral."""
    checkpoint_path = str(config.temporal_frame_init_model_path).strip()
    if not checkpoint_path:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Temporal-frame initialization checkpoint not found: {}'.format(
                checkpoint_path
            )
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    saved_config = checkpoint.get('temporal_frame', {})
    saved_context_bins = saved_config.get('context_bins')
    saved_width = saved_config.get('width')
    if (
        saved_context_bins is not None
        and int(saved_context_bins) != int(config.temporal_frame_context_bins)
    ):
        raise ValueError(
            'Initialization checkpoint context_bins={} does not match {}.'
            .format(saved_context_bins, config.temporal_frame_context_bins)
        )
    if saved_width is not None and int(saved_width) != int(config.temporal_frame_width):
        raise ValueError(
            'Initialization checkpoint width={} does not match {}.'.format(
                saved_width,
                config.temporal_frame_width,
            )
        )

    target_has_contrast = bool(config.temporal_frame_local_contrast_enabled)
    saved_has_contrast = _checkpoint_bool(
        saved_config.get('local_contrast_enabled', False)
    )
    target_has_motion_persistence = bool(
        config.temporal_frame_motion_persistence_enabled
    )
    saved_has_motion_persistence = _checkpoint_bool(
        saved_config.get('motion_persistence_enabled', False)
    )
    target_has_fine_detail = bool(
        config.temporal_frame_fine_detail_enabled
    )
    saved_has_fine_detail = _checkpoint_bool(
        saved_config.get('fine_detail_enabled', False)
    )
    target_has_target_center = bool(
        getattr(config, 'temporal_frame_target_center_enabled', False)
    )
    saved_has_target_center = _checkpoint_bool(
        saved_config.get('target_center_enabled', False)
    )
    if saved_has_contrast and not target_has_contrast:
        raise ValueError(
            'Cannot initialize a non-contrast model from a contrast checkpoint.'
        )
    if saved_has_motion_persistence and not target_has_motion_persistence:
        raise ValueError(
            'Cannot initialize a non-motion model from a motion checkpoint.'
        )
    if saved_has_fine_detail and not target_has_fine_detail:
        raise ValueError(
            'Cannot initialize a non-fine-detail model from a fine-detail '
            'checkpoint.'
        )
    if saved_has_target_center and not target_has_target_center:
        raise ValueError(
            'Cannot initialize a model without the target-centre branch from '
            'a target-centre checkpoint.'
        )
    if saved_has_contrast and target_has_contrast:
        saved_kernel_size = int(
            saved_config.get('local_contrast_kernel_size', 9)
        )
        if saved_kernel_size != int(config.temporal_frame_local_contrast_kernel_size):
            raise ValueError(
                'Initialization checkpoint local_contrast_kernel_size={} '
                'does not match {}.'.format(
                    saved_kernel_size,
                    config.temporal_frame_local_contrast_kernel_size,
                )
            )
    if saved_has_motion_persistence and target_has_motion_persistence:
        saved_radius_per_bin = int(
            saved_config.get('motion_persistence_radius_per_bin', 4)
        )
        if saved_radius_per_bin != int(
            config.temporal_frame_motion_persistence_radius_per_bin
        ):
            raise ValueError(
                'Initialization checkpoint motion_persistence_radius_per_bin={} '
                'does not match {}.'.format(
                    saved_radius_per_bin,
                    config.temporal_frame_motion_persistence_radius_per_bin,
                )
            )
    if saved_has_fine_detail and target_has_fine_detail:
        saved_fine_temporal_bin_size = int(
            saved_config.get('fine_temporal_bin_size', 25)
        )
        if saved_fine_temporal_bin_size != int(
            config.temporal_frame_fine_temporal_bin_size
        ):
            raise ValueError(
                'Initialization checkpoint fine_temporal_bin_size={} does '
                'not match {}.'.format(
                    saved_fine_temporal_bin_size,
                    config.temporal_frame_fine_temporal_bin_size,
                )
            )
        saved_fine_context_bins = int(
            saved_config.get('fine_context_bins', 9)
        )
        if saved_fine_context_bins != int(
            config.temporal_frame_fine_context_bins
        ):
            raise ValueError(
                'Initialization checkpoint fine_context_bins={} does not '
                'match {}.'.format(
                    saved_fine_context_bins,
                    config.temporal_frame_fine_context_bins,
                )
            )

    if (
        saved_has_contrast == target_has_contrast
        and saved_has_motion_persistence == target_has_motion_persistence
        and saved_has_fine_detail == target_has_fine_detail
        and saved_has_target_center == target_has_target_center
    ):
        model.load_state_dict(state_dict, strict=True)
    else:
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = set()
        if target_has_contrast and not saved_has_contrast:
            allowed_missing.update({
                'local_contrast_adapter.weight',
                'local_contrast_adapter.bias',
            })
        if target_has_motion_persistence and not saved_has_motion_persistence:
            allowed_missing.update({
                'motion_persistence_adapter.weight',
                'motion_persistence_adapter.bias',
            })
        if target_has_fine_detail and not saved_has_fine_detail:
            allowed_missing.update({
                'fine_detail_adapter.weight',
                'fine_detail_adapter.bias',
            })
        if target_has_target_center and not saved_has_target_center:
            allowed_missing.update(
                name
                for name in model.state_dict()
                if name.startswith('target_center_')
            )
        if set(incompatible.missing_keys) != allowed_missing:
            raise RuntimeError(
                'Unexpected missing initialization parameters: {}'.format(
                    incompatible.missing_keys
                )
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                'Unexpected initialization checkpoint parameters: {}'.format(
                    incompatible.unexpected_keys
                )
            )
    return checkpoint_path


def build_optimizer(model, config):
    """Use faster schedules for new adapters without disturbing P23 weights."""
    adapter_groups = []
    if model.local_contrast_adapter is not None:
        multiplier = float(config.temporal_frame_local_contrast_lr_multiplier)
        if multiplier <= 0.0:
            raise ValueError('temporal_frame_local_contrast_lr_multiplier must be positive.')
        adapter_groups.append((
            list(model.local_contrast_adapter.parameters()),
            multiplier,
        ))
    if model.motion_persistence_adapter is not None:
        multiplier = float(
            config.temporal_frame_motion_persistence_lr_multiplier
        )
        if multiplier <= 0.0:
            raise ValueError(
                'temporal_frame_motion_persistence_lr_multiplier must be positive.'
            )
        adapter_groups.append((
            list(model.motion_persistence_adapter.parameters()),
            multiplier,
        ))
    if model.fine_detail_adapter is not None:
        multiplier = float(config.temporal_frame_fine_detail_lr_multiplier)
        if multiplier <= 0.0:
            raise ValueError(
                'temporal_frame_fine_detail_lr_multiplier must be positive.'
            )
        adapter_groups.append((
            list(model.fine_detail_adapter.parameters()),
            multiplier,
        ))
    if model.target_center_enabled:
        multiplier = float(config.temporal_frame_target_center_lr_multiplier)
        if multiplier <= 0.0:
            raise ValueError(
                'temporal_frame_target_center_lr_multiplier must be positive.'
            )
        adapter_groups.append((
            list(model.target_center_head.parameters())
            + list(model.target_center_residual.parameters()),
            multiplier,
        ))

    adapter_parameter_ids = {
        id(parameter)
        for parameters, _ in adapter_groups
        for parameter in parameters
    }
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in adapter_parameter_ids and parameter.requires_grad
    ]
    parameter_groups = []
    if base_parameters:
        parameter_groups.append({
            'params': base_parameters,
            'lr': float(config.lr),
        })
    parameter_groups.extend(
        {
            'params': parameters,
            'lr': float(config.lr) * multiplier,
        }
        for parameters, multiplier in adapter_groups
    )
    if not parameter_groups:
        raise ValueError('Temporal-frame optimizer received no trainable parameters.')
    return optim.AdamW(
        parameter_groups,
        weight_decay=1e-4,
    )


def frame_config_summary(config):
    return (
        'enabled (bin_size={}, context_bins={}, width={}, batch_size={}, '
        'views_per_video={}, positive_frame_probability={}, '
        'target_positive_loss_mass={}, max_positive_weight={}, '
        'local_contrast={}, local_contrast_kernel_size={}, '
        'motion_persistence={}, motion_radius_per_bin={}, '
        'fine_detail={}, fine_temporal_bin_size={}, fine_context_bins={}, '
        'target_center={}, target_center_loss_weight={}, '
        'target_center_sigma={}, target_center_radius={}, '
        'target_center_freeze_base={}, '
        'dense_sampling={}, dense_event_count_cutoff={}, '
        'dense_view_multiplier={}, '
        'trajectory_extrapolation={})'
    ).format(
        config.temporal_frame_bin_size,
        config.temporal_frame_context_bins,
        config.temporal_frame_width,
        config.temporal_frame_batch_size,
        config.temporal_frame_train_views_per_video,
        config.temporal_frame_positive_frame_probability,
        config.temporal_frame_target_positive_loss_mass,
        config.temporal_frame_max_positive_weight,
        config.temporal_frame_local_contrast_enabled,
        config.temporal_frame_local_contrast_kernel_size,
        config.temporal_frame_motion_persistence_enabled,
        config.temporal_frame_motion_persistence_radius_per_bin,
        config.temporal_frame_fine_detail_enabled,
        config.temporal_frame_fine_temporal_bin_size,
        config.temporal_frame_fine_context_bins,
        config.temporal_frame_target_center_enabled,
        config.temporal_frame_target_center_loss_weight,
        config.temporal_frame_target_center_sigma,
        config.temporal_frame_target_center_radius,
        config.temporal_frame_target_center_freeze_base_enabled,
        config.temporal_frame_dense_sampling_enabled,
        config.temporal_frame_dense_event_count_cutoff,
        config.temporal_frame_dense_view_multiplier,
        config.temporal_frame_trajectory_extrapolation_enabled,
    )


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for temporal-frame training.')

    if int(cfg.temporal_frame_context_bins) % 2 == 0:
        raise ValueError('TEMPORAL_FRAME.context_bins must be odd.')
    if int(cfg.temporal_frame_train_workers) != 0 and bool(
        cfg.temporal_frame_cache_all_videos
    ):
        raise ValueError(
            'Use TEMPORAL_FRAME.train_workers=0 when cache_all_videos=true '
            'to avoid duplicating complete event streams in worker processes.'
        )
    if int(cfg.epochs) <= 0:
        raise ValueError('TRAIN.epochs must be positive.')
    if bool(cfg.temporal_frame_local_contrast_enabled) and (
        int(cfg.temporal_frame_local_contrast_kernel_size) <= 0
        or int(cfg.temporal_frame_local_contrast_kernel_size) % 2 == 0
    ):
        raise ValueError(
            'TEMPORAL_FRAME.local_contrast_kernel_size must be a positive '
            'odd integer.'
        )
    if bool(cfg.temporal_frame_motion_persistence_enabled) and (
        int(cfg.temporal_frame_context_bins) < 3
        or int(cfg.temporal_frame_context_bins) % 2 == 0
    ):
        raise ValueError(
            'TEMPORAL_FRAME.motion_persistence requires at least three '
            'odd context bins.'
        )
    if int(cfg.temporal_frame_motion_persistence_radius_per_bin) < 0:
        raise ValueError(
            'TEMPORAL_FRAME.motion_persistence_radius_per_bin must be '
            'non-negative.'
        )
    if bool(cfg.temporal_frame_fine_detail_enabled):
        fine_temporal_bin_size = int(
            cfg.temporal_frame_fine_temporal_bin_size
        )
        if fine_temporal_bin_size <= 0:
            raise ValueError(
                'TEMPORAL_FRAME.fine_temporal_bin_size must be positive.'
            )
        if fine_temporal_bin_size > int(cfg.temporal_frame_bin_size):
            raise ValueError(
                'TEMPORAL_FRAME.fine_temporal_bin_size must not exceed '
                'temporal_frame_bin_size.'
            )
        if int(cfg.temporal_frame_bin_size) % fine_temporal_bin_size != 0:
            raise ValueError(
                'TEMPORAL_FRAME.temporal_frame_bin_size must be divisible by '
                'fine_temporal_bin_size.'
            )
        if (
            int(cfg.temporal_frame_fine_context_bins) < 1
            or int(cfg.temporal_frame_fine_context_bins) % 2 == 0
        ):
            raise ValueError(
                'TEMPORAL_FRAME.fine_context_bins must be a positive odd '
                'integer.'
            )

    coverage_enabled = bool(
        getattr(cfg, 'temporal_frame_target_coverage_enabled', False)
    )
    coverage_weight = float(
        getattr(cfg, 'temporal_frame_target_coverage_weight', 0.01)
    )
    coverage_warmup_epochs = int(
        getattr(cfg, 'temporal_frame_target_coverage_warmup_epochs', 5)
    )
    coverage_score_floor = float(
        getattr(cfg, 'temporal_frame_target_coverage_score_floor', 0.70)
    )
    coverage_correct_fraction = float(
        getattr(cfg, 'temporal_frame_target_coverage_correct_fraction', 0.0001)
    )
    if coverage_enabled:
        if coverage_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_coverage_weight must be positive.'
            )
        if coverage_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_FRAME.target_coverage_warmup_epochs must be non-negative.'
            )
        if not 0.0 < coverage_score_floor < 1.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_coverage_score_floor must be in (0, 1).'
            )
        if not 0.0 < coverage_correct_fraction <= 1.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_coverage_correct_fraction must be in (0, 1].'
            )

    target_center_enabled = bool(
        getattr(cfg, 'temporal_frame_target_center_enabled', False)
    )
    target_center_loss_weight = float(
        getattr(cfg, 'temporal_frame_target_center_loss_weight', 0.05)
    )
    target_center_warmup_epochs = int(
        getattr(cfg, 'temporal_frame_target_center_warmup_epochs', 0)
    )
    target_center_sigma = float(
        getattr(cfg, 'temporal_frame_target_center_sigma', 2.5)
    )
    target_center_radius = int(
        getattr(cfg, 'temporal_frame_target_center_radius', 6)
    )
    target_center_positive_loss_mass = float(
        getattr(cfg, 'temporal_frame_target_center_positive_loss_mass', 0.20)
    )
    target_center_max_positive_weight = float(
        getattr(cfg, 'temporal_frame_target_center_max_positive_weight', 512.0)
    )
    target_center_empty_loss_weight = float(
        getattr(cfg, 'temporal_frame_target_center_empty_loss_weight', 0.10)
    )
    target_center_freeze_base = bool(
        getattr(
            cfg,
            'temporal_frame_target_center_freeze_base_enabled',
            False,
        )
    )
    if target_center_enabled:
        if target_center_loss_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_center_loss_weight must be positive.'
            )
        if target_center_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_FRAME.target_center_warmup_epochs must be non-negative.'
            )
        if target_center_sigma <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_center_sigma must be positive.'
            )
        if target_center_radius <= 0:
            raise ValueError(
                'TEMPORAL_FRAME.target_center_radius must be positive.'
            )
        if not 0.0 < target_center_positive_loss_mass < 1.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_center_positive_loss_mass must be in (0, 1).'
            )
        if target_center_max_positive_weight < 1.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_center_max_positive_weight must be at least one.'
            )
        if not 0.0 <= target_center_empty_loss_weight <= 1.0:
            raise ValueError(
                'TEMPORAL_FRAME.target_center_empty_loss_weight must be in [0, 1].'
            )

    traj_extrap_enabled = bool(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_enabled', False)
    )
    traj_extrap_weight = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_weight', 0.05)
    )
    traj_extrap_margin = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_margin_logit', 1.0)
    )
    traj_extrap_min_points = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_min_points', 3)
    )
    traj_extrap_warmup_epochs = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_warmup_epochs', 3)
    )
    if traj_extrap_enabled:
        if traj_extrap_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_weight must be positive.'
            )
        if traj_extrap_min_points < 2:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_min_points must be at least 2.'
            )
        if traj_extrap_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_warmup_epochs must be non-negative.'
            )

    # M6: Gaussian soft labels + Quality Focal Loss
    m6_gaussian_soft_labels_enabled = bool(
        getattr(cfg, 'temporal_frame_gaussian_soft_labels_enabled', False)
    )
    m6_gaussian_sigma = float(
        getattr(cfg, 'temporal_frame_gaussian_sigma', 2.5)
    )
    m6_qfl_enabled = bool(
        getattr(cfg, 'temporal_frame_quality_focal_loss_enabled', False)
    )
    m6_qfl_beta = float(
        getattr(cfg, 'temporal_frame_quality_focal_beta', 2.0)
    )
    if m6_gaussian_soft_labels_enabled:
        if m6_gaussian_sigma <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.temporal_frame_gaussian_sigma must be positive.'
            )
    if m6_qfl_enabled:
        if m6_qfl_beta < 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.temporal_frame_quality_focal_beta must be non-negative.'
            )

    setup_seed(cfg.seed)
    device = torch.device('cuda:0')
    run_dir, started_at = create_run_directory(cfg)
    checkpoint_interval = int(getattr(cfg, 'checkpoint_interval', 0))
    if checkpoint_interval < 0:
        raise ValueError('checkpoint_interval must be non-negative.')
    periodic_checkpoints = []
    train_root = Path(cfg.root) / 'train'
    dataset = TemporalFrameTrainDataset(
        root=train_root,
        whole_t=cfg.whole_t,
        temporal_bin_size=cfg.temporal_frame_bin_size,
        context_bins=cfg.temporal_frame_context_bins,
        width=cfg.res[0],
        height=cfg.res[1],
        views_per_video=cfg.temporal_frame_train_views_per_video,
        positive_frame_probability=cfg.temporal_frame_positive_frame_probability,
        random_seed=cfg.seed,
        log_count_clip=cfg.temporal_frame_log_count_clip,
        cache_all_videos=cfg.temporal_frame_cache_all_videos,
        cache_video_count=cfg.temporal_frame_cache_video_count,
        dense_sampling_enabled=cfg.temporal_frame_dense_sampling_enabled,
        dense_event_count_cutoff=cfg.temporal_frame_dense_event_count_cutoff,
        dense_view_multiplier=cfg.temporal_frame_dense_view_multiplier,
        fine_detail_enabled=cfg.temporal_frame_fine_detail_enabled,
        fine_temporal_bin_size=cfg.temporal_frame_fine_temporal_bin_size,
        fine_context_bins=cfg.temporal_frame_fine_context_bins,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.temporal_frame_batch_size),
        shuffle=False,
        num_workers=int(cfg.temporal_frame_train_workers),
        collate_fn=temporal_frame_collate,
        pin_memory=True,
    )
    model = TemporalFrameNet(
        input_channels=int(cfg.temporal_frame_context_bins) * 2,
        width=int(cfg.temporal_frame_width),
        local_contrast_channels=(
            int(cfg.temporal_frame_context_bins) * 2
            if cfg.temporal_frame_local_contrast_enabled else 0
        ),
        motion_persistence_channels=(
            multiscale_motion_channel_count(cfg.temporal_frame_context_bins)
            if cfg.temporal_frame_motion_persistence_enabled else 0
        ),
        fine_detail_channels=(
            int(cfg.temporal_frame_fine_context_bins) * 2
            if cfg.temporal_frame_fine_detail_enabled else 0
        ),
        target_center_enabled=target_center_enabled,
        confidence_head_enabled=getattr(
            cfg, 'temporal_frame_confidence_head_enabled', False,
        ),
        density_calibration_enabled=getattr(
            cfg, 'temporal_frame_density_calibration_enabled', False,
        ),
    ).to(device)
    initialized_from = load_initial_temporal_frame_weights(model, cfg)
    if target_center_enabled and target_center_freeze_base:
        for parameter_name, parameter in model.named_parameters():
            if not parameter_name.startswith('target_center_'):
                parameter.requires_grad_(False)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    print('random seed:{}'.format(cfg.seed))
    print('run directory:', run_dir)
    print('config overrides:', ', '.join(cfg.config_overrides) or '(none)')
    if checkpoint_interval:
        print('periodic checkpoint interval:', checkpoint_interval, 'epochs')
    print('temporal-frame model:', frame_config_summary(cfg))
    if coverage_enabled:
        print(
            'target-group coverage loss: enabled (weight={}, warmup_epochs={}, '
            'score_floor={}, correct_fraction={})'.format(
                coverage_weight,
                coverage_warmup_epochs,
                coverage_score_floor,
                coverage_correct_fraction,
            )
        )
    else:
        print('target-group coverage loss: disabled')
    if target_center_enabled:
        print(
            'P32 target-centre branch: enabled (loss_weight={}, '
            'warmup_epochs={}, sigma={}, radius={}, '
            'positive_loss_mass={}, max_positive_weight={}, '
            'empty_loss_weight={})'.format(
                target_center_loss_weight,
                target_center_warmup_epochs,
                target_center_sigma,
                target_center_radius,
                target_center_positive_loss_mass,
                target_center_max_positive_weight,
                target_center_empty_loss_weight,
            )
        )
        if target_center_freeze_base:
            print('P32 target-centre branch: P23 backbone and event head frozen')
    else:
        print('P32 target-centre branch: disabled')
    if traj_extrap_enabled:
        print(
            'M5 trajectory extrapolation loss: enabled (weight={}, '
            'margin_logit={}, min_points={}, warmup_epochs={})'.format(
                traj_extrap_weight,
                traj_extrap_margin,
                traj_extrap_min_points,
                traj_extrap_warmup_epochs,
            )
        )
    else:
        print('M5 trajectory extrapolation loss: disabled')
    if m6_gaussian_soft_labels_enabled:
        print(
            'M6 Gaussian soft labels: enabled (sigma={:.2f})'.format(
                m6_gaussian_sigma,
            )
        )
    else:
        print('M6 Gaussian soft labels: disabled')
    if m6_qfl_enabled:
        print(
            'M6 Quality Focal Loss: enabled (beta={:.2f})'.format(m6_qfl_beta)
        )
    else:
        print('M6 Quality Focal Loss: disabled')
    print('training videos:', len(dataset.file_paths))
    print('training views per epoch:', len(dataset))
    if cfg.temporal_frame_dense_sampling_enabled:
        print(
            'dense training videos: {} (event_count >= {})'.format(
                dataset.dense_video_count,
                cfg.temporal_frame_dense_event_count_cutoff,
            )
        )
    if initialized_from is not None:
        print('initialized temporal-frame weights from:', initialized_from)
    print('learning-rate scheduler:', cfg.scheduler)

    best_loss = float('inf')
    best_epoch = None
    for epoch in range(int(cfg.epochs)):
        dataset.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        positive_fraction_sum = 0.0
        positive_weight_sum = 0.0
        coverage_loss_sum = 0.0
        target_center_loss_sum = 0.0
        traj_extrap_loss_sum = 0.0
        batch_count = 0
        pbar = tqdm.tqdm(
            dataloader,
            desc='Epoch: {}'.format(epoch),
            unit='Batch',
            position=0,
            leave=True,
        )
        for batch in pbar:
            raw_frames = batch['frames'].to(device, non_blocking=True)
            frames = raw_frames
            if cfg.temporal_frame_local_contrast_enabled:
                frames = append_local_contrast_channels(
                    raw_frames,
                    cfg.temporal_frame_local_contrast_kernel_size,
                )
            if cfg.temporal_frame_motion_persistence_enabled:
                motion_channels = build_multiscale_motion_persistence_channels(
                    raw_frames,
                    cfg.temporal_frame_context_bins,
                )
                frames = torch.cat((frames, motion_channels), dim=1)
            if cfg.temporal_frame_fine_detail_enabled:
                fine_detail_frames = batch['fine_detail_frames'].to(
                    device,
                    non_blocking=True,
                )
                frames = torch.cat((frames, fine_detail_frames), dim=1)
            event_batch_indices = batch['event_batch_indices'].to(
                device,
                non_blocking=True,
            )
            event_y = batch['event_y'].to(device, non_blocking=True)
            event_x = batch['event_x'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            confidence_head_enabled = getattr(
                cfg, 'temporal_frame_confidence_head_enabled', False,
            )
            if target_center_enabled or confidence_head_enabled:
                extra_outputs = model(
                    frames,
                    return_target_center_logits=target_center_enabled,
                    return_confidence_logits=confidence_head_enabled,
                )
                if target_center_enabled and confidence_head_enabled:
                    logit_maps, target_center_logits, confidence_logits = extra_outputs
                elif target_center_enabled:
                    logit_maps, target_center_logits = extra_outputs
                    confidence_logits = None
                else:
                    logit_maps, confidence_logits = extra_outputs
                    target_center_logits = None
            else:
                logit_maps = model(frames)
                target_center_logits = None
                confidence_logits = None
            event_logits = gather_event_logits(
                logit_maps,
                event_batch_indices,
                event_y,
                event_x,
            )
            effective_labels = labels
            if m6_gaussian_soft_labels_enabled:
                effective_labels = generate_gaussian_soft_labels(
                    event_x,
                    event_y,
                    labels,
                    target_ids,
                    event_batch_indices,
                    sigma=m6_gaussian_sigma,
                )
            if m6_qfl_enabled:
                loss, diagnostics = frame_balanced_quality_focal_loss(
                    event_logits,
                    effective_labels,
                    event_batch_indices,
                    target_positive_loss_mass=(
                        cfg.temporal_frame_target_positive_loss_mass
                    ),
                    max_positive_weight=cfg.temporal_frame_max_positive_weight,
                    beta=m6_qfl_beta,
                )
            else:
                loss, diagnostics = frame_balanced_event_bce(
                    event_logits,
                    effective_labels,
                    event_batch_indices,
                    target_positive_loss_mass=(
                        cfg.temporal_frame_target_positive_loss_mass
                    ),
                    max_positive_weight=cfg.temporal_frame_max_positive_weight,
                )
            coverage_loss = event_logits.sum() * 0.0
            if coverage_enabled and epoch >= coverage_warmup_epochs:
                coverage_loss, _ = target_group_coverage_loss(
                    event_logits,
                    labels,
                    target_ids,
                    event_batch_indices,
                    score_floor=coverage_score_floor,
                    correct_fraction=coverage_correct_fraction,
                )
                loss = loss + coverage_weight * coverage_loss
            target_center_loss = event_logits.sum() * 0.0
            if (
                target_center_enabled
                and epoch >= target_center_warmup_epochs
            ):
                target_heatmaps = build_target_center_heatmaps(
                    event_x,
                    event_y,
                    labels,
                    target_ids,
                    event_batch_indices,
                    batch_size=logit_maps.shape[0],
                    height=logit_maps.shape[2],
                    width=logit_maps.shape[3],
                    sigma=target_center_sigma,
                    radius=target_center_radius,
                )
                target_center_loss, _ = target_center_heatmap_loss(
                    target_center_logits,
                    target_heatmaps,
                    target_positive_loss_mass=(
                        target_center_positive_loss_mass
                    ),
                    max_positive_weight=target_center_max_positive_weight,
                    empty_loss_weight=target_center_empty_loss_weight,
                )
                loss = loss + target_center_loss_weight * target_center_loss
            traj_extrap_loss = event_logits.sum() * 0.0
            if traj_extrap_enabled and epoch >= traj_extrap_warmup_epochs:
                traj_extrap_loss, traj_extrap_stats = (
                    trajectory_extrapolation_loss_p23(
                        logit_maps,
                        batch['video_indices'].to(device, non_blocking=True),
                        batch['center_bins'].to(device, non_blocking=True),
                        event_x,
                        event_y,
                        labels,
                        target_ids,
                        event_batch_indices,
                        min_known_points=traj_extrap_min_points,
                        margin_logit=traj_extrap_margin,
                    )
                )
                loss = loss + traj_extrap_weight * traj_extrap_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.detach().item())
            positive_fraction_sum += diagnostics['positive_fraction']
            positive_weight_sum += diagnostics['mean_positive_weight']
            coverage_loss_sum += float(coverage_loss.detach().item())
            target_center_loss_sum += float(target_center_loss.detach().item())
            traj_extrap_loss_sum += float(traj_extrap_loss.detach().item())
            batch_count += 1
            postfix = {
                'loss': '{:.5f}'.format(loss_sum / batch_count),
                'pos': '{:.4f}'.format(positive_fraction_sum / batch_count),
                'pos_w': '{:.2f}'.format(positive_weight_sum / batch_count),
            }
            if coverage_enabled and epoch >= coverage_warmup_epochs:
                postfix['p26'] = '{:.4f}'.format(
                    coverage_loss_sum / batch_count
                )
            if target_center_enabled and epoch >= target_center_warmup_epochs:
                postfix['p32'] = '{:.4f}'.format(
                    target_center_loss_sum / batch_count
                )
            if traj_extrap_enabled and epoch >= traj_extrap_warmup_epochs:
                postfix['m5'] = '{:.4f}'.format(
                    traj_extrap_loss_sum / batch_count
                )
            pbar.set_postfix(postfix)
        pbar.close()
        scheduler.step()

        epoch_loss = loss_sum / max(batch_count, 1)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'temporal_frame': {
                'temporal_bin_size': int(cfg.temporal_frame_bin_size),
                'context_bins': int(cfg.temporal_frame_context_bins),
                'width': int(cfg.temporal_frame_width),
                'log_count_clip': float(cfg.temporal_frame_log_count_clip),
                'local_contrast_enabled': bool(
                    cfg.temporal_frame_local_contrast_enabled
                ),
                'local_contrast_kernel_size': int(
                    cfg.temporal_frame_local_contrast_kernel_size
                ),
                'motion_persistence_enabled': bool(
                    cfg.temporal_frame_motion_persistence_enabled
                ),
                'motion_persistence_radius_per_bin': int(
                    cfg.temporal_frame_motion_persistence_radius_per_bin
                ),
                'fine_detail_enabled': bool(
                    cfg.temporal_frame_fine_detail_enabled
                ),
                'fine_temporal_bin_size': int(
                    cfg.temporal_frame_fine_temporal_bin_size
                ),
                'fine_context_bins': int(
                    cfg.temporal_frame_fine_context_bins
                ),
                'target_center_enabled': target_center_enabled,
                'target_center_loss_weight': target_center_loss_weight,
                'target_center_warmup_epochs': target_center_warmup_epochs,
                'target_center_sigma': target_center_sigma,
                'target_center_radius': target_center_radius,
                'target_center_positive_loss_mass': (
                    target_center_positive_loss_mass
                ),
                'target_center_max_positive_weight': (
                    target_center_max_positive_weight
                ),
                'target_center_empty_loss_weight': (
                    target_center_empty_loss_weight
                ),
                'target_center_freeze_base_enabled': target_center_freeze_base,
                'confidence_head_enabled': getattr(
                    cfg, 'temporal_frame_confidence_head_enabled', False,
                ),
                'density_calibration_enabled': getattr(
                    cfg, 'temporal_frame_density_calibration_enabled', False,
                ),
                'trajectory_extrapolation_enabled': traj_extrap_enabled,
                'trajectory_extrapolation_weight': traj_extrap_weight,
                'trajectory_extrapolation_margin_logit': traj_extrap_margin,
                'trajectory_extrapolation_min_points': traj_extrap_min_points,
                'trajectory_extrapolation_warmup_epochs': traj_extrap_warmup_epochs,
                'gaussian_soft_labels_enabled': m6_gaussian_soft_labels_enabled,
                'gaussian_sigma': m6_gaussian_sigma,
                'qfl_enabled': m6_qfl_enabled,
                'qfl_beta': m6_qfl_beta,
            },
        }
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            save_checkpoint(
                checkpoint,
                run_dir / 'best_loss_seed{}.pt'.format(cfg.seed),
            )
        save_checkpoint(
            checkpoint,
            run_dir / 'last_seed{}.pt'.format(cfg.seed),
        )
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            periodic_path = run_dir / 'epoch_{:03d}_seed{}.pt'.format(
                epoch + 1,
                cfg.seed,
            )
            save_checkpoint(checkpoint, periodic_path)
            periodic_checkpoints.append(str(periodic_path))
        print(
            'epoch {}: loss={:.6f}, lr={:.8f}, best_loss={:.6f}'.format(
                epoch,
                epoch_loss,
                optimizer.param_groups[0]['lr'],
                best_loss,
            )
        )

    summary = {
        'started_at': started_at.isoformat(timespec='seconds'),
        'seed': int(cfg.seed),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'best_loss_checkpoint': str(
            run_dir / 'best_loss_seed{}.pt'.format(cfg.seed)
        ),
        'last_checkpoint': str(run_dir / 'last_seed{}.pt'.format(cfg.seed)),
        'checkpoint_interval': checkpoint_interval,
        'periodic_checkpoints': periodic_checkpoints,
        'config_overrides': list(cfg.config_overrides),
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print('best loss checkpoint:', summary['best_loss_checkpoint'])
    print('last checkpoint:', summary['last_checkpoint'])
