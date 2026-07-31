import json
import os
import argparse
from datetime import datetime
from pathlib import Path
import yaml
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
import random
from model.evspsegnet import evspsegnet
from utils.stcloss import STCLoss

import torch.optim as optim
import mlflow
import tqdm
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.checkpoint import load_state_dict_with_optional_compatibility
from utils.density_threshold import DensityAdaptiveThresholdConfig
from utils.eval import evalute
from utils.lr_scheduler import build_lr_scheduler, describe_lr_scheduler
from utils.postprocess import ChallengePostprocessor


PREDICTION_THRESHOLD = float(getattr(cfg, 'prediction_threshold', 0.9))


def setup(seed):
    seed_n = seed
    print('random seed:' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)


def create_run_directory(config, seed):
    """Create an immutable per-run checkpoint directory and save its resolved config."""
    started_at = datetime.now().astimezone()
    run_name = '{}_seed{}_pid{}'.format(
        started_at.strftime('%Y%m%d-%H%M%S'), seed, os.getpid()
    )
    run_dir = Path(config.model_save_root) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / 'config.yaml').open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            config.resolved_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )
    return run_dir, started_at


def save_checkpoint(state_dict, checkpoint_path):
    """Avoid leaving a partial checkpoint if saving is interrupted."""
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def save_full_checkpoint(model, optimizer, scheduler, epoch, best_loss, best_iou,
                       best_score, best_score_metrics, path):
    """Save full training state for resumption."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'best_iou': best_iou,
        'best_score': best_score,
        'best_score_metrics': best_score_metrics,
    }
    save_checkpoint(state, path)


def load_full_checkpoint(path, model, optimizer, scheduler):
    """Load full training state. Returns (epoch, best_loss, best_iou, best_score, best_score_metrics)."""
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    return (state['epoch'], state['best_loss'], state['best_iou'],
            state.get('best_score'), state.get('best_score_metrics'))


def load_initial_weights(net, model_path, device):
    """Initialize a new run from compatible model weights, if requested."""
    if not model_path:
        return False

    checkpoint_path = Path(model_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError('Initial model weight not found: {}'.format(
            checkpoint_path
        ))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    initialized_optional_keys = load_state_dict_with_optional_compatibility(
        net,
        state_dict,
        p2b_enabled=bool(getattr(net, 'p2b_density_gdsca_enabled', False)),
        p11_enabled=bool(getattr(net, 'p11_local_activity_enabled', False)),
        p12_enabled=bool(getattr(net, 'p12_local_density_enabled', False)),
    )
    print('initialized model weights from:', checkpoint_path)
    if initialized_optional_keys:
        print(
            'initialized missing optional parameters with neutral values:',
            ', '.join(initialized_optional_keys),
        )
    return True


def build_optimizer(net, config):
    """Keep baseline Adam while allowing neutral optional branches to catch up."""
    gate_multiplier = float(getattr(config, 'p2b_gate_lr_multiplier', 1.0))
    p11_multiplier = float(getattr(config, 'p11_feature_lr_multiplier', 1.0))
    p12_multiplier = float(getattr(config, 'p12_feature_lr_multiplier', 1.0))
    if gate_multiplier <= 0:
        raise ValueError('p2b_gate_lr_multiplier must be positive.')
    if p11_multiplier <= 0:
        raise ValueError('p11_feature_lr_multiplier must be positive.')
    if p12_multiplier <= 0:
        raise ValueError('p12_feature_lr_multiplier must be positive.')
    p2b_enabled = bool(getattr(config, 'p2b_density_gdsca_enabled', False))
    p11_enabled = bool(getattr(config, 'p11_local_activity_enabled', False))
    p12_enabled = bool(getattr(config, 'p12_local_density_enabled', False))
    if not p2b_enabled and not p11_enabled and not p12_enabled:
        return optim.Adam(
            filter(lambda parameter: parameter.requires_grad, net.parameters()),
            lr=config.lr,
        )

    gate_parameters = []
    p11_parameters = []
    p12_parameters = []
    backbone_parameters = []
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        if p2b_enabled and 'density_gate' in name:
            gate_parameters.append(parameter)
        elif p11_enabled and name.startswith('p11_activity_projection.'):
            p11_parameters.append(parameter)
        elif p12_enabled and name.startswith('p12_density_projection.'):
            p12_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    if p2b_enabled and not gate_parameters:
        raise RuntimeError('P2b is enabled but no density-gate parameters were found.')
    if p11_enabled and not p11_parameters:
        raise RuntimeError('P11 is enabled but no activity-projection parameters were found.')
    if p12_enabled and not p12_parameters:
        raise RuntimeError('P12 is enabled but no density-projection parameters were found.')
    parameter_groups = [{'params': backbone_parameters, 'lr': config.lr}]
    if gate_parameters:
        parameter_groups.append(
            {'params': gate_parameters, 'lr': config.lr * gate_multiplier}
        )
    if p11_parameters:
        parameter_groups.append(
            {'params': p11_parameters, 'lr': config.lr * p11_multiplier}
        )
    if p12_parameters:
        parameter_groups.append(
            {'params': p12_parameters, 'lr': config.lr * p12_multiplier}
        )
    return optim.Adam(parameter_groups)


def write_run_summary(
    run_dir,
    started_at,
    seed,
    best_loss,
    best_iou,
    best_score,
    best_score_metrics,
    config_overrides,
):
    summary = {
        'started_at': started_at.isoformat(timespec='seconds'),
        'seed': seed,
        'best_loss': best_loss,
        'best_iou': best_iou,
        'best_loss_checkpoint': str(run_dir / 'best_loss_seed{}.pt'.format(seed)),
        'best_iou_checkpoint': (
            str(run_dir / 'best_iou_seed{}.pt'.format(seed))
            if best_iou is not None else None
        ),
        'best_score': best_score,
        'best_score_metrics': best_score_metrics,
        'best_score_checkpoint': (
            str(run_dir / 'best_score_seed{}.pt'.format(seed))
            if best_score is not None else None
        ),
        'last_checkpoint': str(run_dir / 'last_seed{}.pt'.format(seed)),
        'config_overrides': config_overrides,
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint directory to resume from')
    args = parser.parse_known_args()[0]  # configs.py already consumed --config/--set

    seed = int(getattr(cfg, 'seed', 37))
    setup(seed)
    device = "cuda:0"

    # Determine run directory: reuse existing if resuming, otherwise create new
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.is_dir():
            raise NotADirectoryError('Resume directory not found: {}'.format(resume_dir))
        run_dir = resume_dir
        started_at = datetime.now().astimezone()
        resume_ckpt_path = run_dir / 'last_seed{}.pt'.format(seed)
        if not resume_ckpt_path.is_file():
            raise FileNotFoundError('Checkpoint not found: {}'.format(resume_ckpt_path))
        print('resuming from:', run_dir)
    else:
        run_dir, started_at = create_run_directory(cfg, seed)
    best_loss_path = run_dir / 'best_loss_seed{}.pt'.format(seed)
    best_iou_path = run_dir / 'best_iou_seed{}.pt'.format(seed)
    best_score_path = run_dir / 'best_score_seed{}.pt'.format(seed)
    last_path = run_dir / 'last_seed{}.pt'.format(seed)
    checkpoint_interval = int(getattr(cfg, 'checkpoint_interval', 0))
    if checkpoint_interval < 0:
        raise ValueError('checkpoint_interval must be non-negative.')
    print('run directory:', run_dir)
    if cfg.config_overrides:
        print('config overrides:', ', '.join(cfg.config_overrides))
    if checkpoint_interval:
        print('periodic checkpoint interval:', checkpoint_interval, 'epochs')

    net = evspsegnet(cfg).train()
    net.cuda()
    initialized_from_checkpoint = load_initial_weights(
        net,
        cfg.init_model_path,
        device,
    )
    if cfg.p2b_density_gdsca_enabled:
        print('P2b density-conditioned GDSCA: enabled (encoder stages 1-3)')
    else:
        print('P2b density-conditioned GDSCA: disabled')
    if getattr(cfg, 'p16_global_patch_attention_enabled', False):
        print('P16 global patch attention: enabled')
    else:
        print('P16 global patch attention: disabled (legacy token-isolated path)')
    if getattr(cfg, 'p11_local_activity_enabled', False):
        print(
            'P11 local activity feature: enabled '
            '(same-pixel temporal radius={})'.format(
                cfg.p11_local_activity_radius,
            )
        )
    else:
        print('P11 local activity feature: disabled')
    if getattr(cfg, 'p12_local_density_enabled', False):
        print(
            'P12 local density feature: enabled '
            '(spatial_cell_size={}, temporal_cell_size={}, '
            'neighborhood_radius={})'.format(
                cfg.p12_spatial_cell_size,
                cfg.p12_temporal_cell_size,
                cfg.p12_neighborhood_radius,
            )
        )
    else:
        print('P12 local density feature: disabled')

    dataset = EvUAV(cfg,mode='train')
    train_sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate, sampler=train_sampler)
    if dataset.temporal_chunk_enabled:
        print(
            'training event sampling: contiguous temporal chunks '
            '({} chunks from {} videos, max_events_num={})'.format(
                len(dataset),
                dataset.source_video_count,
                cfg.max_events_num,
            )
        )
    elif dataset.density_dual_view_enabled:
        print(
            'training event sampling: density-aware dual views '
            '({} dense videos, {} extra uniform views; {} samples from {} videos)'.format(
                dataset.density_dual_view_source_video_count,
                dataset.density_dual_view_extra_sample_count,
                len(dataset),
                dataset.source_video_count,
            )
        )
    elif dataset.dense_target_oversampling_enabled:
        print(
            'training event sampling: target-preserving dense-video oversampling '
            '({} dense videos, {} extra target-preserving views; {} samples '
            'from {} videos; cutoff={}, factor={})'.format(
                dataset.dense_target_oversampling_source_video_count,
                dataset.dense_target_oversampling_extra_sample_count,
                len(dataset),
                dataset.source_video_count,
                dataset.dense_target_oversampling_event_count_cutoff,
                dataset.dense_target_oversampling_factor,
            )
        )
    elif dataset.dense_specialist_enabled:
        dense_specialist_view = (
            'target-preserving'
            if dataset.dense_specialist_target_preserving_enabled
            else 'uniform'
        )
        print(
            'training event sampling: dense-scene specialist '
            '({} oversized videos, {} views per epoch; cutoff={}, '
            'views_per_video={}, sampling={})'.format(
                dataset.dense_specialist_source_video_count,
                dataset.dense_specialist_sample_count,
                dataset.dense_specialist_event_count_cutoff,
                dataset.dense_specialist_views_per_video,
                dense_specialist_view,
            )
        )
    elif cfg.target_preserving_enabled:
        print('training event sampling: target-preserving')
    else:
        print('training event sampling: uniform random (baseline)')
    if dataset.horizontal_flip_augmentation_enabled:
        print(
            'training spatial augmentation: horizontal flip '
            '(probability={})'.format(
                dataset.horizontal_flip_augmentation_probability,
            )
        )

    stc_criterion = STCLoss(k=cfg.k,t=cfg.t,cfg=cfg).cuda()
    print('P1 background hard-negative loss:', stc_criterion.describe_p1())
    print('P2 positive STC floor:', stc_criterion.describe_p2())
    print('P4 target-frame detection loss:', stc_criterion.describe_p4())
    print('P13 component hard-negative loss:', stc_criterion.describe_p13())
    print('P17 positive ranking loss:', stc_criterion.describe_p17())
    print('P22 target-frame balanced loss:', stc_criterion.describe_p22())

    optimizer = build_optimizer(net, cfg)
    if cfg.p2b_density_gdsca_enabled:
        print('P2b density-gate LR multiplier:', cfg.p2b_gate_lr_multiplier)
    if getattr(cfg, 'p11_local_activity_enabled', False):
        print('P11 activity-feature LR multiplier:', cfg.p11_feature_lr_multiplier)
    if getattr(cfg, 'p12_local_density_enabled', False):
        print('P12 density-feature LR multiplier:', cfg.p12_feature_lr_multiplier)
    scheduler = build_lr_scheduler(
        optimizer,
        cfg.scheduler,
        total_epochs=cfg.epochs,
        step_size=cfg.scheduler_step_size,
        gamma=cfg.scheduler_gamma,
        min_lr=cfg.scheduler_min_lr,
        cosine_t_max=cfg.scheduler_t_max,
    )
    print(
        'learning-rate scheduler:',
        describe_lr_scheduler(
            cfg.scheduler,
            cfg.epochs,
            cfg.scheduler_step_size,
            cfg.scheduler_gamma,
            cfg.scheduler_min_lr,
            cfg.scheduler_t_max,
        ),
    )

    # Resume from checkpoint
    if args.resume:
        start_epoch, best_loss, best_iou, best_score, best_score_metrics = load_full_checkpoint(
            resume_ckpt_path, net, optimizer, scheduler)
        start_epoch += 1  # resume from next epoch
        print('loaded epoch {}, best_loss={:.6f}, best_iou={}'.format(
            start_epoch - 1, best_loss, best_iou if best_iou else "N/A"))
        if best_score is not None:
            print('best_score={:.6f}'.format(best_score))

    best_loss = 1e5
    best_iou = None
    best_score = None
    best_score_metrics = None
    start_epoch = 0

    #for val
    val_dataset = EvUAV(cfg, mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,collate_fn=val_dataset.custom_collate)
    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    if threshold_policy.enabled and cfg.batch_size != 1:
        raise ValueError('P6 density-adaptive threshold requires batch_size=1.')
    postprocessor = ChallengePostprocessor.from_cfg(cfg, PREDICTION_THRESHOLD)
    print('validation postprocessor:', postprocessor.describe())
    print('validation threshold policy:', threshold_policy.describe(PREDICTION_THRESHOLD))

    # mlflow
    mlflow.set_experiment('train')
    mlflow.start_run(run_name=run_dir.name)
    mlflow.log_params({
        'run_directory': str(run_dir),
        'config_path': str(Path(cfg.config).resolve()),
        'seed': seed,
        'max_events_num': cfg.max_events_num,
        'scheduler': cfg.scheduler,
        'scheduler_step_size': cfg.scheduler_step_size,
        'scheduler_gamma': cfg.scheduler_gamma,
        'scheduler_min_lr': cfg.scheduler_min_lr,
        'scheduler_t_max': cfg.scheduler_t_max,
        'validation_start_epoch': cfg.validation_start_epoch,
        'checkpoint_interval': checkpoint_interval,
        'init_model_path': cfg.init_model_path,
        'initialized_from_checkpoint': initialized_from_checkpoint,
        'target_preserving_enabled': cfg.target_preserving_enabled,
        'density_dual_view_enabled': dataset.density_dual_view_enabled,
        'density_dual_view_event_count_cutoff': (
            dataset.density_dual_view_event_count_cutoff
        ),
        'density_dual_view_source_video_count': (
            dataset.density_dual_view_source_video_count
        ),
        'density_dual_view_extra_sample_count': (
            dataset.density_dual_view_extra_sample_count
        ),
        'dense_specialist_enabled': dataset.dense_specialist_enabled,
        'dense_specialist_event_count_cutoff': (
            dataset.dense_specialist_event_count_cutoff
        ),
        'dense_specialist_views_per_video': (
            dataset.dense_specialist_views_per_video
        ),
        'dense_specialist_target_preserving_enabled': (
            dataset.dense_specialist_target_preserving_enabled
        ),
        'dense_specialist_source_video_count': (
            dataset.dense_specialist_source_video_count
        ),
        'dense_specialist_sample_count': dataset.dense_specialist_sample_count,
        'config_overrides': ' '.join(cfg.config_overrides),
        'p1_hard_negative_enabled': stc_criterion.p1_hard_negative_enabled,
        'p1_hard_negative_weight': stc_criterion.p1_hard_negative_weight,
        'p1_hard_negative_ratio': stc_criterion.p1_hard_negative_ratio,
        'p1_hard_negative_warmup_epochs': stc_criterion.p1_hard_negative_warmup_epochs,
        'p2_positive_stc_floor_enabled': stc_criterion.p2_positive_stc_floor_enabled,
        'p2_positive_stc_floor': stc_criterion.p2_positive_stc_floor,
        'p4_target_frame_enabled': stc_criterion.p4_target_frame_enabled,
        'p4_target_frame_weight': stc_criterion.p4_target_frame_weight,
        'p4_target_frame_warmup_epochs': stc_criterion.p4_target_frame_warmup_epochs,
        'p13_component_hard_negative_enabled': (
            stc_criterion.p13_component_hard_negative_enabled
        ),
        'p13_component_hard_negative_weight': (
            stc_criterion.p13_component_hard_negative_weight
        ),
        'p13_target_frame_weight': stc_criterion.p13_target_frame_weight,
        'p13_component_hard_negative_ratio': (
            stc_criterion.p13_component_hard_negative_ratio
        ),
        'p13_component_hard_negative_warmup_epochs': (
            stc_criterion.p13_component_hard_negative_warmup_epochs
        ),
        'p13_spatial_cell_size': stc_criterion.p13_spatial_cell_size,
        'p13_temporal_bin_size': stc_criterion.p13_temporal_bin_size,
        'p13_min_cell_events': stc_criterion.p13_min_cell_events,
        'p13_activation_threshold': stc_criterion.p13_activation_threshold,
        'p13_activation_temperature': stc_criterion.p13_activation_temperature,
        'p17_positive_ranking_enabled': stc_criterion.p17_positive_ranking_enabled,
        'p17_positive_ranking_weight': stc_criterion.p17_positive_ranking_weight,
        'p17_positive_ranking_ratio': stc_criterion.p17_positive_ranking_ratio,
        'p17_positive_ranking_margin': stc_criterion.p17_positive_ranking_margin,
        'p17_positive_ranking_warmup_epochs': (
            stc_criterion.p17_positive_ranking_warmup_epochs
        ),
        'p22_target_frame_balanced_enabled': (
            stc_criterion.p22_target_frame_balanced_enabled
        ),
        'p22_target_frame_balanced_weight': (
            stc_criterion.p22_target_frame_balanced_weight
        ),
        'p22_target_frame_balanced_warmup_epochs': (
            stc_criterion.p22_target_frame_balanced_warmup_epochs
        ),
        'p22_temporal_bin_size': stc_criterion.p22_temporal_bin_size,
        'p2b_density_gdsca_enabled': cfg.p2b_density_gdsca_enabled,
        'p16_global_patch_attention_enabled': getattr(
            cfg,
            'p16_global_patch_attention_enabled',
            False,
        ),
        'p2b_gate_lr_multiplier': cfg.p2b_gate_lr_multiplier,
        'p11_local_activity_enabled': getattr(
            cfg,
            'p11_local_activity_enabled',
            False,
        ),
        'p11_local_activity_radius': getattr(
            cfg,
            'p11_local_activity_radius',
            50,
        ),
        'p11_feature_lr_multiplier': getattr(
            cfg,
            'p11_feature_lr_multiplier',
            1.0,
        ),
        'p12_local_density_enabled': getattr(
            cfg,
            'p12_local_density_enabled',
            False,
        ),
        'p12_spatial_cell_size': getattr(
            cfg,
            'p12_spatial_cell_size',
            3,
        ),
        'p12_temporal_cell_size': getattr(
            cfg,
            'p12_temporal_cell_size',
            50,
        ),
        'p12_neighborhood_radius': getattr(
            cfg,
            'p12_neighborhood_radius',
            1,
        ),
        'p12_feature_lr_multiplier': getattr(
            cfg,
            'p12_feature_lr_multiplier',
            1.0,
        ),
    })

    for epoch in range(start_epoch, cfg.epochs):
        stc_criterion.set_epoch(epoch)
        pbar = tqdm.tqdm(total=len(train_dataloader), unit="Batch", unit_scale=True,
                         desc="Epoch: {}".format(epoch),position=0,leave=True)

        for ev in train_dataloader:
            x = ev['voxel_ev']
            event_frame = ev.get('event_frame')
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            target_ids = None
            locations = None
            if stc_criterion.requires_target_ids:
                target_ids = torch.as_tensor(
                    ev['idx_label'],
                    dtype=torch.long,
                    device=label.device,
                )
            if stc_criterion.requires_locations:
                locations = ev['locs'].long().to(label.device)

            preds,voxel = net(x, event_frame=event_frame)

            loss = stc_criterion(
                voxel,
                p2v_map,
                preds,
                label,
                target_ids=target_ids,
                locations=locations,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            postfix = {'loss': loss_value}
            if stc_criterion.p13_component_hard_negative_active:
                postfix['p13'] = (
                    stc_criterion.p13_target_frame_weight
                    * stc_criterion.last_p13_target_frame_loss.item()
                    + stc_criterion.p13_component_hard_negative_weight
                    * stc_criterion.last_p13_component_hard_negative_loss.item()
                )
            if stc_criterion.p17_positive_ranking_active:
                postfix['p17'] = (
                    stc_criterion.p17_positive_ranking_weight
                    * stc_criterion.last_p17_positive_ranking_loss.item()
                )
            if stc_criterion.p22_target_frame_balanced_active:
                postfix['p22'] = (
                    stc_criterion.p22_target_frame_balanced_weight
                    * stc_criterion.last_p22_target_frame_balanced_loss.item()
                )
            pbar.set_postfix(**postfix)
            pbar.update(1)

            with torch.no_grad():
                mlflow.log_metric('loss', loss_value)
                if stc_criterion.p1_hard_negative_enabled:
                    mlflow.log_metric(
                        'p1_hard_negative_loss',
                        stc_criterion.last_p1_hard_negative_loss.item(),
                    )
                    mlflow.log_metric(
                        'p1_hard_negative_count',
                        stc_criterion.last_p1_hard_negative_count,
                    )
                if stc_criterion.p2_positive_stc_floor_enabled:
                    mlflow.log_metric(
                        'p2_boosted_positive_count',
                        stc_criterion.last_p2_boosted_positive_count,
                    )
                if stc_criterion.p4_target_frame_enabled:
                    mlflow.log_metric(
                        'p4_target_frame_loss',
                        stc_criterion.last_p4_target_frame_loss.item(),
                    )
                    mlflow.log_metric(
                        'p4_target_frame_count',
                        stc_criterion.last_p4_target_frame_count,
                    )
                    mlflow.log_metric(
                        'p4_missed_target_frame_count',
                        stc_criterion.last_p4_missed_target_frame_count,
                    )
                if stc_criterion.p13_component_hard_negative_enabled:
                    mlflow.log_metric(
                        'p13_target_frame_loss',
                        stc_criterion.last_p13_target_frame_loss.item(),
                    )
                    mlflow.log_metric(
                        'p13_target_frame_count',
                        stc_criterion.last_p13_target_frame_count,
                    )
                    mlflow.log_metric(
                        'p13_missed_target_frame_count',
                        stc_criterion.last_p13_missed_target_frame_count,
                    )
                    mlflow.log_metric(
                        'p13_component_hard_negative_loss',
                        stc_criterion.last_p13_component_hard_negative_loss.item(),
                    )
                    mlflow.log_metric(
                        'p13_candidate_cell_count',
                        stc_criterion.last_p13_candidate_cell_count,
                    )
                    mlflow.log_metric(
                        'p13_hard_cell_count',
                        stc_criterion.last_p13_hard_cell_count,
                    )
                if stc_criterion.p17_positive_ranking_enabled:
                    mlflow.log_metric(
                        'p17_positive_ranking_loss',
                        stc_criterion.last_p17_positive_ranking_loss.item(),
                    )
                    mlflow.log_metric(
                        'p17_positive_count',
                        stc_criterion.last_p17_positive_count,
                    )
                    mlflow.log_metric(
                        'p17_background_count',
                        stc_criterion.last_p17_background_count,
                    )
                if stc_criterion.p22_target_frame_balanced_enabled:
                    mlflow.log_metric(
                        'p22_target_frame_balanced_loss',
                        stc_criterion.last_p22_target_frame_balanced_loss.item(),
                    )
                    mlflow.log_metric(
                        'p22_target_frame_count',
                        stc_criterion.last_p22_target_frame_count,
                    )
                if loss_value < best_loss:
                    save_full_checkpoint(net, optimizer, scheduler, epoch, best_loss, best_iou, best_score, best_score_metrics, best_loss_path)
                    best_loss = loss_value
            torch.cuda.empty_cache()

        scheduler.step()
        save_full_checkpoint(net, optimizer, scheduler, epoch, best_loss, best_iou, best_score, best_score_metrics, last_path)
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            periodic_path = run_dir / 'epoch_{:03d}_seed{}.pt'.format(
                epoch,
                seed,
            )
            save_full_checkpoint(net, optimizer, scheduler, epoch, best_loss, best_iou, best_score, best_score_metrics, periodic_path)

        with torch.no_grad():
            if epoch >= cfg.validation_start_epoch:
                net.eval()
                evaluter = evalute(cfg)
                postprocess_stats = postprocessor.new_stats()
                validation_threshold_usage = {}
                sample_number = 0
                for ev in val_dataloader:
                    x = ev['voxel_ev']
                    event_frame = ev.get('event_frame')
                    p2v_map = ev['p2v_map'].long().cuda()

                    preds, voxel = net(x, event_frame=event_frame)
                    preds = preds[p2v_map].reshape(-1).cpu()
                    batch_threshold = threshold_policy.threshold_for_event_count(
                        preds.numel(),
                        PREDICTION_THRESHOLD,
                    )
                    batch_postprocessor = (
                        ChallengePostprocessor.from_cfg(cfg, batch_threshold)
                        if threshold_policy.enabled else postprocessor
                    )
                    preds, batch_postprocess_stats = batch_postprocessor.apply(
                        preds,
                        ev['locs'],
                    )
                    postprocess_stats.merge(batch_postprocess_stats)
                    if threshold_policy.enabled:
                        preds = (preds >= batch_threshold).to(preds.dtype)
                    validation_threshold_usage[batch_threshold] = (
                        validation_threshold_usage.get(batch_threshold, 0) + 1
                    )
                    sample_number = add_batch_to_evaluator(
                        evaluter,
                        ev,
                        preds,
                        sample_number,
                        batch_threshold,
                        collect_roc=cfg.roc,
                    )

                if cfg.roc:
                    metrics = evaluate_challenge_metrics(
                        evaluter,
                        PREDICTION_THRESHOLD,
                    )
                    iou = metrics.iou
                else:
                    metrics = None
                    iou = float(
                        evaluter.evaluate_semantic_segmantation_miou(
                            thresh=PREDICTION_THRESHOLD
                        ).item()
                    )

                if best_iou is None or iou > best_iou:
                    save_full_checkpoint(net, optimizer, scheduler, epoch, best_loss, best_iou, best_score, best_score_metrics, best_iou_path)
                    best_iou = iou
                mlflow.log_metric('val_iou', iou, step=epoch)

                if metrics is not None:
                    mlflow.log_metric('val_acc', metrics.acc, step=epoch)
                    mlflow.log_metric('val_pd', metrics.pd, step=epoch)
                    mlflow.log_metric('val_fa', metrics.fa, step=epoch)
                    mlflow.log_metric('val_score_fa', metrics.score_fa, step=epoch)
                    mlflow.log_metric('val_score', metrics.score, step=epoch)
                    if best_score is None or metrics.score > best_score:
                        save_full_checkpoint(net, optimizer, scheduler, epoch, best_loss, best_iou, best_score, best_score_metrics, best_score_path)
                        best_score = metrics.score
                        best_score_metrics = metrics.to_dict()
                    print(
                        'validation epoch {}: IoU={:.6f}, Acc={:.6f}, '
                        'Pd={:.6f}, Fa={:.6e}, Score={:.6f}'.format(
                            epoch,
                            metrics.iou,
                            metrics.acc,
                            metrics.pd,
                            metrics.fa,
                            metrics.score,
                        )
                    )
                print('validation postprocess result:', postprocess_stats.summary())
                if threshold_policy.enabled:
                    print(
                        'validation P6 threshold usage:',
                        ', '.join(
                            '{:.3f}: {} videos'.format(threshold, count)
                            for threshold, count in sorted(validation_threshold_usage.items())
                        ),
                    )
                net.train()

        write_run_summary(
            run_dir,
            started_at,
            seed,
            best_loss,
            best_iou,
            best_score,
            best_score_metrics,
            cfg.config_overrides,
        )

    mlflow.end_run()
    print('best loss checkpoint:', best_loss_path)
    if best_iou is not None:
        print('best IoU checkpoint:', best_iou_path)
    if best_score is not None:
        print('best Score checkpoint:', best_score_path)
    print('last checkpoint:', last_path)
