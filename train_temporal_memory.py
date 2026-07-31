"""Train a bidirectional full-stream temporal-memory event segmentation model."""

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
from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    temporal_memory_collate,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.temporal_frame_loss import frame_balanced_event_bce


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
    raise ValueError('Unsupported scheduler: {}'.format(config.scheduler))


def load_p23_base_weights(
    model,
    checkpoint_path,
    context_bins,
    width,
    density_calibration_enabled=False,
):
    checkpoint_path = Path(str(checkpoint_path).strip())
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'P23 initialization checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved = checkpoint.get('temporal_frame', {})
    if saved.get('context_bins') is not None and int(
        saved['context_bins']
    ) != int(context_bins):
        raise ValueError(
            'P23 context_bins={} does not match {}.'.format(
                saved['context_bins'], context_bins
            )
        )
    if saved.get('width') is not None and int(saved['width']) != int(width):
        raise ValueError(
            'P23 width={} does not match {}.'.format(saved['width'], width)
        )
    # A pure-P23 checkpoint has no density-calibrator keys; leave the
    # calibrator at its safe identity init (Sigmoid(4) ~ 1.0) instead.
    model.base.load_state_dict(
        checkpoint['model_state_dict'],
        strict=not bool(density_calibration_enabled),
    )
    return checkpoint_path


def build_optimizer(model, config):
    base_multiplier = float(config.temporal_memory_base_lr_multiplier)
    memory_multiplier = float(config.temporal_memory_memory_lr_multiplier)
    if base_multiplier <= 0.0 or memory_multiplier <= 0.0:
        raise ValueError('Temporal-memory learning-rate multipliers must be positive.')
    memory_parameters = list(model.forward_memory.parameters())
    memory_parameters += list(model.backward_memory.parameters())
    memory_parameters += list(model.memory_projection.parameters())
    return optim.AdamW(
        [
            {
                'params': model.base.parameters(),
                'lr': float(config.lr) * base_multiplier,
            },
            {
                'params': memory_parameters,
                'lr': float(config.lr) * memory_multiplier,
            },
        ],
        weight_decay=1e-4,
    )


def memory_config_summary(config):
    return (
        'enabled (bin_size={}, context_bins={}, width={}, sequence_length={}, '
        'views_per_video={}, positive_frame_probability={}, '
        'target_positive_loss_mass={}, max_positive_weight={}, '
        'base_lr_multiplier={}, memory_lr_multiplier={})'
    ).format(
        config.temporal_memory_bin_size,
        config.temporal_memory_context_bins,
        config.temporal_memory_width,
        config.temporal_memory_sequence_length,
        config.temporal_memory_train_views_per_video,
        config.temporal_memory_positive_frame_probability,
        config.temporal_memory_target_positive_loss_mass,
        config.temporal_memory_max_positive_weight,
        config.temporal_memory_base_lr_multiplier,
        config.temporal_memory_memory_lr_multiplier,
    )


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for temporal-memory training.')
    if not bool(cfg.temporal_memory_enabled):
        raise ValueError('Set TEMPORAL_MEMORY.temporal_memory_enabled=true.')
    if int(cfg.temporal_memory_context_bins) % 2 == 0:
        raise ValueError('TEMPORAL_MEMORY.context_bins must be odd.')
    if int(cfg.temporal_memory_sequence_length) <= 1:
        raise ValueError('TEMPORAL_MEMORY.sequence_length must exceed one.')
    if int(cfg.temporal_memory_train_workers) != 0 and bool(
        cfg.temporal_memory_cache_all_videos
    ):
        raise ValueError(
            'Use TEMPORAL_MEMORY.train_workers=0 when cache_all_videos=true.'
        )
    if int(cfg.epochs) <= 0:
        raise ValueError('TRAIN.epochs must be positive.')

    setup_seed(cfg.seed)
    device = torch.device('cuda:0')
    run_dir, started_at = create_run_directory(cfg)
    dataset = TemporalMemoryTrainDataset(
        root=Path(cfg.root) / 'train',
        whole_t=cfg.whole_t,
        temporal_bin_size=cfg.temporal_memory_bin_size,
        context_bins=cfg.temporal_memory_context_bins,
        sequence_length=cfg.temporal_memory_sequence_length,
        width=cfg.res[0],
        height=cfg.res[1],
        views_per_video=cfg.temporal_memory_train_views_per_video,
        positive_frame_probability=cfg.temporal_memory_positive_frame_probability,
        random_seed=cfg.seed,
        log_count_clip=cfg.temporal_memory_log_count_clip,
        cache_all_videos=cfg.temporal_memory_cache_all_videos,
        cache_video_count=cfg.temporal_memory_cache_video_count,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.temporal_memory_train_workers),
        collate_fn=temporal_memory_collate,
        pin_memory=True,
    )
    density_calibration_enabled = bool(
        getattr(cfg, 'temporal_frame_density_calibration_enabled', False)
    )
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(cfg.temporal_memory_context_bins) * 2,
        width=int(cfg.temporal_memory_width),
        density_calibration_enabled=density_calibration_enabled,
    ).to(device)
    initialized_from = load_p23_base_weights(
        model,
        cfg.temporal_memory_init_model_path,
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        density_calibration_enabled=density_calibration_enabled,
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    print('random seed:{}'.format(cfg.seed))
    print('run directory:', run_dir)
    print('config overrides:', ', '.join(cfg.config_overrides) or '(none)')
    print('temporal-memory model:', memory_config_summary(cfg))
    print('training videos:', len(dataset.file_paths))
    print('training sequences per epoch:', len(dataset))
    print('initialized P23 base weights from:', initialized_from)
    print('learning-rate scheduler:', cfg.scheduler)

    best_loss = float('inf')
    best_epoch = None
    for epoch in range(int(cfg.epochs)):
        dataset.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        positive_fraction_sum = 0.0
        positive_weight_sum = 0.0
        batch_count = 0
        pbar = tqdm.tqdm(
            dataloader,
            desc='Epoch: {}'.format(epoch),
            unit='Sequence',
            position=0,
            leave=True,
        )
        for batch in pbar:
            frames = batch['frames'].to(device, non_blocking=True).unsqueeze(0)
            event_time_indices = batch['event_time_indices'].to(
                device,
                non_blocking=True,
            )
            event_y = batch['event_y'].to(device, non_blocking=True)
            event_x = batch['event_x'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logit_maps = model(frames).squeeze(0)
            event_logits = logit_maps[
                event_time_indices,
                0,
                event_y,
                event_x,
            ]
            loss, diagnostics = frame_balanced_event_bce(
                event_logits,
                labels,
                event_time_indices,
                target_positive_loss_mass=(
                    cfg.temporal_memory_target_positive_loss_mass
                ),
                max_positive_weight=cfg.temporal_memory_max_positive_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.detach().item())
            positive_fraction_sum += diagnostics['positive_fraction']
            positive_weight_sum += diagnostics['mean_positive_weight']
            batch_count += 1
            pbar.set_postfix(
                loss='{:.5f}'.format(loss_sum / batch_count),
                pos='{:.4f}'.format(positive_fraction_sum / batch_count),
                pos_w='{:.2f}'.format(positive_weight_sum / batch_count),
            )
        pbar.close()
        scheduler.step()

        epoch_loss = loss_sum / max(batch_count, 1)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'temporal_memory': {
                'temporal_bin_size': int(cfg.temporal_memory_bin_size),
                'context_bins': int(cfg.temporal_memory_context_bins),
                'width': int(cfg.temporal_memory_width),
                'sequence_length': int(cfg.temporal_memory_sequence_length),
                'log_count_clip': float(cfg.temporal_memory_log_count_clip),
                'density_calibration_enabled': bool(
                    getattr(
                        cfg,
                        'temporal_frame_density_calibration_enabled',
                        False,
                    )
                ),
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
            checkpoint, run_dir / 'last_seed{}.pt'.format(cfg.seed)
        )
        print(
            'epoch {}: loss={:.6f}, lr_base={:.8f}, lr_memory={:.8f}, '
            'best_loss={:.6f}'.format(
                epoch,
                epoch_loss,
                optimizer.param_groups[0]['lr'],
                optimizer.param_groups[1]['lr'],
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
        'config_overrides': list(cfg.config_overrides),
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print('best loss checkpoint:', summary['best_loss_checkpoint'])
    print('last checkpoint:', summary['last_checkpoint'])
