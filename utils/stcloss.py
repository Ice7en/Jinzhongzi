import torch
import torch.nn as nn
from torch.autograd import Function
import HAIS_OP
import numpy as np
import spconv.pytorch as spconv
from utils.component_hard_negative import (
    component_hard_negative_loss,
    target_frame_activation_loss,
)
from utils.hard_negative import top_ratio_background_bce
from utils.positive_reweight import (
    apply_positive_stc_floor,
    validate_positive_stc_floor,
)
from utils.target_frame_loss import target_frame_detection_loss
from utils.positive_ranking import positive_hard_ranking_loss
from utils.target_frame_balanced import target_frame_balanced_positive_loss


class STCLoss(nn.Module):
    def __init__(self, k, t, cfg,weight_clip_eps=1e-5):
        super(STCLoss, self).__init__()
        self.k = k
        self.t = t
        self.vol = self.k * self.k * self.t
        self.cfg = cfg

        self.stc_conv = spconv.SubMConv3d(1, 1, kernel_size=[self.k, self.k, self.t], stride=1,
                                          padding=[int(self.k / 2), int(self.k / 2), int(self.t / 2)], bias=False)

        weights = self.stc_conv.weight.data
        weights.fill_(1)
        self.stc_conv.requires_grad_(False)

        self.eps = weight_clip_eps
        self.p1_hard_negative_enabled = bool(
            getattr(cfg, 'p1_hard_negative_enabled', False)
        )
        self.p1_hard_negative_weight = float(
            getattr(cfg, 'p1_hard_negative_weight', 0.02)
        )
        self.p1_hard_negative_ratio = float(
            getattr(cfg, 'p1_hard_negative_ratio', 0.01)
        )
        self.p1_hard_negative_warmup_epochs = int(
            getattr(cfg, 'p1_hard_negative_warmup_epochs', 10)
        )
        self.p2_positive_stc_floor_enabled = bool(
            getattr(cfg, 'p2_positive_stc_floor_enabled', False)
        )
        self.p2_positive_stc_floor = validate_positive_stc_floor(
            getattr(cfg, 'p2_positive_stc_floor', 0.35)
        )
        self.p4_target_frame_enabled = bool(
            getattr(cfg, 'p4_target_frame_enabled', False)
        )
        self.p4_target_frame_weight = float(
            getattr(cfg, 'p4_target_frame_weight', 0.05)
        )
        self.p4_target_frame_warmup_epochs = int(
            getattr(cfg, 'p4_target_frame_warmup_epochs', 10)
        )
        self.p4_prediction_threshold = float(
            getattr(cfg, 'prediction_threshold', 0.9)
        )
        self.p4_correct_threshold = float(
            getattr(cfg, 'correct_thresh', 0.0001)
        )
        self.p4_temporal_bin_size = int(getattr(cfg, 'pd_detT', 50))
        self.p13_component_hard_negative_enabled = bool(
            getattr(cfg, 'p13_component_hard_negative_enabled', False)
        )
        self.p13_component_hard_negative_weight = float(
            getattr(cfg, 'p13_component_hard_negative_weight', 0.005)
        )
        self.p13_target_frame_weight = float(
            getattr(cfg, 'p13_target_frame_weight', 0.01)
        )
        self.p13_component_hard_negative_ratio = float(
            getattr(cfg, 'p13_component_hard_negative_ratio', 0.01)
        )
        self.p13_component_hard_negative_warmup_epochs = int(
            getattr(cfg, 'p13_component_hard_negative_warmup_epochs', 10)
        )
        self.p13_spatial_cell_size = int(
            getattr(cfg, 'p13_spatial_cell_size', 3)
        )
        self.p13_temporal_bin_size = int(
            getattr(cfg, 'p13_temporal_bin_size', 50)
        )
        self.p13_min_cell_events = int(
            getattr(cfg, 'p13_min_cell_events', 2)
        )
        self.p13_activation_threshold = float(
            getattr(cfg, 'p13_activation_threshold', 0.45)
        )
        self.p13_activation_temperature = float(
            getattr(cfg, 'p13_activation_temperature', 0.10)
        )
        self.p17_positive_ranking_enabled = bool(
            getattr(cfg, 'p17_positive_ranking_enabled', False)
        )
        self.p17_positive_ranking_weight = float(
            getattr(cfg, 'p17_positive_ranking_weight', 0.02)
        )
        self.p17_positive_ranking_ratio = float(
            getattr(cfg, 'p17_positive_ranking_ratio', 0.01)
        )
        self.p17_positive_ranking_margin = float(
            getattr(cfg, 'p17_positive_ranking_margin', 0.05)
        )
        self.p17_positive_ranking_warmup_epochs = int(
            getattr(cfg, 'p17_positive_ranking_warmup_epochs', 10)
        )
        self.p22_target_frame_balanced_enabled = bool(
            getattr(cfg, 'p22_target_frame_balanced_enabled', False)
        )
        self.p22_target_frame_balanced_weight = float(
            getattr(cfg, 'p22_target_frame_balanced_weight', 0.02)
        )
        self.p22_target_frame_balanced_warmup_epochs = int(
            getattr(cfg, 'p22_target_frame_balanced_warmup_epochs', 0)
        )
        self.p22_temporal_bin_size = int(
            getattr(cfg, 'p22_temporal_bin_size', getattr(cfg, 'pd_detT', 50))
        )
        if self.p1_hard_negative_weight < 0:
            raise ValueError('p1_hard_negative_weight must be non-negative.')
        if not 0 < self.p1_hard_negative_ratio <= 1:
            raise ValueError('p1_hard_negative_ratio must be in (0, 1].')
        if self.p1_hard_negative_warmup_epochs < 0:
            raise ValueError('p1_hard_negative_warmup_epochs must be non-negative.')
        if self.p4_target_frame_weight < 0:
            raise ValueError('p4_target_frame_weight must be non-negative.')
        if self.p4_target_frame_warmup_epochs < 0:
            raise ValueError('p4_target_frame_warmup_epochs must be non-negative.')
        if self.p13_component_hard_negative_weight < 0:
            raise ValueError('p13_component_hard_negative_weight must be non-negative.')
        if self.p13_target_frame_weight < 0:
            raise ValueError('p13_target_frame_weight must be non-negative.')
        if not 0 < self.p13_component_hard_negative_ratio <= 1:
            raise ValueError('p13_component_hard_negative_ratio must be in (0, 1].')
        if self.p13_component_hard_negative_warmup_epochs < 0:
            raise ValueError(
                'p13_component_hard_negative_warmup_epochs must be non-negative.'
            )
        if self.p13_spatial_cell_size <= 0:
            raise ValueError('p13_spatial_cell_size must be positive.')
        if self.p13_temporal_bin_size <= 0:
            raise ValueError('p13_temporal_bin_size must be positive.')
        if self.p13_min_cell_events <= 0:
            raise ValueError('p13_min_cell_events must be positive.')
        if not 0 < self.p13_activation_threshold < 1:
            raise ValueError('p13_activation_threshold must be in (0, 1).')
        if self.p13_activation_temperature <= 0:
            raise ValueError('p13_activation_temperature must be positive.')
        if self.p17_positive_ranking_weight < 0:
            raise ValueError('p17_positive_ranking_weight must be non-negative.')
        if not 0 < self.p17_positive_ranking_ratio <= 1:
            raise ValueError('p17_positive_ranking_ratio must be in (0, 1].')
        if self.p17_positive_ranking_margin < 0:
            raise ValueError('p17_positive_ranking_margin must be non-negative.')
        if self.p17_positive_ranking_warmup_epochs < 0:
            raise ValueError(
                'p17_positive_ranking_warmup_epochs must be non-negative.'
            )
        if self.p22_target_frame_balanced_weight < 0:
            raise ValueError('p22_target_frame_balanced_weight must be non-negative.')
        if self.p22_target_frame_balanced_warmup_epochs < 0:
            raise ValueError(
                'p22_target_frame_balanced_warmup_epochs must be non-negative.'
            )
        if self.p22_temporal_bin_size <= 0:
            raise ValueError('p22_temporal_bin_size must be positive.')

        self.current_epoch = 0
        self.last_stc_loss = None
        self.last_p1_hard_negative_loss = None
        self.last_p1_hard_negative_count = 0
        self.last_p2_boosted_positive_count = 0
        self.last_p4_target_frame_loss = None
        self.last_p4_target_frame_count = 0
        self.last_p4_missed_target_frame_count = 0
        self.last_p13_component_hard_negative_loss = None
        self.last_p13_candidate_cell_count = 0
        self.last_p13_hard_cell_count = 0
        self.last_p13_target_frame_loss = None
        self.last_p13_target_frame_count = 0
        self.last_p13_missed_target_frame_count = 0
        self.last_p17_positive_ranking_loss = None
        self.last_p17_positive_count = 0
        self.last_p17_background_count = 0
        self.last_p22_target_frame_balanced_loss = None
        self.last_p22_target_frame_count = 0

    @property
    def p1_hard_negative_active(self):
        return (
            self.p1_hard_negative_enabled
            and self.current_epoch >= self.p1_hard_negative_warmup_epochs
        )

    @property
    def p4_target_frame_active(self):
        return (
            self.p4_target_frame_enabled
            and self.current_epoch >= self.p4_target_frame_warmup_epochs
        )

    @property
    def p13_component_hard_negative_active(self):
        return (
            self.p13_component_hard_negative_enabled
            and self.current_epoch >= self.p13_component_hard_negative_warmup_epochs
        )

    @property
    def p13_target_frame_active(self):
        return (
            self.p13_component_hard_negative_active
            and self.p13_target_frame_weight > 0
        )

    @property
    def p13_background_component_active(self):
        return (
            self.p13_component_hard_negative_active
            and self.p13_component_hard_negative_weight > 0
        )

    @property
    def p17_positive_ranking_active(self):
        return (
            self.p17_positive_ranking_enabled
            and self.current_epoch >= self.p17_positive_ranking_warmup_epochs
            and self.p17_positive_ranking_weight > 0
        )

    @property
    def p22_target_frame_balanced_active(self):
        return (
            self.p22_target_frame_balanced_enabled
            and self.current_epoch >= self.p22_target_frame_balanced_warmup_epochs
            and self.p22_target_frame_balanced_weight > 0
        )

    @property
    def requires_target_ids(self):
        return (
            self.p4_target_frame_active
            or self.p13_target_frame_active
            or self.p22_target_frame_balanced_active
        )

    @property
    def requires_locations(self):
        return (
            self.p4_target_frame_active
            or self.p13_target_frame_active
            or self.p13_background_component_active
            or self.p22_target_frame_balanced_active
        )

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def describe_p1(self):
        if not self.p1_hard_negative_enabled:
            return 'disabled'
        return (
            'enabled (weight={}, top_ratio={}, warmup_epochs={})'.format(
                self.p1_hard_negative_weight,
                self.p1_hard_negative_ratio,
                self.p1_hard_negative_warmup_epochs,
            )
        )

    def describe_p2(self):
        if not self.p2_positive_stc_floor_enabled:
            return 'disabled'
        return 'enabled (positive_stc_floor={})'.format(
            self.p2_positive_stc_floor
        )

    def describe_p4(self):
        if not self.p4_target_frame_enabled:
            return 'disabled'
        return (
            'enabled (weight={}, warmup_epochs={}, threshold={}, '
            'correct_threshold={}, temporal_bin_size={})'
        ).format(
            self.p4_target_frame_weight,
            self.p4_target_frame_warmup_epochs,
            self.p4_prediction_threshold,
            self.p4_correct_threshold,
            self.p4_temporal_bin_size,
        )

    def describe_p13(self):
        if not self.p13_component_hard_negative_enabled:
            return 'disabled'
        return (
            'enabled (target_frame_weight={}, background_weight={}, '
            'top_cell_ratio={}, warmup_epochs={}, '
            'spatial_cell_size={}, temporal_bin_size={}, min_cell_events={}, '
            'activation_threshold={}, activation_temperature={})'
        ).format(
            self.p13_target_frame_weight,
            self.p13_component_hard_negative_weight,
            self.p13_component_hard_negative_ratio,
            self.p13_component_hard_negative_warmup_epochs,
            self.p13_spatial_cell_size,
            self.p13_temporal_bin_size,
            self.p13_min_cell_events,
            self.p13_activation_threshold,
            self.p13_activation_temperature,
        )

    def describe_p17(self):
        if not self.p17_positive_ranking_enabled:
            return 'disabled'
        return (
            'enabled (weight={}, ratio={}, margin={}, warmup_epochs={})'
        ).format(
            self.p17_positive_ranking_weight,
            self.p17_positive_ranking_ratio,
            self.p17_positive_ranking_margin,
            self.p17_positive_ranking_warmup_epochs,
        )

    def describe_p22(self):
        if not self.p22_target_frame_balanced_enabled:
            return 'disabled'
        return (
            'enabled (weight={}, warmup_epochs={}, temporal_bin_size={})'
        ).format(
            self.p22_target_frame_balanced_weight,
            self.p22_target_frame_balanced_warmup_epochs,
            self.p22_temporal_bin_size,
        )

    def forward(self, voxel, p2v_map, preds, label, target_ids=None, locations=None):
        stc_voxel = self.stc_conv(voxel)
        mean_stc = torch.mean(stc_voxel.features)
        stc_weights = torch.sigmoid(stc_voxel.features-mean_stc)
        stc_weights = stc_weights[p2v_map].squeeze().detach()
        preds = preds[p2v_map].squeeze()
        preds = torch.clamp(preds, 0, 1)

        pos_loss = -torch.log(preds + self.eps)
        neg_loss = -torch.log(1 - preds + self.eps)

        positive_stc_weights = stc_weights
        p2_boosted_positive_count = 0
        if self.p2_positive_stc_floor_enabled:
            positive_stc_weights, p2_boosted_positive_count = (
                apply_positive_stc_floor(
                    stc_weights,
                    label,
                    self.p2_positive_stc_floor,
                )
            )

        stc_loss = (
            (label * positive_stc_weights * pos_loss)
            + ((1 - label) * (1 - stc_weights) * neg_loss)
        ).mean()
        p1_hard_negative_loss = preds.sum() * 0
        p1_hard_negative_count = 0
        if self.p1_hard_negative_active:
            p1_hard_negative_loss, p1_hard_negative_count = top_ratio_background_bce(
                preds,
                label,
                self.p1_hard_negative_ratio,
                self.eps,
            )

        p4_target_frame_loss = preds.sum() * 0
        p4_target_frame_count = 0
        p4_missed_target_frame_count = 0
        if self.p4_target_frame_active:
            if target_ids is None or locations is None:
                raise ValueError(
                    'target_ids and locations are required when P4 is enabled.'
                )
            p4_target_frame_loss, p4_target_frame_count, p4_missed_target_frame_count = (
                target_frame_detection_loss(
                    preds,
                    label,
                    target_ids,
                    locations,
                    self.p4_prediction_threshold,
                    self.p4_correct_threshold,
                    self.p4_temporal_bin_size,
                )
            )

        p13_target_frame_loss = preds.sum() * 0
        p13_target_frame_count = 0
        p13_missed_target_frame_count = 0
        p13_component_hard_negative_loss = preds.sum() * 0
        p13_candidate_cell_count = 0
        p13_hard_cell_count = 0
        if self.p13_target_frame_active or self.p13_background_component_active:
            if locations is None:
                raise ValueError('locations are required when P13 is enabled.')
            if self.p13_target_frame_active:
                if target_ids is None:
                    raise ValueError('target_ids are required for the P13 target loss.')
                (
                    p13_target_frame_loss,
                    p13_target_frame_count,
                    p13_missed_target_frame_count,
                ) = target_frame_activation_loss(
                    preds,
                    label,
                    target_ids,
                    locations,
                    self.p13_temporal_bin_size,
                    self.p13_activation_threshold,
                    self.p13_activation_temperature,
                    self.eps,
                )
            if self.p13_background_component_active:
                (
                    p13_component_hard_negative_loss,
                    p13_candidate_cell_count,
                    p13_hard_cell_count,
                ) = component_hard_negative_loss(
                    preds,
                    label,
                    locations,
                    self.p13_spatial_cell_size,
                    self.p13_temporal_bin_size,
                    self.p13_min_cell_events,
                    self.p13_component_hard_negative_ratio,
                    self.p13_activation_threshold,
                    self.p13_activation_temperature,
                    self.eps,
                )

        p17_positive_ranking_loss = preds.sum() * 0
        p17_positive_count = 0
        p17_background_count = 0
        if self.p17_positive_ranking_active:
            (
                p17_positive_ranking_loss,
                p17_positive_count,
                p17_background_count,
            ) = positive_hard_ranking_loss(
                preds,
                label,
                self.p17_positive_ranking_ratio,
                self.p17_positive_ranking_margin,
            )

        p22_target_frame_balanced_loss = preds.sum() * 0
        p22_target_frame_count = 0
        if self.p22_target_frame_balanced_active:
            if target_ids is None or locations is None:
                raise ValueError(
                    'target_ids and locations are required when P22 is enabled.'
                )
            (
                p22_target_frame_balanced_loss,
                p22_target_frame_count,
            ) = target_frame_balanced_positive_loss(
                preds,
                label,
                target_ids,
                locations,
                self.p22_temporal_bin_size,
                self.eps,
            )

        self.last_stc_loss = stc_loss.detach()
        self.last_p1_hard_negative_loss = p1_hard_negative_loss.detach()
        self.last_p1_hard_negative_count = p1_hard_negative_count
        self.last_p2_boosted_positive_count = p2_boosted_positive_count
        self.last_p4_target_frame_loss = p4_target_frame_loss.detach()
        self.last_p4_target_frame_count = p4_target_frame_count
        self.last_p4_missed_target_frame_count = p4_missed_target_frame_count
        self.last_p13_component_hard_negative_loss = (
            p13_component_hard_negative_loss.detach()
        )
        self.last_p13_candidate_cell_count = p13_candidate_cell_count
        self.last_p13_hard_cell_count = p13_hard_cell_count
        self.last_p13_target_frame_loss = p13_target_frame_loss.detach()
        self.last_p13_target_frame_count = p13_target_frame_count
        self.last_p13_missed_target_frame_count = p13_missed_target_frame_count
        self.last_p17_positive_ranking_loss = p17_positive_ranking_loss.detach()
        self.last_p17_positive_count = p17_positive_count
        self.last_p17_background_count = p17_background_count
        self.last_p22_target_frame_balanced_loss = (
            p22_target_frame_balanced_loss.detach()
        )
        self.last_p22_target_frame_count = p22_target_frame_count
        return (
            stc_loss
            + self.p1_hard_negative_weight * p1_hard_negative_loss
            + self.p4_target_frame_weight * p4_target_frame_loss
            + self.p13_component_hard_negative_weight
            * p13_component_hard_negative_loss
            + self.p13_target_frame_weight * p13_target_frame_loss
            + self.p17_positive_ranking_weight * p17_positive_ranking_loss
            + self.p22_target_frame_balanced_weight
            * p22_target_frame_balanced_loss
        )
