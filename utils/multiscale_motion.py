"""
Multi-scale motion persistence feature module.
Builds a temporal feature pyramid to adapt to targets with different speeds.
"""

import torch
import torch.nn.functional as F


def build_multiscale_motion_persistence_channels(
    inputs,
    context_bins,
    spatial_radii=(1, 2, 4, 8),
):
    """Generate multi-scale motion persistence features.

    Args:
        inputs (torch.Tensor): Input features [B, T*C, H, W]
        context_bins (int): Number of temporal bins
        spatial_radii (tuple): Multi-scale spatial radii, e.g. (1, 2, 4, 8)

    Returns:
        torch.Tensor: Multi-scale features [B, len(spatial_radii)*(context_bins-1), H, W]
    """
    if inputs.ndim != 4:
        raise ValueError('inputs must have shape [B, C, H, W].')
    context_bins = int(context_bins)
    if context_bins < 1 or context_bins % 2 == 0:
        raise ValueError('context_bins must be a positive odd integer.')
    if inputs.shape[1] != context_bins * 2:
        raise ValueError(
            'inputs have {} channels, expected {} raw temporal channels.'.format(
                inputs.shape[1], context_bins * 2,
            )
        )

    centre_bin = context_bins // 2
    centre_start = centre_bin * 2
    centre_activity = inputs[:, centre_start:centre_start + 2].sum(
        dim=1, keepdim=True,
    )

    all_scales = []
    for base_radius in spatial_radii:
        scale_feats = []
        for t in range(context_bins):
            if t == centre_bin:
                continue
            ns = t * 2
            neighbour = inputs[:, ns:ns + 2].sum(dim=1, keepdim=True)

            radius = base_radius * abs(t - centre_bin)
            if radius > 0:
                neighbour = F.max_pool2d(
                    neighbour,
                    kernel_size=radius * 2 + 1,
                    stride=1,
                    padding=radius,
                )

            scale_feats.append(torch.minimum(centre_activity, neighbour))

        all_scales.append(torch.cat(scale_feats, dim=1))

    return torch.cat(all_scales, dim=1)


def multiscale_motion_channel_count(context_bins, spatial_radii=(1, 2, 4, 8)):
    """Return the number of output channels for a given configuration."""
    return len(spatial_radii) * (int(context_bins) - 1)
