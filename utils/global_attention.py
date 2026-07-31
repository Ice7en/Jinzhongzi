"""Batch-safe global token attention for sparse feature tensors."""

import torch


def apply_batchwise_global_attention(attention, features, batch_indices):
    """Apply ``MultiheadAttention`` across tokens inside each sparse sample.

    PyTorch's default ``MultiheadAttention`` layout is ``[tokens, batch,
    channels]``. Sparse tensors store active voxels in one flat feature array,
    so each batch member must be attended separately to prevent tokens from
    different videos from interacting.
    """
    if features.ndim != 2:
        raise ValueError('features must have shape [N, C].')
    if batch_indices.ndim != 1 or batch_indices.numel() != features.shape[0]:
        raise ValueError('batch_indices must have shape [N].')
    if features.shape[0] == 0:
        return features

    attended_features = torch.empty_like(features)
    for batch_index in torch.unique(batch_indices, sorted=True):
        token_indices = torch.nonzero(
            batch_indices == batch_index,
            as_tuple=False,
        ).reshape(-1)
        tokens = features.index_select(0, token_indices).unsqueeze(1)
        attended_tokens, _ = attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        attended_features = attended_features.index_copy(
            0,
            token_indices,
            attended_tokens.squeeze(1),
        )

    return attended_features
