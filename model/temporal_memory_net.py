"""Bidirectional temporal memory on top of the P23 full-frame backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.temporal_frame_net import TemporalFrameNet


class TemporalSelfAttentionMemory(nn.Module):
    """Temporal self-attention memory module.

    Core ideas:
    - Self-attention along the time dimension for every spatial position
    - Any two frames can interact in one step, no recurrent propagation needed
    - Spatial pooling to save memory
    - Zero-initialized residual: initial output is 0, preserving original predictions
    """

    def __init__(self, channels, num_heads=4, pool_size=16):
        super().__init__()
        self.pool_size = pool_size
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            batch_first=True,
        )
        self.output_projection = nn.Linear(channels, channels)

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, bottlenecks):
        """Args:
            bottlenecks: Temporal feature sequence [B, T, C, H, W]

        Returns:
            torch.Tensor: Residual features [B, T, C, H, W]
        """
        B, T, C, H, W = bottlenecks.shape

        pooled = bottlenecks
        h, w = H, W
        if self.pool_size and (H > self.pool_size or W > self.pool_size):
            flat = pooled.reshape(B * T, C, H, W)
            flat = F.adaptive_avg_pool2d(flat, (self.pool_size, self.pool_size))
            h, w = self.pool_size, self.pool_size
            pooled = flat.reshape(B, T, C, h, w)

        tokens = pooled.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, C)

        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)

        residual = self.output_projection(attended)
        residual = residual.reshape(B, h, w, T, C).permute(0, 3, 4, 1, 2)

        if (h, w) != (H, W):
            flat_r = residual.reshape(B * T, C, h, w)
            flat_r = F.interpolate(
                flat_r, size=(H, W), mode='bilinear', align_corners=False,
            )
            residual = flat_r.reshape(B, T, C, H, W)

        return residual


class ConvGRUCell(nn.Module):
    """A compact spatial ConvGRU cell used only at the U-Net bottleneck."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError('channels must be positive.')
        self.channels = channels
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, inputs, state=None):
        if inputs.ndim != 4:
            raise ValueError('inputs must have shape [B, C, H, W].')
        if inputs.shape[1] != self.channels:
            raise ValueError('Unexpected ConvGRU input channels.')
        if state is None:
            state = torch.zeros_like(inputs)
        if state.shape != inputs.shape:
            raise ValueError('ConvGRU state shape does not match inputs.')
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, state), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * state), dim=1))
        )
        return (1.0 - update) * state + update * candidate


class BidirectionalTemporalMemoryNet(nn.Module):
    """P23 U-Net with a zero-initialized bidirectional temporal residual.

    Every temporal step first receives the original P23 local context stack.
    A pair of ConvGRU cells then propagates low-resolution evidence forward
    and backward through a sequence.  The residual projection is initialized
    to zero, allowing a P23 checkpoint to be loaded without changing its
    initial predictions before memory training begins.
    """

    def __init__(
        self,
        input_channels,
        width=16,
        temporal_attention_enabled=False,
        density_calibration_enabled=False,
        confidence_head_enabled=False,
    ):
        super().__init__()
        self.base = TemporalFrameNet(
            input_channels=int(input_channels),
            width=int(width),
            density_calibration_enabled=bool(density_calibration_enabled),
            confidence_head_enabled=bool(confidence_head_enabled),
        )
        self.confidence_head_enabled = bool(confidence_head_enabled)
        bottleneck_channels = int(width) * 6
        self.forward_memory = ConvGRUCell(bottleneck_channels)
        self.backward_memory = ConvGRUCell(bottleneck_channels)
        self.memory_projection = nn.Conv2d(
            bottleneck_channels * 2,
            bottleneck_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)

        self.temporal_attention_enabled = bool(temporal_attention_enabled)
        self.temporal_attn = None
        if self.temporal_attention_enabled:
            self.temporal_attn = TemporalSelfAttentionMemory(
                channels=bottleneck_channels,
                num_heads=4,
                pool_size=16,
            )

    @property
    def input_channels(self):
        return self.base.input_channels

    def _encode(self, frames):
        if frames.ndim != 4:
            raise ValueError('frames must have shape [B, C, H, W].')
        if frames.shape[1] != self.input_channels:
            raise ValueError(
                'frames have {} channels, expected {}.'.format(
                    frames.shape[1], self.input_channels
                )
            )
        level0 = self.base.encoder0(frames)
        level1 = self.base.encoder1(level0)
        level2 = self.base.encoder2(level1)
        bottleneck = self.base.context(self.base.encoder3(level2))
        return level0, level1, level2, bottleneck

    def encode_bottleneck(self, frames):
        """Encode a frame batch for full-stream inference memory passes."""
        return self._encode(frames)[-1]

    def _memory_residual(self, bottlenecks):
        if bottlenecks.ndim != 5:
            raise ValueError('bottlenecks must have shape [B, T, C, H, W].')
        batch_size, sequence_length = bottlenecks.shape[:2]
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')

        forward_states = []
        state = None
        for time_index in range(sequence_length):
            state = self.forward_memory(bottlenecks[:, time_index], state)
            forward_states.append(state)

        backward_states = [None] * sequence_length
        state = None
        for time_index in range(sequence_length - 1, -1, -1):
            state = self.backward_memory(bottlenecks[:, time_index], state)
            backward_states[time_index] = state

        memory_features = torch.cat(
            (
                torch.stack(forward_states, dim=1),
                torch.stack(backward_states, dim=1),
            ),
            dim=2,
        )
        flat_features = memory_features.reshape(
            batch_size * sequence_length,
            memory_features.shape[2],
            memory_features.shape[3],
            memory_features.shape[4],
        )
        projected = self.memory_projection(flat_features)
        residual = projected.reshape(
            batch_size,
            sequence_length,
            projected.shape[1],
            projected.shape[2],
            projected.shape[3],
        )

        if self.temporal_attention_enabled:
            attn_residual = self.temporal_attn(bottlenecks)
            residual = residual + attn_residual

        return residual

    def temporal_residual(self, bottlenecks):
        """Return one zero-initialized temporal residual per bottleneck map."""
        if bottlenecks.ndim == 4:
            return self._memory_residual(bottlenecks.unsqueeze(0)).squeeze(0)
        return self._memory_residual(bottlenecks)

    def _decode(
        self,
        level0,
        level1,
        level2,
        bottleneck,
        base_input=None,
        return_confidence_logits=False,
    ):
        decoded2 = self.base.decoder2(bottleneck, level2)
        decoded1 = self.base.decoder1(decoded2, level1)
        decoded0 = self.base.decoder0(decoded1, level0)
        if self.base.density_calibration_enabled:
            if base_input is None:
                raise ValueError(
                    'density calibration requires base_input in _decode.'
                )
            decoded0 = self.base.density_calibrator(decoded0, base_input)
        logits = self.base.head(decoded0)
        if self.confidence_head_enabled and return_confidence_logits:
            confidence_logits = self.base.confidence_head(decoded0)
            return logits, confidence_logits
        return logits

    def decode_with_residual(
        self,
        frames,
        residual,
        return_confidence_logits=False,
    ):
        """Decode a frame batch after a full-stream memory pass."""
        level0, level1, level2, bottleneck = self._encode(frames)
        if residual.shape != bottleneck.shape:
            raise ValueError('Temporal residual does not match bottleneck shape.')
        return self._decode(
            level0,
            level1,
            level2,
            bottleneck + residual,
            base_input=frames[:, :self.input_channels],
            return_confidence_logits=return_confidence_logits,
        )

    def forward(self, frames):
        """Predict logit maps for ``[B, T, C, H, W]`` temporal sequences."""
        if frames.ndim != 5:
            raise ValueError('frames must have shape [B, T, C, H, W].')
        batch_size, sequence_length = frames.shape[:2]
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')
        flat_frames = frames.reshape(
            batch_size * sequence_length,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )
        level0, level1, level2, bottleneck = self._encode(flat_frames)
        bottleneck = bottleneck.reshape(
            batch_size,
            sequence_length,
            bottleneck.shape[1],
            bottleneck.shape[2],
            bottleneck.shape[3],
        )
        residual = self._memory_residual(bottleneck).reshape_as(
            bottleneck.reshape(
                batch_size * sequence_length,
                bottleneck.shape[2],
                bottleneck.shape[3],
                bottleneck.shape[4],
            )
        )
        decode_output = self._decode(
            level0,
            level1,
            level2,
            bottleneck.reshape(
                batch_size * sequence_length,
                bottleneck.shape[2],
                bottleneck.shape[3],
                bottleneck.shape[4],
            ) + residual,
            base_input=flat_frames[:, :self.input_channels],
            return_confidence_logits=self.confidence_head_enabled,
        )
        if self.confidence_head_enabled:
            logits, confidence_logits = decode_output
            logits = logits.reshape(
                batch_size,
                sequence_length,
                logits.shape[1],
                logits.shape[2],
                logits.shape[3],
            )
            confidence_logits = confidence_logits.reshape(
                batch_size,
                sequence_length,
                confidence_logits.shape[1],
                confidence_logits.shape[2],
                confidence_logits.shape[3],
            )
            return logits, confidence_logits
        logits = decode_output
        return logits.reshape(
            batch_size,
            sequence_length,
            logits.shape[1],
            logits.shape[2],
            logits.shape[3],
        )
