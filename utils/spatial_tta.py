"""Label-free spatial transforms used by test-time augmentation."""

from dataclasses import dataclass

import numpy as np


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError("Expected a boolean value, got {!r}.".format(value))
    return bool(value)


@dataclass(frozen=True)
class HorizontalFlipTTAConfig:
    """Optional probability averaging over original and mirrored event streams."""

    enabled: bool = False
    original_weight: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.original_weight <= 1.0:
            raise ValueError("p14_horizontal_flip_original_weight must be in [0, 1].")

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, "p14_horizontal_flip_enabled", False)),
            original_weight=float(
                getattr(cfg, "p14_horizontal_flip_original_weight", 1.0)
            ),
        )

    def describe(self):
        if not self.enabled:
            return "disabled"
        return "enabled (original_weight={:.3f}, flipped_weight={:.3f})".format(
            self.original_weight,
            1.0 - self.original_weight,
        )


def padded_feature_width(image_width, alignment=32):
    """Return the x-axis padding used by the sparse input tensor."""
    image_width = int(image_width)
    alignment = int(alignment)
    if image_width <= 0 or alignment <= 0:
        raise ValueError("image_width and alignment must be positive.")
    return ((image_width + alignment - 1) // alignment) * alignment


def horizontal_flip_event_inputs(ev_loc, evs_norm, image_width, feature_width=None):
    """Mirror event coordinates and their normalized x input feature."""
    image_width = int(image_width)
    if image_width <= 0:
        raise ValueError("image_width must be positive.")
    if feature_width is None:
        feature_width = padded_feature_width(image_width)
    feature_width = int(feature_width)
    if feature_width < image_width:
        raise ValueError("feature_width must be at least image_width.")

    ev_loc = np.asarray(ev_loc)
    evs_norm = np.asarray(evs_norm)
    if ev_loc.ndim != 2 or ev_loc.shape[1] < 3:
        raise ValueError("ev_loc must have shape [N, 3+].")
    if (
        evs_norm.ndim != 2
        or evs_norm.shape[0] != ev_loc.shape[0]
        or evs_norm.shape[1] < 1
    ):
        raise ValueError("evs_norm must align with ev_loc and contain normalized x.")
    if ev_loc.size and (
        ev_loc[:, 0].min() < 0 or ev_loc[:, 0].max() >= image_width
    ):
        raise ValueError("Event x coordinates are outside the configured image width.")

    mirrored_locations = ev_loc.copy()
    mirrored_locations[:, 0] = image_width - 1 - mirrored_locations[:, 0]
    mirrored_features = evs_norm.copy()
    mirrored_features[:, 0] = mirrored_locations[:, 0] / float(feature_width)
    return mirrored_locations, mirrored_features


def horizontal_flip_sample(sample, image_width, feature_width=None):
    """Mirror x coordinates and their normalized feature without labels.

    ``evs_norm[:, 0]`` is the x coordinate normalized by the sparse tensor's
    padded width.  Labels and target ids are copied unchanged because they are
    not model inputs and the transform preserves event order.
    """
    if "event_frame" in sample:
        raise ValueError("Horizontal flip TTA does not support P3-Lite event frames.")

    mirrored_locations, mirrored_features = horizontal_flip_event_inputs(
        sample["ev_loc"],
        sample["evs_norm"],
        image_width,
        feature_width,
    )

    return {
        "ev_loc": mirrored_locations,
        "evs_norm": mirrored_features,
        "seg_label": np.asarray(sample["seg_label"]).copy(),
        "idx": np.asarray(sample["idx"]).copy(),
    }
