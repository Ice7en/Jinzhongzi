"""Strict checkpoint loading helpers for optional architecture extensions."""


def _is_p2b_auxiliary_key(name):
    parts = str(name).split('.')
    return 'density_conv' in parts or 'density_gate' in parts


def _is_p11_auxiliary_key(name):
    return str(name).startswith('p11_activity_projection.')


def _is_p12_auxiliary_key(name):
    return str(name).startswith('p12_density_projection.')


def load_state_dict_with_optional_compatibility(
    model,
    state_dict,
    p2b_enabled=False,
    p11_enabled=False,
    p12_enabled=False,
):
    """Load a checkpoint while allowing neutral optional modules to be absent.

    A legacy checkpoint can initialize P2b because its new gates start as an
    exact identity, and can initialize P11/P12 because their residuals start
    at zero. Every non-optional mismatch remains an error so
    incompatible checkpoints cannot be loaded silently.
    """
    if not p2b_enabled and not p11_enabled and not p12_enabled:
        model.load_state_dict(state_dict, strict=True)
        return ()

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    expected_auxiliary_keys = set()
    if p2b_enabled:
        expected_auxiliary_keys.update(
            name for name in model.state_dict() if _is_p2b_auxiliary_key(name)
        )
    if p11_enabled:
        expected_auxiliary_keys.update(
            name for name in model.state_dict() if _is_p11_auxiliary_key(name)
        )
    if p12_enabled:
        expected_auxiliary_keys.update(
            name for name in model.state_dict() if _is_p12_auxiliary_key(name)
        )
    if unexpected_keys or missing_keys not in (set(), expected_auxiliary_keys):
        raise RuntimeError(
            'Incompatible checkpoint for optional-module model. Missing keys: {}; '
            'unexpected keys: {}.'.format(
                sorted(missing_keys),
                sorted(unexpected_keys),
            )
        )
    return tuple(sorted(missing_keys))


def load_state_dict_with_p2b_compatibility(model, state_dict, p2b_enabled):
    """Backward-compatible P2b-only wrapper."""
    return load_state_dict_with_optional_compatibility(
        model,
        state_dict,
        p2b_enabled=p2b_enabled,
        p11_enabled=False,
        p12_enabled=False,
    )
