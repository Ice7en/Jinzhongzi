import argparse
from copy import deepcopy
import yaml
import os


def apply_overrides(config, overrides):
    """Apply typed SECTION.KEY=value command-line overrides to a YAML mapping."""
    resolved = deepcopy(config)

    for override in overrides:
        if '=' not in override:
            raise ValueError(
                'Invalid --set value {!r}. Expected SECTION.KEY=value.'.format(override)
            )

        option_path, raw_value = override.split('=', 1)
        option_keys = option_path.split('.')
        if len(option_keys) < 2 or any(not key for key in option_keys):
            raise ValueError(
                'Invalid --set path {!r}. Expected SECTION.KEY=value.'.format(option_path)
            )

        target = resolved
        for key in option_keys[:-1]:
            if not isinstance(target, dict) or key not in target:
                raise KeyError('Unknown configuration section in --set: {}'.format(option_path))
            target = target[key]

        option_key = option_keys[-1]
        if not isinstance(target, dict) or option_key not in target:
            raise KeyError('Unknown configuration option in --set: {}'.format(option_path))

        target[option_key] = yaml.safe_load(raw_value)

    return resolved


def get_parser():
    parser = argparse.ArgumentParser(description='Event Point Cloud Segmentation')
    parser.add_argument('--config', default='/home/a/chennuo/EV-UAV-main/configs/evisseg_evuav.yaml',type=str, help='path to config file')
    parser.add_argument(
        '--set',
        dest='overrides',
        nargs='+',
        default=[],
        metavar='SECTION.KEY=VALUE',
        help='override YAML values, for example: --set POSTPROCESS.p0_enabled=true',
    )

    args_cfg, unknown = parser.parse_known_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    if not isinstance(config, dict):
        raise TypeError('The configuration root must be a YAML mapping.')

    config = apply_overrides(config, args_cfg.overrides)
    for key in config:
        if not isinstance(config[key], dict):
            raise TypeError('Configuration section {} must be a mapping.'.format(key))
        for k, v in config[key].items():
            setattr(args_cfg, k, v)
    args_cfg.resolved_config = config
    args_cfg.config_overrides = list(args_cfg.overrides)
    return args_cfg

cfg = get_parser()
