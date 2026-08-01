"""Configurable learning-rate schedules for reproducible training runs."""

import torch


def build_lr_scheduler(
    optimizer,
    scheduler_name,
    total_epochs,
    step_size=10,
    gamma=0.1,
    min_lr=1e-6,
    cosine_t_max=None,
):
    """Build a scheduler while keeping the original StepLR as the default."""
    scheduler_name = str(scheduler_name).strip().lower()
    total_epochs = int(total_epochs)
    step_size = int(step_size)
    gamma = float(gamma)
    min_lr = float(min_lr)

    if total_epochs <= 0:
        raise ValueError('total_epochs must be positive.')
    if min_lr < 0:
        raise ValueError('min_lr must be non-negative.')

    if scheduler_name == 'step':
        if step_size <= 0:
            raise ValueError('step_size must be positive for StepLR.')
        if gamma <= 0:
            raise ValueError('gamma must be positive for StepLR.')
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    if scheduler_name == 'cosine':
        if cosine_t_max is None:
            cosine_t_max = total_epochs
        else:
            cosine_t_max = int(cosine_t_max)
        if cosine_t_max <= 0:
            raise ValueError('cosine_t_max must be positive when provided.')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=min_lr,
        )

    raise ValueError(
        "Unknown TRAIN.scheduler {!r}; expected 'step' or 'cosine'.".format(
            scheduler_name
        )
    )


def describe_lr_scheduler(
    scheduler_name,
    total_epochs,
    step_size,
    gamma,
    min_lr,
    cosine_t_max=None,
):
    """Return the resolved schedule in a compact training-log form."""
    scheduler_name = str(scheduler_name).strip().lower()
    if scheduler_name == 'step':
        return 'step (step_size={}, gamma={})'.format(step_size, gamma)
    if scheduler_name == 'cosine':
        if cosine_t_max is None:
            cosine_t_max = total_epochs
        if int(cosine_t_max) != int(total_epochs):
            return 'cosine (epochs={}, T_max={}, min_lr={})'.format(
                total_epochs,
                cosine_t_max,
                min_lr,
            )
        return 'cosine (epochs={}, min_lr={})'.format(total_epochs, min_lr)
    return scheduler_name
