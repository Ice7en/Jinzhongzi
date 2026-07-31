"""Shared Challenge 2 evaluation helpers for training and standalone validation."""

from dataclasses import asdict, dataclass
import math


SCORE_FA_SCALE = 10000.0


@dataclass(frozen=True)
class ChallengeMetrics:
    iou: float
    acc: float
    pd: float
    fa: float
    score_fa: float
    score: float

    def to_dict(self):
        return asdict(self)


def challenge_score(iou, acc, pd, fa):
    """Compute the official Challenge 2 Score_Fa and Score values."""
    score_fa = math.exp(-SCORE_FA_SCALE * fa)
    score = 0.4 * pd + 0.3 * score_fa + 0.2 * iou + 0.1 * acc
    return score_fa, score


def add_batch_to_evaluator(
    evaluator,
    batch,
    predictions,
    sample_number,
    prediction_threshold,
    collect_roc=True,
):
    """Add each collated video to the evaluator without mixing batch members."""
    labels = batch['seg_label'].float()
    locations = batch['locs']
    target_ids = batch['idx_label']
    batch_ids = locations[:, 0].long()

    if not (predictions.numel() == labels.numel() == locations.shape[0]):
        raise RuntimeError(
            'Prediction, label, and event-location counts do not match: '
            '{}, {}, {}'.format(
                predictions.numel(), labels.numel(), locations.shape[0]
            )
        )

    for local_index in batch_ids.unique(sorted=True).tolist():
        sample_mask = batch_ids == local_index
        sample_mask_np = sample_mask.detach().cpu().numpy()
        sample_predictions = predictions[sample_mask]
        sample_labels = labels[sample_mask]
        sample_locations = locations[sample_mask]
        sample_target_ids = target_ids[sample_mask_np]

        evaluator.matches[str(sample_number)] = {
            'seg_pred': sample_predictions,
            'seg_gt': sample_labels,
        }
        if collect_roc:
            evaluator.roc_update(
                sample_locations[:, 3],
                sample_predictions.clone(),
                sample_target_ids,
                sample_labels,
                sample_locations,
                thresh=prediction_threshold,
            )
        sample_number += 1

    return sample_number


def evaluate_challenge_metrics(evaluator, prediction_threshold):
    """Return all official Challenge 2 metrics from a populated evaluator."""
    iou = float(
        evaluator.evaluate_semantic_segmantation_miou(
            thresh=prediction_threshold
        ).item()
    )
    acc = float(
        evaluator.evaluate_semantic_segmantation_accuracy(
            thresh=prediction_threshold
        ).item()
    )
    pd, fa = evaluator.cal_roc()
    pd = float(pd)
    fa = float(fa)

    if not all(math.isfinite(value) for value in (iou, acc, pd, fa)):
        raise RuntimeError('A non-finite metric was produced; check the validation data.')

    score_fa, score = challenge_score(iou, acc, pd, fa)
    return ChallengeMetrics(
        iou=iou,
        acc=acc,
        pd=pd,
        fa=fa,
        score_fa=score_fa,
        score=score,
    )
