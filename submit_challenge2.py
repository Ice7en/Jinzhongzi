"""Generate Challenge 2 prediction text files from a YAML configuration."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.density_threshold import DensityAdaptiveThresholdConfig
from utils.ensemble import ChallengePredictor
from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
)
from utils.postprocess import ChallengePostprocessor
from utils.spatial_tta import HorizontalFlipTTAConfig
from utils.temporal_frame_inference import (
    TemporalFrameInferenceConfig,
    blend_temporal_frame_scores,
    load_temporal_frame_model,
    predict_temporal_frame_scores,
    temporal_frame_video_from_sample,
)
from utils.temporal_memory_inference import (
    TemporalMemoryInferenceConfig,
    load_temporal_memory_model,
    predict_temporal_memory_scores,
)
from utils.tta_inference import predict_sample_scores


OUTPUT_DIR = Path(cfg.challenge_output_dir)
PREDICTION_THRESHOLD = float(cfg.prediction_threshold)


def save_prediction(source_path, output_path, prediction):
    """Save one video's predictions in the official x y t p label format."""
    with np.load(source_path) as data:
        source_events = data["ev"]

        if len(source_events) != len(prediction):
            raise ValueError(
                f"{source_path.name}: event count {len(source_events)} does not "
                f"match prediction count {len(prediction)}"
            )

        output_events = np.empty(
            len(source_events),
            dtype=[
                ("x", source_events.dtype["x"]),
                ("y", source_events.dtype["y"]),
                ("t", source_events.dtype["t"]),
                ("p", source_events.dtype["p"]),
                ("label", np.int64),
            ],
        )
        for field in ("x", "y", "t", "p"):
            output_events[field] = source_events[field]
        output_events["label"] = prediction

    np.savetxt(
        output_path,
        output_events,
        fmt=["%d", "%d", "%.9f", "%d", "%d"],
        delimiter=" ",
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Challenge 2 inference.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    chunk_config = InferenceChunkConfig.from_cfg(cfg)
    tta_config = HorizontalFlipTTAConfig.from_cfg(cfg)
    temporal_frame_config = TemporalFrameInferenceConfig.from_cfg(cfg)
    temporal_memory_config = TemporalMemoryInferenceConfig.from_cfg(cfg)
    if temporal_frame_config.enabled and temporal_memory_config.enabled:
        raise ValueError(
            'TEMPORAL_FRAME and TEMPORAL_MEMORY cannot be enabled together.'
        )
    fine_detail_bin_ratio = 1
    if temporal_frame_config.fine_detail_enabled:
        if (
            temporal_frame_config.fine_temporal_bin_size
            > int(cfg.temporal_frame_bin_size)
            or int(cfg.temporal_frame_bin_size)
            % temporal_frame_config.fine_temporal_bin_size != 0
        ):
            raise ValueError(
                'TEMPORAL_FRAME.fine_temporal_bin_size must be a positive '
                'divisor no greater than temporal_frame_bin_size.'
            )
        fine_detail_bin_ratio = (
            int(cfg.temporal_frame_bin_size)
            // temporal_frame_config.fine_temporal_bin_size
        )
    temporal_frame_only = temporal_frame_config.frame_only
    temporal_memory_only = temporal_memory_config.enabled
    full_stream_only = temporal_frame_only or temporal_memory_only
    predictor = None
    if not full_stream_only:
        predictor = ChallengePredictor(cfg, device, evspsegnet)
    temporal_frame_model = None
    if temporal_frame_config.enabled:
        temporal_frame_model, _ = load_temporal_frame_model(
            temporal_frame_config.model_path,
            device,
            cfg.temporal_frame_context_bins,
            cfg.temporal_frame_width,
            temporal_frame_config.local_contrast_enabled,
            temporal_frame_config.local_contrast_kernel_size,
            temporal_frame_config.motion_persistence_enabled,
            temporal_frame_config.motion_persistence_radius_per_bin,
            temporal_frame_config.fine_detail_enabled,
            temporal_frame_config.fine_temporal_bin_size,
            temporal_frame_config.fine_context_bins,
            temporal_frame_config.target_center_enabled,
            temporal_frame_config.confidence_head_enabled,
            temporal_frame_config.density_calibration_enabled,
        )
    temporal_memory_model = None
    if temporal_memory_config.enabled:
        if int(cfg.temporal_memory_context_bins) % 2 == 0:
            raise ValueError('TEMPORAL_MEMORY.context_bins must be odd.')
        if int(cfg.temporal_memory_sequence_length) <= 1:
            raise ValueError(
                'TEMPORAL_MEMORY.sequence_length must exceed one.'
            )
        temporal_memory_model, _ = load_temporal_memory_model(
            temporal_memory_config.model_path,
            device,
            cfg.temporal_memory_context_bins,
            cfg.temporal_memory_width,
            cfg.temporal_memory_sequence_length,
        )
    if threshold_policy.enabled and cfg.batch_size != 1:
        raise ValueError("P6 density-adaptive threshold requires batch_size=1.")
    if not full_stream_only and chunk_config.enabled and cfg.batch_size != 1:
        raise ValueError("P8 random chunk inference requires batch_size=1.")
    if (
        not full_stream_only
        and chunk_config.enabled
        and getattr(cfg, "p3_lite_enabled", False)
    ):
        raise ValueError("P8 random chunk inference does not support P3-Lite event frames.")
    if not full_stream_only and tta_config.enabled and cfg.batch_size != 1:
        raise ValueError("P14 horizontal-flip TTA requires batch_size=1.")
    if predictor is not None and predictor.dense_expert_config.enabled and cfg.batch_size != 1:
        raise ValueError("P20 dense-expert inference requires batch_size=1.")
    if temporal_frame_config.enabled and cfg.batch_size != 1:
        raise ValueError(
            "The temporal-frame expert requires batch_size=1."
        )
    if temporal_memory_config.enabled and cfg.batch_size != 1:
        raise ValueError(
            "The temporal-memory expert requires batch_size=1."
        )
    if (
        not full_stream_only
        and tta_config.enabled
        and getattr(cfg, "p3_lite_enabled", False)
    ):
        raise ValueError("P14 horizontal-flip TTA does not support P3-Lite event frames.")
    if predictor is None:
        print("dict load: skipped (full-stream-only inference)")
        print("model ensemble: skipped (full-stream-only inference)")
    else:
        print("dict load:", predictor.primary_model_path)
        print("model ensemble:", predictor.describe())
    print("validation root:", Path(cfg.root) / "val")
    print("prediction threshold:", PREDICTION_THRESHOLD)
    print("threshold policy:", threshold_policy.describe(PREDICTION_THRESHOLD))
    if full_stream_only:
        print("P8 random chunk inference: skipped (full-stream-only inference)")
        print("P14 horizontal-flip TTA: skipped (full-stream-only inference)")
    else:
        print("P8 random chunk inference:", chunk_config.describe())
        print("P14 horizontal-flip TTA:", tta_config.describe())
    print("temporal-frame expert:", temporal_frame_config.describe())
    print("temporal-memory expert:", temporal_memory_config.describe())
    print("prediction output:", OUTPUT_DIR)
    postprocessor = ChallengePostprocessor.from_cfg(cfg, PREDICTION_THRESHOLD)
    postprocess_stats = postprocessor.new_stats()
    threshold_usage = {}
    print("postprocessor:", postprocessor.describe())

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError(f"No validation files found in: {dataset.root}")

    dataloader = None
    sample_level_inference = (
        chunk_config.enabled
        or tta_config.enabled
        or temporal_frame_config.enabled
        or temporal_memory_config.enabled
    )
    if not sample_level_inference:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.custom_collate,
            shuffle=False,
        )
    pbar = tqdm.tqdm(
        total=len(dataset) if sample_level_inference else len(dataloader),
        desc="video",
        unit="video",
        unit_scale=True,
        position=0,
        leave=True,
    )

    sample_number = 0
    p8_partitioned_videos = 0
    p8_chunk_count = 0
    if sample_level_inference:
        for video_index in range(len(dataset)):
            sample = dataset[video_index]
            event_count = len(sample["ev_loc"])
            locations = evaluation_batch_from_sample(sample)["locs"]
            frame_video = None
            fine_detail_video = None
            if temporal_frame_config.enabled or temporal_memory_config.enabled:
                frame_video = temporal_frame_video_from_sample(
                    sample,
                    (
                        cfg.temporal_memory_bin_size
                        if temporal_memory_config.enabled
                        else cfg.temporal_frame_bin_size
                    ),
                    cfg.whole_t,
                )
                if temporal_frame_config.fine_detail_enabled:
                    fine_detail_video = temporal_frame_video_from_sample(
                        sample,
                        temporal_frame_config.fine_temporal_bin_size,
                        cfg.whole_t,
                    )
            if temporal_memory_only:
                predictions = predict_temporal_memory_scores(
                    temporal_memory_model,
                    frame_video,
                    device,
                    cfg.temporal_memory_context_bins,
                    cfg.res[0],
                    cfg.res[1],
                    cfg.temporal_memory_inference_batch_size,
                    cfg.temporal_memory_log_count_clip,
                )
                chunk_count = 0
            elif temporal_frame_only:
                predictions = predict_temporal_frame_scores(
                    temporal_frame_model,
                    frame_video,
                    device,
                    cfg.temporal_frame_context_bins,
                    cfg.res[0],
                    cfg.res[1],
                    cfg.temporal_frame_inference_batch_size,
                    cfg.temporal_frame_log_count_clip,
                    temporal_frame_config.local_contrast_enabled,
                    temporal_frame_config.local_contrast_kernel_size,
                    temporal_frame_config.motion_persistence_enabled,
                    temporal_frame_config.motion_persistence_radius_per_bin,
                    temporal_frame_config.fine_detail_enabled,
                    fine_detail_video,
                    temporal_frame_config.fine_context_bins,
                    fine_detail_bin_ratio,
                )
                chunk_count = 0
            else:
                predictions, chunk_count = predict_sample_scores(
                    predictor,
                    dataset,
                    sample,
                    device,
                    chunk_config,
                    tta_config,
                )
                if temporal_frame_config.enabled:
                    frame_scores = predict_temporal_frame_scores(
                        temporal_frame_model,
                        frame_video,
                        device,
                        cfg.temporal_frame_context_bins,
                        cfg.res[0],
                        cfg.res[1],
                        cfg.temporal_frame_inference_batch_size,
                        cfg.temporal_frame_log_count_clip,
                        temporal_frame_config.local_contrast_enabled,
                        temporal_frame_config.local_contrast_kernel_size,
                        temporal_frame_config.motion_persistence_enabled,
                        temporal_frame_config.motion_persistence_radius_per_bin,
                        temporal_frame_config.fine_detail_enabled,
                        fine_detail_video,
                        temporal_frame_config.fine_context_bins,
                        fine_detail_bin_ratio,
                    )
                    predictions = blend_temporal_frame_scores(
                        predictions,
                        frame_scores,
                        temporal_frame_config.sparse_weight,
                    )
            if not full_stream_only and chunk_config.should_partition(event_count):
                p8_partitioned_videos += 1
                p8_chunk_count += chunk_count
            batch_threshold = threshold_policy.threshold_for_event_count(
                event_count,
                PREDICTION_THRESHOLD,
            )
            batch_postprocessor = (
                ChallengePostprocessor.from_cfg(cfg, batch_threshold)
                if threshold_policy.enabled else postprocessor
            )
            predictions, batch_postprocess_stats = batch_postprocessor.apply(
                predictions,
                locations,
            )
            postprocess_stats.merge(batch_postprocess_stats)
            threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1

            source_path = Path(dataset.root) / dataset.file_list[video_index]
            output_path = OUTPUT_DIR / f"{source_path.stem}.txt"
            output_prediction = (predictions >= batch_threshold).to(torch.int64).numpy()
            save_prediction(source_path, output_path, output_prediction)
            pbar.update(1)
    else:
        for batch in dataloader:
            with torch.no_grad():
                p2v_map = batch["p2v_map"].long().to(device)
                locations = batch["locs"]
                batch_ids = locations[:, 0].long()
                predictions = predictor.predict_event_scores(
                    batch["voxel_ev"],
                    p2v_map,
                    event_frame=batch.get("event_frame"),
                    source_event_count=batch["locs"].shape[0],
                )
                batch_threshold = threshold_policy.threshold_for_event_count(
                    predictions.numel(),
                    PREDICTION_THRESHOLD,
                )
                batch_postprocessor = (
                    ChallengePostprocessor.from_cfg(cfg, batch_threshold)
                    if threshold_policy.enabled else postprocessor
                )
                predictions, batch_postprocess_stats = batch_postprocessor.apply(
                    predictions,
                    locations,
                )
                postprocess_stats.merge(batch_postprocess_stats)
                threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1

                for local_index in batch_ids.unique(sorted=True).tolist():
                    sample_mask = batch_ids == local_index
                    source_path = Path(dataset.root) / dataset.file_list[sample_number]
                    output_path = OUTPUT_DIR / f"{source_path.stem}.txt"
                    output_prediction = (
                        predictions[sample_mask] >= batch_threshold
                    ).to(torch.int64).numpy()
                    save_prediction(source_path, output_path, output_prediction)
                    sample_number += 1

            pbar.update(1)

    pbar.close()
    print("postprocess result:", postprocess_stats.summary())
    if chunk_config.enabled and not full_stream_only:
        print(
            "P8 random chunk result: {} high-density videos, {} chunk forwards".format(
                p8_partitioned_videos,
                p8_chunk_count,
            )
        )
    if threshold_policy.enabled:
        print(
            "P6 threshold usage:",
            ", ".join(
                "{:.3f}: {} videos".format(threshold, count)
                for threshold, count in sorted(threshold_usage.items())
            ),
        )
    print(f"prediction txt files saved to: {OUTPUT_DIR}")
