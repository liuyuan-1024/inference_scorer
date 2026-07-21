"""基于真实时间戳的胸前 / 腕部多视角视频证据提取。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from domain.models import TaskEvidence, VisionEvidence
from domain.rules_v5 import V5_RULES
from inputs.video import (
    DecodedFrames,
    decode_selected_frames,
    load_camera_frame_timestamps,
    select_frame_indices,
)
from inputs.segments import TaskSegmentAssignments, resolve_task_segments
from evidence.vision_cv import (
    box_relation,
    distance,
    red_mask,
    track_motion_norm,
    track_observations,
    yellow_mask,
)

__all__ = [
    "analyze_chest_frames",
    "analyze_wrist_frames",
    "attach_vision_evidence",
]


def _event_index(
    timestamps: np.ndarray,
    event_timestamp: float | None,
    *,
    fallback_fraction: float | None,
    n_frames: int,
) -> int:
    if (
        event_timestamp is not None
        and len(timestamps)
        and np.isfinite(timestamps).any()
    ):
        valid = np.where(np.isfinite(timestamps))[0]
        return int(valid[np.argmin(np.abs(timestamps[valid] - event_timestamp))])
    if fallback_fraction is not None:
        return max(0, min(n_frames - 1, round(fallback_fraction * (n_frames - 1))))
    return 0


def analyze_chest_frames(
    decoded: DecodedFrames,
    *,
    grasp_timestamp: float | None = None,
    release_timestamp: float | None = None,
    grasp_fraction: float | None = None,
) -> VisionEvidence:
    frames = decoded.frames
    if len(frames) == 0:
        return VisionEvidence(notes=["胸前相机未读取到视频帧"])

    objects = track_observations(frames, red_mask, V5_RULES.vision_min_object_pixels)
    boxes = track_observations(frames, yellow_mask, V5_RULES.vision_min_box_pixels)
    n = len(frames)
    early_n = max(1, min(n, max(3, n // 4)))
    object_rate = sum(item is not None for item in objects[:early_n]) / early_n
    box_rate = sum(item is not None for item in boxes[:early_n]) / early_n
    object_present = object_rate >= V5_RULES.vision_present_rate
    box_present = box_rate >= V5_RULES.vision_present_rate
    if object_present and box_present:
        scenario = "normal"
    elif object_present:
        scenario = "object_without_box"
    elif box_present:
        scenario = "box_without_object"
    else:
        scenario = "empty_or_unrecognized"

    diagonal = math.hypot(frames.shape[2], frames.shape[1])
    grasp_idx = _event_index(
        decoded.timestamps,
        grasp_timestamp,
        fallback_fraction=grasp_fraction,
        n_frames=n,
    )
    release_idx = _event_index(
        decoded.timestamps,
        release_timestamp,
        fallback_fraction=None,
        n_frames=n,
    )
    valid_indices = [i for i, item in enumerate(objects) if item is not None]
    object_motion = track_motion_norm(objects, valid_indices, diagonal)
    post_indices = [i for i in valid_indices if i >= grasp_idx]
    motion_after_grasp = track_motion_norm(objects, post_indices, diagonal)

    distance_drop = 0.0
    initial_distances = [
        distance(objects[i].centroid, boxes[i].centroid)
        for i in range(early_n)
        if objects[i] is not None and boxes[i] is not None
    ]
    post_distances = [
        distance(objects[i].centroid, boxes[i].centroid)
        for i in range(grasp_idx, n)
        if objects[i] is not None and boxes[i] is not None
    ]
    if initial_distances and post_distances:
        distance_drop = (
            max(0.0, float(np.median(initial_distances)) - min(post_distances))
            / diagonal
        )

    final_start = (
        release_idx if release_timestamp is not None else max(0, n - max(3, n // 5))
    )
    final_candidates: list[tuple[str, float, int]] = []
    for i in range(final_start, n):
        relation, ratio = box_relation(objects[i], boxes[i])
        if relation != "unknown":
            final_candidates.append((relation, ratio, i))
    if final_candidates:
        final_relation, final_ratio, _ = final_candidates[-1]
    else:
        final_relation, final_ratio = "unknown", 0.0

    settle_indices = [
        i
        for i in range(final_start, n)
        if objects[i] is not None
        and (
            release_timestamp is None
            or not np.isfinite(decoded.timestamps[i])
            or (
                release_timestamp + V5_RULES.vision_settle_release_delay_sec
                <= decoded.timestamps[i]
                <= release_timestamp
                + V5_RULES.vision_settle_release_delay_sec
                + V5_RULES.vision_settle_window_sec
            )
        )
    ]
    settle_motion = track_motion_norm(objects, settle_indices, diagonal)
    settled = (
        len(settle_indices) >= V5_RULES.vision_settle_min_frames
        and settle_motion <= V5_RULES.vision_settle_motion_norm
    )

    alignment_error = float("inf")
    event_times = [
        value for value in (grasp_timestamp, release_timestamp) if value is not None
    ]
    valid_times = decoded.timestamps[np.isfinite(decoded.timestamps)]
    if event_times and len(valid_times):
        alignment_error = max(
            float(np.min(np.abs(valid_times - event_time)))
            for event_time in event_times
        )

    confidence = 0.45
    confidence += 0.18 * min(1.0, max(object_rate, box_rate))
    confidence += 0.12 if scenario != "empty_or_unrecognized" else 0.0
    confidence += 0.10 if n >= 20 else 0.0
    confidence += 0.08 if alignment_error <= 0.05 else 0.0
    confidence = min(0.90, confidence)
    notes = [
        f"胸前场景={scenario}",
        f"红色玩具检出率={object_rate:.0%}",
        f"黄色盒子检出率={box_rate:.0%}",
    ]
    if final_relation != "unknown":
        notes.append(f"释放后玩具位置={final_relation}(盒内覆盖={final_ratio:.0%})")

    return VisionEvidence(
        available=True,
        camera="cam_mid_chest",
        cameras=["cam_mid_chest"],
        chest_available=True,
        sample_count=n,
        scenario=scenario,
        object_present=object_present,
        box_present=box_present,
        object_detection_rate=object_rate,
        box_detection_rate=box_rate,
        object_motion_norm=object_motion,
        object_motion_after_grasp_norm=motion_after_grasp,
        object_box_distance_drop_norm=distance_drop,
        moved_toward_box=distance_drop >= V5_RULES.vision_toward_box_norm,
        final_object_relation=final_relation,
        final_inside_ratio=final_ratio,
        object_settled_after_release=settled,
        temporal_alignment_sec=alignment_error,
        confidence=confidence,
        notes=notes,
    )


def analyze_wrist_frames(
    decoded: DecodedFrames,
    *,
    grasp_timestamp: float | None,
    release_timestamp: float | None,
) -> VisionEvidence:
    frames = decoded.frames
    if len(frames) == 0:
        return VisionEvidence(notes=["腕部相机未读取到视频帧"])

    objects = track_observations(frames, red_mask, V5_RULES.vision_min_object_pixels)
    n = len(frames)
    diagonal = math.hypot(frames.shape[2], frames.shape[1])
    anchor = (
        frames.shape[2] * V5_RULES.vision_wrist_gripper_x_norm,
        frames.shape[1] * V5_RULES.vision_wrist_gripper_y_norm,
    )
    distances = np.array(
        [
            distance(item.centroid, anchor) / diagonal if item is not None else np.nan
            for item in objects
        ],
        dtype=float,
    )
    has_grasp_event = grasp_timestamp is not None
    grasp_idx = (
        _event_index(
            decoded.timestamps,
            grasp_timestamp,
            fallback_fraction=None,
            n_frames=n,
        )
        if has_grasp_event
        else n - 1
    )
    release_idx = (
        _event_index(
            decoded.timestamps,
            release_timestamp,
            fallback_fraction=None,
            n_frames=n,
        )
        if release_timestamp is not None
        else n - 1
    )
    release_idx = min(n, max(grasp_idx + 1, release_idx))
    pre = distances[: max(1, grasp_idx + 1)]
    valid_pre = pre[np.isfinite(pre)]
    if len(valid_pre):
        initial_distance = float(np.median(valid_pre[: max(1, len(valid_pre) // 4)]))
        min_distance = float(np.min(valid_pre))
        approach_drop = max(0.0, initial_distance - min_distance)
    else:
        min_distance = float("inf")
        approach_drop = 0.0

    # 释放帧及其紧邻窗口不参与掉落判定，避免把正常张开误判为提前掉落。
    hold_end = release_idx if release_timestamp is not None else n
    held = distances[grasp_idx:hold_end]
    held_times = decoded.timestamps[grasp_idx:hold_end]
    observed = np.isfinite(held)
    near = observed & (held <= V5_RULES.vision_wrist_near_gripper_norm)
    observation_rate = float(observed.mean()) if len(observed) else 0.0
    near_rate = float(near.sum() / observed.sum()) if observed.any() else 0.0

    if (
        has_grasp_event
        and grasp_timestamp is not None
        and np.isfinite(held_times).any()
    ):
        acquire_window = (
            np.isfinite(held_times)
            & (held_times >= grasp_timestamp)
            & (held_times <= grasp_timestamp + V5_RULES.vision_wrist_acquire_window_sec)
        )
    elif has_grasp_event:
        acquire_window = np.zeros(len(held), dtype=bool)
        acquire_window[: min(5, len(held))] = True
    else:
        acquire_window = np.zeros(len(held), dtype=bool)
    acquire_observed = observed & acquire_window
    acquire_rate = (
        float((near & acquire_window).sum() / acquire_observed.sum())
        if acquire_observed.any()
        else 0.0
    )
    object_near = bool(
        has_grasp_event
        and acquire_observed.sum() >= 2
        and acquire_rate >= V5_RULES.vision_wrist_acquire_rate
    )

    early_window = acquire_window
    early_observed = observed & early_window
    early_hold_rate = (
        float((near & early_window).sum() / early_observed.sum())
        if early_observed.any()
        else 0.0
    )
    late_count = max(3, len(held) // 3)
    late_window = np.zeros(len(held), dtype=bool)
    late_window[-min(late_count, len(held)) :] = True
    late_observed = observed & late_window
    late_hold_rate = (
        float((near & late_window).sum() / late_observed.sum())
        if late_observed.any()
        else 0.0
    )
    object_retained = bool(
        object_near
        and observed.sum() >= 3
        and near_rate >= V5_RULES.vision_wrist_retain_rate
        and late_hold_rate >= V5_RULES.vision_wrist_retain_rate
    )

    drop_window = observed.copy()
    if (
        release_timestamp is not None
        and len(held_times)
        and np.isfinite(held_times).any()
    ):
        drop_window &= ~np.isfinite(held_times) | (
            held_times <= release_timestamp - V5_RULES.vision_wrist_release_guard_sec
        )
    far = drop_window & ~near
    near_indices = np.flatnonzero(near & acquire_window)
    far_run = 0
    if len(near_indices) >= 2:
        current_run = 0
        for is_far in far[near_indices[1] + 1 :]:
            current_run = current_run + 1 if is_far else 0
            far_run = max(far_run, current_run)
    late_drop_observed = drop_window & late_window
    late_far_rate = (
        float((far & late_window).sum() / late_drop_observed.sum())
        if late_drop_observed.any()
        else 0.0
    )
    drop_detected = bool(
        object_near
        and far_run >= V5_RULES.vision_wrist_drop_far_min_frames
        and late_far_rate >= V5_RULES.vision_wrist_drop_far_rate
    )

    detected_rate = sum(item is not None for item in objects) / n
    confidence = 0.42 + 0.28 * min(1.0, detected_rate)
    if grasp_timestamp is not None and np.isfinite(decoded.timestamps).any():
        confidence += 0.10
    if n >= 20:
        confidence += 0.08
    confidence = min(0.88, confidence)
    notes = [
        f"腕部玩具检出率={detected_rate:.0%}",
        f"夹爪获取率={acquire_rate:.0%}",
        f"夹持前段/后段保持率={early_hold_rate:.0%}/{late_hold_rate:.0%}",
    ]
    if drop_detected:
        notes.append("腕部视角检测到释放前脱离夹爪")

    return VisionEvidence(
        available=True,
        camera="cam_left_wrist",
        cameras=["cam_left_wrist"],
        wrist_available=True,
        sample_count=n,
        object_detection_rate=detected_rate,
        wrist_object_near_gripper=object_near,
        wrist_object_retained=object_retained,
        wrist_drop_detected=drop_detected,
        wrist_observation_rate=observation_rate,
        wrist_acquire_rate=acquire_rate,
        wrist_early_hold_rate=early_hold_rate,
        wrist_late_hold_rate=late_hold_rate,
        wrist_approach_drop_norm=approach_drop,
        wrist_min_object_gripper_norm=min_distance,
        confidence=confidence,
        notes=notes,
    )


def _merge_evidence(chest: VisionEvidence, wrist: VisionEvidence) -> VisionEvidence:
    if not chest.available and not wrist.available:
        return VisionEvidence(notes=chest.notes + wrist.notes)
    if chest.available:
        merged = chest
    else:
        merged = VisionEvidence(
            available=True,
            camera=wrist.camera,
            scenario="unknown",
            object_present=None,
            box_present=None,
        )
    if wrist.available:
        merged.wrist_available = True
        merged.wrist_object_near_gripper = wrist.wrist_object_near_gripper
        merged.wrist_object_retained = wrist.wrist_object_retained
        merged.wrist_drop_detected = wrist.wrist_drop_detected
        merged.wrist_observation_rate = wrist.wrist_observation_rate
        merged.wrist_acquire_rate = wrist.wrist_acquire_rate
        merged.wrist_early_hold_rate = wrist.wrist_early_hold_rate
        merged.wrist_late_hold_rate = wrist.wrist_late_hold_rate
        merged.wrist_approach_drop_norm = wrist.wrist_approach_drop_norm
        merged.wrist_min_object_gripper_norm = wrist.wrist_min_object_gripper_norm
    merged.available = True
    merged.cameras = [
        name
        for name, available in (
            ("cam_mid_chest", chest.available),
            ("cam_left_wrist", wrist.available),
        )
        if available
    ]
    merged.camera = "+".join(merged.cameras)
    merged.sample_count = chest.sample_count + wrist.sample_count
    merged.notes = chest.notes + wrist.notes
    if chest.available and wrist.available:
        merged.confidence = min(
            0.93, 0.58 * chest.confidence + 0.42 * wrist.confidence + 0.08
        )
    else:
        merged.confidence = max(chest.confidence, wrist.confidence) * 0.88
    return merged


def attach_vision_evidence(
    data_dir: Path,
    signals: list[TaskEvidence],
    *,
    segment_assignments: TaskSegmentAssignments | None = None,
) -> None:
    """按真实时间戳为 TaskEvidence 附加胸前与腕部视频证据。"""
    videos = {
        "cam_mid": data_dir / "videos" / "cam_mid_chest.mp4",
        "cam_left": data_dir / "videos" / "cam_left_wrist.mp4",
    }
    assignments = segment_assignments or resolve_task_segments(data_dir)
    ranges = {
        key: assignments.video_ranges.get(key, {})
        for key in videos
    }
    timestamp_maps = load_camera_frame_timestamps(data_dir)

    for task in signals:
        event_times = [
            value
            for value in (task.grasp_timestamp, task.release_timestamp)
            if value is not None
        ]
        camera_evidence: dict[str, VisionEvidence] = {}
        for key, video_path in videos.items():
            frame_range = ranges[key].get(task.task_index)
            if frame_range is None or not video_path.is_file():
                camera_evidence[key] = VisionEvidence(
                    notes=[f"{key} 缺少视频或 task 帧范围"]
                )
                continue
            timestamp_map = timestamp_maps.get(key, {})
            selected = select_frame_indices(
                *frame_range,
                timestamp_by_frame=timestamp_map,
                event_timestamps=event_times,
            )
            decoded = decode_selected_frames(
                video_path,
                selected,
                timestamp_by_frame=timestamp_map,
            )
            if key == "cam_mid":
                camera_evidence[key] = analyze_chest_frames(
                    decoded,
                    grasp_timestamp=task.grasp_timestamp,
                    release_timestamp=task.release_timestamp,
                )
            else:
                camera_evidence[key] = analyze_wrist_frames(
                    decoded,
                    grasp_timestamp=task.grasp_timestamp,
                    release_timestamp=task.release_timestamp,
                )
        task.vision = _merge_evidence(
            camera_evidence["cam_mid"], camera_evidence["cam_left"]
        )
