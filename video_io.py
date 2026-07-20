"""视频 task 边界、真实时间戳和按需解码。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    VISION_EVENT_WINDOW_SEC,
    VISION_HEIGHT,
    VISION_SAMPLE_STRIDE,
    VISION_WIDTH,
)


@dataclass
class DecodedFrames:
    frames: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray


def load_task_video_ranges(
    data_dir: Path, camera_key: str = "cam_mid"
) -> dict[int, tuple[int, int]]:
    """读取某个相机在各 task 内的闭区间帧范围。"""
    path = data_dir / "task_segments.jsonl"
    if not path.is_file():
        return {}

    boundaries: dict[int, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") != "task_boundary":
                continue
            task_idx = item.get("task_segment_index")
            event = item.get("event")
            frame_count = (item.get("video_frame_counts") or {}).get(camera_key)
            if task_idx is None or event not in {"start", "end"} or frame_count is None:
                continue
            boundaries.setdefault(int(task_idx), {})[event] = int(frame_count)

    ranges: dict[int, tuple[int, int]] = {}
    for task_idx, values in boundaries.items():
        if "start" not in values or "end" not in values:
            continue
        start = values["start"]
        end = values["end"] - 1
        if end >= start:
            ranges[task_idx] = (start, end)
    return ranges


def load_camera_frame_timestamps(data_dir: Path) -> dict[str, dict[int, float]]:
    """读取 camera_key -> video_frame_index -> image_timestamp。"""
    path = data_dir / "videos" / "frame_timestamps.jsonl"
    result: dict[str, dict[int, float]] = {}
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("camera_key")
            frame_idx = item.get("video_frame_index")
            timestamp = item.get("image_timestamp")
            if key is None or frame_idx is None or timestamp is None:
                continue
            result.setdefault(str(key), {})[int(frame_idx)] = float(timestamp)
    return result


def select_frame_indices(
    start_frame: int,
    end_frame: int,
    *,
    timestamp_by_frame: dict[int, float] | None,
    event_timestamps: list[float],
    stride: int = VISION_SAMPLE_STRIDE,
) -> list[int]:
    """普通阶段稀疏采样，事件前后窗口保留全部帧。"""
    selected = set(range(start_frame, end_frame + 1, max(1, stride)))
    selected.update((start_frame, end_frame))
    if timestamp_by_frame:
        for frame_idx in range(start_frame, end_frame + 1):
            timestamp = timestamp_by_frame.get(frame_idx)
            if timestamp is None:
                continue
            if any(
                abs(timestamp - event_time) <= VISION_EVENT_WINDOW_SEC
                for event_time in event_timestamps
            ):
                selected.add(frame_idx)
    return sorted(selected)


def decode_selected_frames(
    video_path: Path,
    selected_indices: list[int],
    *,
    timestamp_by_frame: dict[int, float] | None = None,
    width: int = VISION_WIDTH,
    height: int = VISION_HEIGHT,
) -> DecodedFrames:
    """顺序解码选定帧，并保留真实采集时间。"""
    empty = DecodedFrames(
        np.empty((0, height, width, 3), dtype=np.uint8),
        np.empty(0),
        np.empty(0, dtype=int),
    )
    if not video_path.is_file() or not selected_indices:
        return empty

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return empty

    wanted = set(selected_indices)
    first, last = selected_indices[0], selected_indices[-1]
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    indices: list[int] = []
    frame_idx = first
    try:
        while frame_idx <= last:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_idx in wanted:
                resized = cv2.resize(
                    bgr, (width, height), interpolation=cv2.INTER_AREA
                )
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                indices.append(frame_idx)
                timestamps.append(
                    (timestamp_by_frame or {}).get(frame_idx, float("nan"))
                )
            frame_idx += 1
    finally:
        capture.release()

    if not frames:
        return empty
    return DecodedFrames(
        np.stack(frames),
        np.asarray(timestamps, dtype=float),
        np.asarray(indices, dtype=int),
    )
