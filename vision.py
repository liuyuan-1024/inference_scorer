"""
轻量视频证据提取。

使用 OpenCV 精确读取任务帧，通过 HSV 分割、形态学去噪和轮廓分析识别
红色玩具与黄色盒子。视觉结果用于否决明显误判，并为低置信度样本提供
人工复核线索；它不是通用目标检测器。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from config import (
    VISION_HEIGHT,
    VISION_MIN_BOX_PIXELS,
    VISION_MIN_OBJECT_PIXELS,
    VISION_PRESENT_RATE,
    VISION_SAMPLE_STRIDE,
    VISION_TOWARD_BOX_NORM,
    VISION_WIDTH,
)
from models import TaskSignals, VisionEvidence


def load_task_video_ranges(
    data_dir: Path, camera_key: str = "cam_mid"
) -> dict[int, tuple[int, int]]:
    """从 task_segments.jsonl 读取每个任务对应的视频帧范围。"""
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


def _decode_sampled_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    *,
    stride: int = VISION_SAMPLE_STRIDE,
    width: int = VISION_WIDTH,
    height: int = VISION_HEIGHT,
) -> np.ndarray:
    """按帧号抽样并返回 RGB 帧；失败时返回空数组。"""
    if not video_path.is_file():
        return np.empty((0, height, width, 3), dtype=np.uint8)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return np.empty((0, height, width, 3), dtype=np.uint8)

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    frame_idx = start_frame
    sample_stride = max(1, stride)
    try:
        while frame_idx <= end_frame:
            ok, bgr = capture.read()
            if not ok:
                break
            if (frame_idx - start_frame) % sample_stride == 0:
                resized = cv2.resize(
                    bgr, (width, height), interpolation=cv2.INTER_AREA
                )
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        capture.release()

    if not frames:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    return np.stack(frames)


def _red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    low_red = cv2.inRange(
        hsv,
        np.array([0, 95, 38], dtype=np.uint8),
        np.array([12, 255, 255], dtype=np.uint8),
    )
    high_red = cv2.inRange(
        hsv,
        np.array([168, 95, 38], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    return _clean_mask(cv2.bitwise_or(low_red, high_red))


def _yellow_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array([14, 80, 45], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8),
    )
    return _clean_mask(yellow)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """去除高光噪点并连接目标内部的小孔洞。"""
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)


def _mask_observation(
    mask: np.ndarray, min_pixels: int
) -> tuple[tuple[float, float] | None, tuple[int, int, int, int] | None, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0
    contour = max(contours, key=cv2.contourArea)
    count = int(round(cv2.contourArea(contour)))
    if count < min_pixels:
        return None, None, count
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return None, None, count
    centroid = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    x, y, width, height = cv2.boundingRect(contour)
    bbox = (x, y, x + width - 1, y + height - 1)
    return centroid, bbox, count


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rect_relation(
    obj: tuple[int, int, int, int] | None,
    box: tuple[int, int, int, int] | None,
) -> str:
    if obj is None or box is None:
        return "unknown"
    ox1, oy1, ox2, oy2 = obj
    bx1, by1, bx2, by2 = box
    margin = 3
    inside = (
        ox1 >= bx1 + margin
        and oy1 >= by1 + margin
        and ox2 <= bx2 - margin
        and oy2 <= by2 - margin
    )
    if inside:
        return "inside"
    intersects = max(ox1, bx1) <= min(ox2, bx2) and max(oy1, by1) <= min(oy2, by2)
    return "edge" if intersects else "outside"


def analyze_frames(
    frames: np.ndarray, *, grasp_fraction: float | None = None
) -> VisionEvidence:
    """分析已抽样的 RGB 帧。公开此函数便于合成数据单元测试。"""
    if len(frames) == 0:
        return VisionEvidence(notes=["未读取到视频帧"])

    object_centers: list[tuple[float, float] | None] = []
    object_boxes: list[tuple[int, int, int, int] | None] = []
    box_centers: list[tuple[float, float] | None] = []
    box_boxes: list[tuple[int, int, int, int] | None] = []

    for frame in frames:
        object_center, object_box, _ = _mask_observation(
            _red_mask(frame), VISION_MIN_OBJECT_PIXELS
        )
        box_center, box_box, _ = _mask_observation(
            _yellow_mask(frame), VISION_MIN_BOX_PIXELS
        )
        object_centers.append(object_center)
        object_boxes.append(object_box)
        box_centers.append(box_center)
        box_boxes.append(box_box)

    n = len(frames)
    early_n = max(3, n // 4)
    object_rate = sum(x is not None for x in object_centers[:early_n]) / early_n
    box_rate = sum(x is not None for x in box_centers[:early_n]) / early_n
    object_present = object_rate >= VISION_PRESENT_RATE
    box_present = box_rate >= VISION_PRESENT_RATE

    if object_present and box_present:
        scenario = "normal"
    elif object_present:
        scenario = "object_without_box"
    elif box_present:
        scenario = "box_without_object"
    else:
        scenario = "empty_or_unrecognized"

    diagonal = math.hypot(frames.shape[2], frames.shape[1])
    valid_object = [(i, p) for i, p in enumerate(object_centers) if p is not None]
    object_motion = 0.0
    if valid_object:
        origin = valid_object[0][1]
        object_motion = max(_distance(origin, p) for _, p in valid_object) / diagonal

    close_idx = 0
    if grasp_fraction is not None:
        close_idx = max(0, min(n - 1, round(grasp_fraction * (n - 1))))
    close_candidates = [(i, p) for i, p in valid_object if i <= close_idx]
    if close_candidates:
        close_origin = close_candidates[-1][1]
    elif valid_object:
        close_origin = valid_object[0][1]
    else:
        close_origin = None
    post_object = [p for i, p in valid_object if i >= close_idx]
    motion_after_grasp = (
        max(_distance(close_origin, p) for p in post_object) / diagonal
        if close_origin is not None and post_object
        else 0.0
    )

    distance_drop = 0.0
    initial_pairs = [
        (object_centers[i], box_centers[i])
        for i in range(min(early_n, n))
        if object_centers[i] is not None and box_centers[i] is not None
    ]
    post_pairs = [
        (object_centers[i], box_centers[i])
        for i in range(close_idx, n)
        if object_centers[i] is not None and box_centers[i] is not None
    ]
    if initial_pairs and post_pairs:
        initial_distance = float(
            np.median([_distance(obj, box) for obj, box in initial_pairs])
        )
        closest_after = min(_distance(obj, box) for obj, box in post_pairs)
        distance_drop = max(0.0, initial_distance - closest_after) / diagonal

    final_relation = "unknown"
    if object_present and box_present:
        for i in range(n - 1, max(-1, n - max(3, n // 5) - 1), -1):
            relation = _rect_relation(object_boxes[i], box_boxes[i])
            if relation != "unknown":
                final_relation = relation
                break

    confidence = 0.55
    if max(object_rate, box_rate) >= 0.8:
        confidence += 0.15
    if scenario in {"normal", "object_without_box", "box_without_object"}:
        confidence += 0.15
    if n >= 20:
        confidence += 0.10
    confidence = min(0.95, confidence)

    notes = [
        f"场景={scenario}",
        f"红色玩具检出率={object_rate:.0%}",
        f"黄色盒子检出率={box_rate:.0%}",
    ]
    if final_relation != "unknown":
        notes.append(f"最终玩具位置={final_relation}")

    return VisionEvidence(
        available=True,
        camera="cam_mid_chest",
        sample_count=n,
        scenario=scenario,
        object_present=object_present,
        box_present=box_present,
        object_detection_rate=object_rate,
        box_detection_rate=box_rate,
        object_motion_norm=object_motion,
        object_motion_after_grasp_norm=motion_after_grasp,
        object_box_distance_drop_norm=distance_drop,
        moved_toward_box=distance_drop >= VISION_TOWARD_BOX_NORM,
        final_object_relation=final_relation,
        confidence=confidence,
        notes=notes,
    )


def attach_vision_evidence(data_dir: Path, signals: list[TaskSignals]) -> None:
    """为已有 TaskSignals 就地附加胸前视频证据。"""
    video_path = data_dir / "videos" / "cam_mid_chest.mp4"
    ranges = load_task_video_ranges(data_dir)
    if not video_path.is_file() or not ranges:
        return

    for task in signals:
        frame_range = ranges.get(task.task_index)
        if frame_range is None:
            task.vision.notes.append("缺少任务视频帧范围")
            continue
        grasp_fraction = (
            task.grasp_phase_idx / max(task.n_frames - 1, 1)
            if task.grasp_phase_idx is not None
            else None
        )
        frames = _decode_sampled_frames(video_path, *frame_range)
        task.vision = analyze_frames(frames, grasp_fraction=grasp_fraction)
