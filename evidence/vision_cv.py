"""OpenCV 颜色分割、连通域跟踪和几何关系。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from domain.rules_v5 import V5_RULES


@dataclass
class MaskObservation:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    mask: np.ndarray


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clean_mask(mask: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)


def red_mask(frame: np.ndarray) -> np.ndarray:
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
    return clean_mask(cv2.bitwise_or(low_red, high_red))


def yellow_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array([14, 80, 45], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8),
    )
    return clean_mask(yellow)


def mask_candidates(mask: np.ndarray, min_pixels: int) -> list[MaskObservation]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[MaskObservation] = []
    for contour in contours:
        area = int(round(cv2.contourArea(contour)))
        if area < min_pixels:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        centroid = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
        x, y, width, height = cv2.boundingRect(contour)
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
        candidates.append(
            MaskObservation(
                centroid=centroid,
                bbox=(x, y, x + width - 1, y + height - 1),
                area=area,
                mask=component,
            )
        )
    return candidates


def track_observations(
    frames: np.ndarray, mask_fn, min_pixels: int
) -> list[MaskObservation | None]:
    """按面积初始化，后续优先选择位置连续的连通域。"""
    tracked: list[MaskObservation | None] = []
    previous: MaskObservation | None = None
    diagonal = math.hypot(frames.shape[2], frames.shape[1]) if len(frames) else 1.0
    for frame in frames:
        candidates = mask_candidates(mask_fn(frame), min_pixels)
        if not candidates:
            tracked.append(None)
            continue
        if previous is None:
            chosen = max(candidates, key=lambda item: item.area)
        else:
            chosen = max(
                candidates,
                key=lambda item: (
                    math.log1p(item.area)
                    - 4.0
                    * distance(item.centroid, previous.centroid)
                    / max(diagonal, 1.0)
                ),
            )
        tracked.append(chosen)
        previous = chosen
    return tracked


def box_relation(
    obj: MaskObservation | None, box: MaskObservation | None
) -> tuple[str, float]:
    """计算物体相对盒子内圈的 inside / edge / outside。"""
    if obj is None or box is None:
        return "unknown", 0.0
    bx1, by1, bx2, by2 = box.bbox
    margin_x = max(2, round((bx2 - bx1 + 1) * V5_RULES.vision_box_inner_margin_frac))
    margin_y = max(2, round((by2 - by1 + 1) * V5_RULES.vision_box_inner_margin_frac))
    inner = np.zeros_like(box.mask)
    cv2.rectangle(
        inner,
        (bx1 + margin_x, by1 + margin_y),
        (bx2 - margin_x, by2 - margin_y),
        255,
        thickness=cv2.FILLED,
    )
    obj_pixels = max(1, int(np.count_nonzero(obj.mask)))
    inside_ratio = float(
        np.count_nonzero(cv2.bitwise_and(obj.mask, inner)) / obj_pixels
    )
    outer_ratio = float(
        np.count_nonzero(cv2.bitwise_and(obj.mask, box.mask)) / obj_pixels
    )
    cx, cy = obj.centroid
    centroid_in_inner = (
        bx1 + margin_x <= cx <= bx2 - margin_x
        and by1 + margin_y <= cy <= by2 - margin_y
    )
    if centroid_in_inner and inside_ratio >= V5_RULES.vision_inside_ratio:
        return "inside", inside_ratio
    if outer_ratio > 0.08 or bx1 <= cx <= bx2 and by1 <= cy <= by2:
        return "edge", inside_ratio
    return "outside", inside_ratio


def track_motion_norm(
    observations: list[MaskObservation | None],
    indices: list[int],
    diagonal: float,
) -> float:
    points = [observations[i].centroid for i in indices if observations[i] is not None]
    if len(points) < 2:
        return 0.0
    origin = points[0]
    return max(distance(origin, point) for point in points) / diagonal
