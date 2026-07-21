"""Task segment 识别接口及当前日志实现。

评分流水线只依赖 :class:`TaskSegmentAssignments`，不关心分段是由
machine_flow/task_segments、YOLO，还是其他工具产生。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


VideoFrameRange = tuple[int, int]


@dataclass(frozen=True)
class TaskSegmentAssignments:
    """一次推理数据中的统一分段结果。

    ``state_frame_to_segment`` 用于切分 eval_log；``video_ranges`` 用于
    切分各相机视频。未来的 YOLO 实现只需产生相同结构。
    """

    state_frame_to_segment: dict[int, int] = field(default_factory=dict)
    video_ranges: dict[str, dict[int, VideoFrameRange]] = field(default_factory=dict)
    source: str = "unknown"
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    @property
    def segment_indices(self) -> tuple[int, ...]:
        indices = set(self.state_frame_to_segment.values())
        for ranges in self.video_ranges.values():
            indices.update(ranges)
        return tuple(sorted(indices))


class TaskSegmentResolver(Protocol):
    """可替换的 task segment 识别器。"""

    def resolve(self, data_dir: Path) -> TaskSegmentAssignments:
        """识别并返回状态帧与视频帧的分段归属。"""
        ...


class RecordedTaskSegmentResolver:
    """使用推理数据中已记录的分段标记。"""

    camera_keys = ("cam_mid", "cam_left")

    def resolve(self, data_dir: Path) -> TaskSegmentAssignments:
        data_dir = data_dir.resolve()
        state_mapping = self._load_state_mapping(data_dir)
        video_ranges = {
            camera_key: self._load_video_ranges(data_dir, camera_key)
            for camera_key in self.camera_keys
        }
        warnings: list[str] = []
        if not state_mapping:
            warnings.append("缺少状态帧到 task segment 的映射")
        if not any(video_ranges.values()):
            warnings.append("缺少 task segment 的视频起止帧")
        return TaskSegmentAssignments(
            state_frame_to_segment=state_mapping,
            video_ranges=video_ranges,
            source="recorded_metadata",
            confidence=None,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _load_state_mapping(data_dir: Path) -> dict[int, int]:
        mapping: dict[int, int] = {}
        machine_flow = data_dir / "machine_flow.jsonl"
        if machine_flow.is_file():
            with machine_flow.open(encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("event") != "chunk_inference":
                        continue
                    frame_index = item.get("frame_index")
                    segment_index = item.get("task_segment_index")
                    if frame_index is not None and segment_index is not None:
                        mapping[int(frame_index)] = int(segment_index)
        if mapping:
            return mapping

        parquet = data_dir / "trajectory_data" / "trajectory.parquet"
        if parquet.is_file():
            import pandas as pd

            frame_table = pd.read_parquet(
                parquet, columns=["inference_frame_index", "task_segment_index"]
            )
            for frame_index, segment_index in zip(
                frame_table["inference_frame_index"],
                frame_table["task_segment_index"],
            ):
                mapping[int(frame_index)] = int(segment_index)
        return mapping

    @staticmethod
    def _load_video_ranges(
        data_dir: Path, camera_key: str
    ) -> dict[int, VideoFrameRange]:
        path = data_dir / "task_segments.jsonl"
        if not path.is_file():
            return {}

        boundaries: dict[int, dict[str, int]] = {}
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("type") != "task_boundary":
                    continue
                segment_index = item.get("task_segment_index")
                event = item.get("event")
                frame_count = (item.get("video_frame_counts") or {}).get(camera_key)
                if (
                    segment_index is None
                    or event not in {"start", "end"}
                    or frame_count is None
                ):
                    continue
                boundaries.setdefault(int(segment_index), {})[event] = int(frame_count)

        ranges: dict[int, VideoFrameRange] = {}
        for segment_index, values in boundaries.items():
            if "start" not in values or "end" not in values:
                continue
            start, end = values["start"], values["end"] - 1
            if end >= start:
                ranges[segment_index] = (start, end)
        return ranges


def resolve_task_segments(
    data_dir: Path, resolver: TaskSegmentResolver | None = None
) -> TaskSegmentAssignments:
    """通过注入的识别器获取分段，默认使用数据中的现有标记。"""
    return (resolver or RecordedTaskSegmentResolver()).resolve(data_dir)
