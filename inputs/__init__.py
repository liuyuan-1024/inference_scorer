"""评测日志、视频与 task segment 输入适配器。"""

from inputs.segments import (
    RecordedTaskSegmentResolver,
    TaskSegmentAssignments,
    TaskSegmentResolver,
    resolve_task_segments,
)

__all__ = [
    "RecordedTaskSegmentResolver",
    "TaskSegmentAssignments",
    "TaskSegmentResolver",
    "resolve_task_segments",
]
