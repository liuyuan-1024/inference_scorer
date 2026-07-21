"""与具体输出格式无关的评分流水线。"""

from __future__ import annotations

from pathlib import Path

from domain.models import ScoringRun
from domain.rules_v5 import RULE_VERSION
from evidence.analysis import analyze_data_dir
from inputs.segments import TaskSegmentResolver
from stages.scoring import compute_task_scores


def evaluate_data_dir(
    data_dir: Path, *, segment_resolver: TaskSegmentResolver | None = None
) -> ScoringRun:
    """读取一个 action_step 目录并生成完整评分结果。

    本函数不要求 Excel 存在，也不写任何输出文件。CLI、JSON 和 Excel
    只是它的消费者。
    """
    resolved = data_dir.resolve()
    evidence = tuple(analyze_data_dir(resolved, segment_resolver=segment_resolver))
    tasks = compute_task_scores(evidence)
    return ScoringRun(
        data_dir=str(resolved),
        evidence=evidence,
        tasks=tasks,
        rule_version=RULE_VERSION,
    )
