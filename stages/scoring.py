"""S1～S5 阶段评估器注册与评分编排。"""

from __future__ import annotations

from collections.abc import Callable

from domain.models import StageResult, TaskEvidence, TaskScore
from domain.rules_v5 import V5_STAGE_SPECS
from stages.grasping import score_grasping
from stages.placing import score_placing
from stages.positioning import score_positioning
from stages.returning import score_return
from stages.transport import score_transport


StageEvaluator = Callable[[TaskEvidence], StageResult]


def stage_evaluators() -> dict[str, StageEvaluator]:
    """返回与 v5 阶段定义顺序一致的评估器注册表。"""
    return {
        "S1定位": score_positioning,
        "S2抓取": score_grasping,
        "S3搬运": score_transport,
        "S4投放": score_placing,
        "S5归位": score_return,
    }


def compute_task_scores(
    evidence: list[TaskEvidence] | tuple[TaskEvidence, ...],
) -> dict[int, TaskScore]:
    """让每个阶段评估器直接消费统一证据，返回领域结果。"""
    evaluators = stage_evaluators()
    expected = [spec.key for spec in V5_STAGE_SPECS]
    if list(evaluators) != expected:
        raise RuntimeError("阶段评估器与 v5 阶段定义不一致")

    return {
        task.task_index: TaskScore(
            task_index=task.task_index,
            stages={name: evaluator(task) for name, evaluator in evaluators.items()},
        )
        for task in evidence
    }
