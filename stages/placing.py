"""S4 投放释放阶段评估器。"""

from domain.models import StageResult, TaskEvidence
from domain.rules_v5 import V5_RULES
from stages.common import stage_confidence, stage_result


def score_placing(task: TaskEvidence) -> StageResult:
    decision = V5_RULES.decide("S4投放", task)
    evidence = [
        f"夹爪释放={task.grip_release_detected}",
        f"最终物体位置={task.vision.final_object_relation}",
        decision.reason,
    ]
    return stage_result(
        "S4投放",
        decision,
        evidence,
        confidence=stage_confidence("S4投放", task, decision.strength),
    )
