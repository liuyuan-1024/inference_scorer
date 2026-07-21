"""S3 搬运至目标阶段评估器。"""

from domain.models import StageResult, TaskEvidence
from domain.rules_v5 import V5_RULES
from stages.common import stage_confidence, stage_result


def score_transport(task: TaskEvidence) -> StageResult:
    decision = V5_RULES.decide("S3搬运", task)
    evidence = [
        f"运输距离={task.grasp_transport_m:.3f}m",
        f"向盒子移动={task.transport_to_place}",
        decision.reason,
    ]
    return stage_result(
        "S3搬运",
        decision,
        evidence,
        confidence=stage_confidence("S3搬运", task, decision.strength),
    )
