"""S2 抓取与抬起阶段评估器。"""

from domain.models import StageResult, TaskEvidence
from domain.rules_v5 import V5_RULES
from stages.common import stage_confidence, stage_result


def score_grasping(task: TaskEvidence) -> StageResult:
    decision = V5_RULES.decide("S2抓取", task)
    evidence = list(task.evidence) + [
        f"抬升={task.retract_z_rise_m:.3f}m",
        f"物体视觉运动={task.vision.object_motion_after_grasp_norm:.3f}",
        decision.reason,
    ]
    return stage_result(
        "S2抓取",
        decision,
        evidence,
        confidence=stage_confidence("S2抓取", task, decision.strength),
    )
