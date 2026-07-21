"""S1 定位与接近阶段评估器。"""

from domain.models import StageResult, TaskEvidence
from domain.rules_v5 import V5_RULES
from stages.common import stage_confidence, stage_result


def score_positioning(task: TaskEvidence) -> StageResult:
    decision = V5_RULES.decide("S1定位", task)
    evidence = [
        f"EE路径={task.ee_path_m:.3f}m",
        f"靠近直线度={task.path_efficiency:.2f}",
        f"抓取候选到达时间={task.grasp_time_sec:.2f}s",
        decision.reason,
    ]
    return stage_result(
        "S1定位",
        decision,
        evidence,
        confidence=stage_confidence("S1定位", task, decision.strength),
    )
