"""S5 机械臂归位阶段评估器。"""

from domain.models import StageResult, TaskEvidence
from domain.rules_v5 import V5_RULES
from stages.common import stage_confidence, stage_result


def score_return(task: TaskEvidence) -> StageResult:
    decision = V5_RULES.decide("S5归位", task)
    origin = ", ".join(f"{value:.3f}" for value in task.task_start_ee) or "unknown"
    evidence = [
        f"task原点=({origin})",
        f"最大离家={task.max_home_excursion_m:.3f}m",
        f"回撤进度={task.return_progress_m:.3f}m",
        f"终点误差: xy={task.home_xy_error_m:.3f}m, z={task.home_z_error_m:.3f}m",
        f"回撤耗时={task.return_duration_sec:.2f}s",
        decision.reason,
    ]
    return stage_result(
        "S5归位",
        decision,
        evidence,
        confidence=stage_confidence("S5归位", task, decision.strength),
    )
