"""
v5 评分映射。

每个维度同时输出分数、置信度和可审计证据。对视频缺失、OOD 规则未定义、
距离接近阈值等情况保守打分并标记人工复核。
"""

from __future__ import annotations

import math

from config import (
    GRIP_OPEN_THRESHOLD,
    RETURN_GRIP_OPEN_REQUIRED,
    RETURN_MAX_SEC,
    RETURN_MIN_EXCURSION_M,
    RETURN_MIN_PROGRESS_M,
    WITHDRAW_HOME_XY_M,
    WITHDRAW_HOME_Z_M,
)
from models import TaskJudgment


def _detail(
    score: int,
    confidence: float,
    evidence: list[str],
    *,
    review_reason: str | None = None,
) -> dict:
    confidence = max(0.0, min(1.0, float(confidence)))
    needs_review = confidence < 0.70 or review_reason is not None
    result = {
        "score": int(score),
        "confidence": round(confidence, 3),
        "needs_review": needs_review,
        "evidence": evidence,
    }
    if review_reason:
        result["review_reason"] = review_reason
    return result


def detail_S1_positioning(t: TaskJudgment, fps: float = 15.0) -> dict:
    evidence = [
        f"EE路径={t.ee_path_m:.3f}m",
        f"靠近直线度={t.path_efficiency:.2f}",
    ]
    if not t.has_motion:
        return _detail(0, 0.90, evidence + ["所有关节/EE均未达到1°/1cm运动阈值"])

    if t.object_present is False:
        return _detail(
            1,
            max(0.80, t.confidence),
            evidence + ["视频确认场景中没有目标物，机械臂仍发生移动"],
        )

    has_approach_geometry = (
        t.approach_z_drop_m >= 0.02 or t.approach_xy_m >= 0.03
    )
    if t.has_grasp_attempt and has_approach_geometry:
        evidence.append(f"抓取候选到达时间={t.grasp_time_sec:.2f}s")
        evidence.append(
            f"靠近位移: xy={t.approach_xy_m:.3f}m, z={t.approach_z_drop_m:.3f}m"
        )
        if t.path_efficiency >= 0.30 and t.grasp_time_sec <= 5.0:
            return _detail(
                4,
                0.68 if t.vision_available else 0.52,
                evidence,
                review_reason="缺少夹爪中心到目标物的厘米级相机标定",
            )
        if t.grasp_time_sec <= 10.0:
            return _detail(
                3,
                0.66 if t.vision_available else 0.50,
                evidence,
                review_reason="需复核夹爪与目标物的2cm/3cm空间阈值",
            )
        return _detail(2, 0.65, evidence + ["到达耗时超过10秒或路径效率不足"])

    if has_approach_geometry and t.path_efficiency >= 0.30:
        return _detail(
            2,
            0.62,
            evidence + ["向目标区域靠近但未形成可信抓取事件"],
            review_reason="无目标物三维标定，定位偏差仅能保守估计",
        )
    return _detail(1, 0.72, evidence + ["有移动，但没有形成有效靠近/抓取证据"])


def detail_S2_grasping(t: TaskJudgment) -> dict:
    evidence = list(t.evidence)
    if t.object_present is False:
        return _detail(0, max(0.88, t.confidence), evidence + ["视频确认无目标物"])
    if not t.has_grasp_attempt:
        return _detail(0, 0.88 if t.vision_available else 0.72, evidence + ["无闭合事件"])
    if t.grip_empty_close:
        return _detail(1, max(0.88, t.confidence), evidence + ["夹爪完全闭合，判定抓空"])
    if t.has_grasp_object:
        return _detail(
            3,
            max(0.78, t.confidence),
            evidence
            + [
                f"抬升={t.retract_z_rise_m:.3f}m",
                f"物体视觉运动={t.object_motion_norm:.3f}",
            ],
        )
    if t.has_grasp_contact and t.retract_z_rise_m >= 0.02:
        return _detail(
            2,
            0.72 if t.vision_available else 0.58,
            evidence + ["存在接触/短暂抬升，但未满足5cm稳定抓取条件"],
            review_reason=None if t.vision_available else "无视频证据确认是否掉落",
        )
    return _detail(
        1,
        0.78 if t.vision_available else 0.55,
        evidence + ["执行闭合但无物体随动/稳定负载证据"],
        review_reason=None if t.vision_available else "仅有传感器证据",
    )


def detail_S3_transport(t: TaskJudgment) -> dict:
    evidence = [
        f"运输距离={t.grasp_transport_m:.3f}m",
        f"向盒子移动={t.transport_to_place}",
    ]
    if t.object_present is False:
        return _detail(0, max(0.88, t.confidence), evidence + ["无目标物"])
    if t.box_present is False:
        return _detail(
            0,
            max(0.82, t.confidence),
            evidence
            + [
                "OOD场景无盒子，v5无N/A列，按未搬运至目标记0分",
                "策略提示：可另设独立鲁棒性指标，但本项无需人工复核",
            ],
        )
    if not t.has_grasp_object:
        return _detail(0, 0.84 if t.vision_available else 0.68, evidence + ["无稳定抓取"])
    if t.transport_to_place and t.final_object_relation in {"inside", "edge"}:
        return _detail(2, max(0.80, t.confidence), evidence + ["物体到达盒口区域"])
    if t.transport_to_place or t.grasp_transport_m >= 0.05:
        return _detail(1, 0.75 if t.vision_available else 0.58, evidence)
    return _detail(0, 0.78, evidence + ["抓取后未形成有效水平搬运"])


def detail_S4_placing(t: TaskJudgment) -> dict:
    evidence = [
        f"夹爪释放={t.grip_release_detected}",
        f"最终物体位置={t.final_object_relation}",
    ]
    if t.object_present is False:
        return _detail(0, max(0.88, t.confidence), evidence + ["无目标物"])
    if t.box_present is False:
        return _detail(
            0,
            max(0.82, t.confidence),
            evidence
            + [
                "OOD场景无盒子，按未投放记0分",
                "策略提示：可另设独立鲁棒性指标，但本项无需人工复核",
            ],
        )
    if not t.grip_release_detected:
        return _detail(0, 0.88, evidence + ["无张开释放事件"])
    if not t.has_grasp_object:
        return _detail(
            0,
            0.84 if t.vision_available else 0.62,
            evidence + ["没有稳定夹持物体，张开动作不构成投放"],
        )
    if t.final_object_relation == "inside":
        return _detail(3, max(0.82, t.confidence), evidence)
    if t.final_object_relation == "edge":
        return _detail(2, max(0.78, t.confidence), evidence)
    if t.final_object_relation == "outside":
        return _detail(1, max(0.80, t.confidence), evidence)
    return _detail(
        1,
        0.45,
        evidence + ["检测到释放，但最终落点不可见"],
        review_reason="必须查看释放后的连续视频",
    )


def detail_S5_return(t: TaskJudgment) -> dict:
    evidence = [
        f"最大离家={t.max_home_excursion_m:.3f}m",
        f"回撤进度={t.return_progress_m:.3f}m",
        f"终点误差: xy={t.home_xy_error_m:.3f}m, z={t.home_z_error_m:.3f}m",
        f"回撤耗时={t.return_duration_sec:.2f}s",
    ]
    if (
        t.max_home_excursion_m < RETURN_MIN_EXCURSION_M
        or t.return_progress_m < RETURN_MIN_PROGRESS_M
    ):
        return _detail(0, 0.86, evidence + ["未形成明确的离开后归位动作"])

    grip_known = not math.isnan(t.grip_final_val)
    grip_ok = (
        t.grip_final_val > GRIP_OPEN_THRESHOLD
        if RETURN_GRIP_OPEN_REQUIRED and grip_known
        else True
    )
    position_ok = (
        t.home_xy_error_m <= WITHDRAW_HOME_XY_M
        and t.home_z_error_m <= WITHDRAW_HOME_Z_M
    )
    time_ok = t.return_duration_sec <= RETURN_MAX_SEC
    if position_ok and time_ok and grip_ok:
        return _detail(
            2,
            0.90 if grip_known else 0.72,
            evidence + ["位置、时间和夹爪状态均满足归位标准"],
        )
    reasons = []
    if not position_ok:
        reasons.append("终点位置超差")
    if not time_ok:
        reasons.append("归位超过10秒")
    if not grip_ok:
        reasons.append("归位后夹爪未张开")
    return _detail(1, 0.86 if grip_known else 0.68, evidence + reasons)


def compute_score_details(
    judgments: list[TaskJudgment], fps: float = 15.0
) -> dict[int, dict[str, dict]]:
    """计算分数及审计信息。"""
    results: dict[int, dict[str, dict]] = {}
    for task in judgments:
        results[task.task_index] = {
            "S1定位": detail_S1_positioning(task, fps),
            "S2抓取": detail_S2_grasping(task),
            "S3搬运": detail_S3_transport(task),
            "S4投放": detail_S4_placing(task),
            "S5归位": detail_S5_return(task),
        }
    return results


def compute_all_scores(
    judgments: list[TaskJudgment], fps: float = 15.0
) -> dict[int, dict[str, int]]:
    """保持原接口：只返回各维度整数分数。"""
    details = compute_score_details(judgments, fps)
    return {
        task_idx: {dim: int(item["score"]) for dim, item in dims.items()}
        for task_idx, dims in details.items()
    }


# 保留旧函数名，避免外部调用方中断。
def score_S1_positioning(t: TaskJudgment, fps: float = 15.0) -> int:
    return int(detail_S1_positioning(t, fps)["score"])


def score_S2_grasping(t: TaskJudgment) -> int:
    return int(detail_S2_grasping(t)["score"])


def score_S3_transport(t: TaskJudgment) -> int:
    return int(detail_S3_transport(t)["score"])


def score_S4_placing(t: TaskJudgment) -> int:
    return int(detail_S4_placing(t)["score"])


def score_S5_return(t: TaskJudgment) -> int:
    return int(detail_S5_return(t)["score"])
