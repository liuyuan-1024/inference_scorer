"""阶段独立置信度与统一阶段结果构造。"""

from __future__ import annotations

import numpy as np

from domain.models import StageResult, TaskEvidence
from domain.rules_v5 import RuleDecision, V5_RULES, V5_STAGE_BY_KEY


def stage_confidence(
    stage: str,
    task: TaskEvidence,
    decision_strength: float,
) -> float:
    """只依据当前阶段所需模态和质量独立计算置信度。"""
    policy = V5_STAGE_BY_KEY[stage].confidence
    sensor = task.sensor_confidence if task.sensor_confidence > 0 else 0.35
    vision_available = task.vision.available
    vision = task.vision.confidence if vision_available else 0.30

    measured = policy.sensor_weight * sensor + policy.vision_weight * vision
    if policy.vision_weight and not vision_available:
        measured -= policy.missing_vision_penalty

    # 数据质量按该阶段对相应模态的依赖比例扣减。
    if task.state_feedback_timeout_rate >= policy.feedback_timeout_threshold:
        measured -= policy.feedback_penalty * policy.sensor_weight
    if (
        policy.vision_weight
        and np.isfinite(task.image_state_diff_p95_sec)
        and task.image_state_diff_p95_sec > V5_RULES.image_state_sync_good_sec
    ):
        measured -= policy.sync_penalty * policy.vision_weight

    # 只有 S2 依赖“接触”融合，视觉/力觉冲突不应污染其他阶段。
    if stage == "S2抓取" and task.vision.wrist_available:
        sensor_contact = (
            task.jc_contact_delta >= V5_RULES.jc_contact_delta
            or task.fz_contact_delta >= V5_RULES.fz_contact_delta_min
        )
        if sensor_contact != task.events.object_between_jaws:
            measured -= policy.modality_conflict_penalty

    # S3/S4 的核心结果若不可见，单独降低对应阶段而不影响 S1/S2/S5。
    if stage in {"S3搬运", "S4投放"} and task.vision.final_object_relation == "unknown":
        measured -= policy.unknown_relation_penalty

    combined = 0.75 * measured + 0.25 * decision_strength
    return round(float(np.clip(combined, 0.20, policy.max_confidence)), 3)


def stage_result(
    stage: str,
    decision: RuleDecision,
    evidence: list[str],
    *,
    confidence: float,
) -> StageResult:
    """将 v5 规则决策封装为标准阶段输出。"""
    spec = V5_STAGE_BY_KEY[stage]
    needs_review = (
        confidence < spec.confidence.review_threshold
        or decision.review_reason is not None
    )
    return StageResult(
        stage=stage,
        score=decision.score,
        max_score=spec.max_score,
        confidence=confidence,
        needs_review=needs_review,
        evidence=tuple(evidence),
        missing_evidence=decision.missing_evidence,
        review_reason=decision.review_reason,
        rule_version=V5_RULES.version,
    )
