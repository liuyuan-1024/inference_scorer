"""单次 task 的语义判定与数据目录分析入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from domain.models import TaskEvidence
from domain.rules_v5 import V5_RULES
from evidence import grasp, motion
from evidence.events import build_action_event_chain
from inputs.segments import TaskSegmentResolver, resolve_task_segments

__all__ = [
    "analyze_data_dir",
    "judge_task",
]


# ===================================================================
# Task 综合判定
# ===================================================================


def judge_task(
    t: TaskEvidence, *, place_center: np.ndarray, jc_baseline: float | None = None
) -> TaskEvidence:
    """对单个 task 做综合语义判定（不依赖首帧视频）。"""
    motion.analyze_motion_phases(t, place_center=place_center)

    has_motion = motion.detect_motion(t)
    has_grasp_attempt = grasp.detect_grasp_attempt(t, has_motion)

    # OOD“有盒无物”中的闭合动作不能算作抓取。
    if t.vision.available and t.vision.object_present is False:
        has_grasp_attempt = False

    events, has_grasp_object = build_action_event_chain(
        t, has_grasp_attempt=has_grasp_attempt
    )
    has_grasp_contact = events.contact_detected

    # --- 放置相位 ---
    if t.vision.available:
        t.transport_to_place = t.vision.moved_toward_box
    has_place_phase = bool(
        events.release_detected
        and (t.vision.box_present is not False)
        and has_grasp_object
    )
    place_at_box = events.placed_inside
    if not t.vision.available and has_grasp_object:
        has_place_phase = bool(
            t.grip_release_detected
            and (
                t.grasp_transport_m >= V5_RULES.place_fallback_transport_m
                or t.retract_dist_m
                >= V5_RULES.retract_dist_m * V5_RULES.place_fallback_retract_ratio
                or t.transport_to_place
            )
        )
        if has_place_phase and t.ee_at_place:
            ee = np.array(t.ee_at_place)
            place_at_box = bool(
                np.all(
                    np.abs(ee - place_center)
                    <= np.array(
                        [
                            V5_RULES.place_roi_xy_m,
                            V5_RULES.place_roi_xy_m,
                            V5_RULES.place_roi_z_m,
                        ]
                    )
                )
            )

    # --- 动作完成 ---
    action_complete = has_motion and has_grasp_object and has_place_phase

    # --- 靠近质量 ---
    _, approach_quality = motion.classify_approach(t, has_motion)

    # --- 撤回分类 ---
    has_retract = (
        t.retract_dist_m >= V5_RULES.retract_dist_m
        or t.retract_z_rise_m >= V5_RULES.retract_z_rise_m
    )
    has_withdraw = (
        t.max_home_excursion_m >= V5_RULES.return_min_excursion_m
        and t.return_progress_m >= V5_RULES.return_min_progress_m
        and t.home_xy_error_m <= V5_RULES.withdraw_home_xy_m
        and t.home_z_error_m <= V5_RULES.withdraw_home_z_m
    )

    if has_grasp_contact and has_place_phase and (place_at_box or t.transport_to_place):
        retract_semantic = "transport_place"
    elif has_withdraw and has_retract:
        retract_semantic = "withdraw_home"
    elif has_retract:
        retract_semantic = "lift_only"
    else:
        retract_semantic = "none"

    # --- 判定理由 ---
    parts = []
    if has_motion:
        parts.append(f"EE位移={t.ee_path_m:.3f}m")
    else:
        parts.append(f"几乎不动(EE={t.ee_path_m:.3f}m)")

    if has_grasp_attempt:
        parts.append(f"抓取相位({t.grasp_detected_by or '?'})")
    if has_grasp_contact:
        parts.append(f"接触(jc={t.jc_at_grasp:.0f},Δjc={t.jc_grasp_rise:.0f})")
    if has_grasp_object:
        parts.append(f"抓上(transport={t.grasp_transport_m:.3f}m)")
    if has_place_phase:
        parts.append(f"放置{'入盒' if place_at_box else '偏'}")
    if has_withdraw:
        parts.append(f"归位({t.withdraw_home_dist_m:.3f}m)")

    evidence = list(t.vision.notes)
    if t.grip_fully_closed:
        evidence.append(
            f"夹爪闭合至{t.grip_closed_min:.3f}，为空夹候选，需结合接触与视觉"
        )
    if t.grasp_phase_idx is not None:
        evidence.append(f"抓取候选帧={t.grasp_phase_idx}({t.grasp_detected_by or '?'})")
    evidence.append(
        f"接触残差: ΔJC={t.jc_contact_delta:.0f}, ΔFz={t.fz_contact_delta:.2f}N"
    )
    evidence.extend(events.notes)
    evidence.append(
        f"数据质量: UDP age p95={t.gripper_udp_age_p95_sec * 1000:.1f}ms, "
        f"图像同步 p95={t.image_state_diff_p95_sec * 1000:.1f}ms, "
        f"反馈超时率={t.state_feedback_timeout_rate:.0%}"
    )
    reason = "；".join(parts) if parts else "信号不足"
    t.has_motion = has_motion
    t.has_grasp_attempt = has_grasp_attempt
    t.has_grasp_contact = has_grasp_contact
    t.has_grasp_object = has_grasp_object
    t.has_place_phase = has_place_phase
    t.place_at_box = place_at_box
    t.action_complete = action_complete
    t.approach_quality = approach_quality
    t.retract_semantic = retract_semantic
    t.has_retract = has_retract
    t.has_withdraw_home = has_withdraw
    t.reason = reason
    t.events = events
    t.evidence = evidence
    return t


# ===================================================================
# 工具: 学习 place_center 和 jc_baseline
# ===================================================================


def learn_place_center(signals: list[TaskEvidence]) -> np.ndarray:
    """
    从各 task 的抓取后末端位置学习放置中心。

    取有负载运输的 task 的终点位置的中位数。
    """
    place_ee = []
    for t in signals:
        ee = t.ee_at_place
        if not ee:
            continue
        if t.grasp_transport_m >= V5_RULES.grasp_transport_m * 0.5:
            place_ee.append(ee)

    if place_ee:
        arr = np.array(place_ee)
        return np.median(arr, axis=0)

    return np.array(V5_RULES.default_place_center)


def learn_place_tolerance(
    signals: list[TaskEvidence], place_center: np.ndarray
) -> np.ndarray:
    """学习放置容差。"""
    diffs = []
    for t in signals:
        ee = t.ee_at_place
        if not ee or t.grasp_transport_m < 0.03:
            continue
        diffs.append(np.abs(np.array(ee) - place_center))

    if diffs:
        arr = np.array(diffs)
        spread = np.std(arr, axis=0)
        return np.maximum(
            spread * 2.5,
            [V5_RULES.place_roi_xy_m, V5_RULES.place_roi_xy_m, V5_RULES.place_roi_z_m],
        )

    return np.array(
        [V5_RULES.place_roi_xy_m, V5_RULES.place_roi_xy_m, V5_RULES.place_roi_z_m]
    )


def learn_jc_baseline(signals: list[TaskEvidence]) -> float:
    """
    学习 JC 基线值。

    使用各 task 抓取前 JC 的 p50（中位数），再取各 task 的 p50 的中位数。
    这样即使某些 task 全程高负载也不污染基线。
    """
    task_baselines = []
    for t in signals:
        jc = t.jc_traj
        if len(jc) < 5:
            continue
        gp = t.grasp_phase_idx
        if gp is not None and gp > 5:
            # 用抓取前的 JC
            pre_jc = jc[:gp]
        else:
            # 无抓取点 → 用前 30% 帧
            pre_jc = jc[: max(5, len(jc) // 3)]
        valid = pre_jc[pre_jc > 0]
        if len(valid) > 3:
            task_baselines.append(float(np.median(valid)))

    if task_baselines:
        return float(np.median(task_baselines))
    return 3000.0  # safety fallback


# ===================================================================
# 主入口: 分析一个 data_dir
# ===================================================================


def analyze_data_dir(
    data_dir: Path, *, segment_resolver: TaskSegmentResolver | None = None
) -> list[TaskEvidence]:
    """完整分析一个 eval 目录。"""
    from inputs.eval_log import ingest_eval_log

    data_dir = data_dir.resolve()
    segment_assignments = resolve_task_segments(data_dir, segment_resolver)
    frame_seg = segment_assignments.state_frame_to_segment
    if not frame_seg:
        print(
            f"分段识别器未返回 state frame→task segment 映射（{data_dir}）",
            file=sys.stderr,
        )
        return []

    signals = ingest_eval_log(data_dir, frame_seg)
    if not signals:
        print(f"无法读取 eval_log（{data_dir}）", file=sys.stderr)
        return []

    raw_signals = list(signals.values())

    # --- 抓取事件检测（填充 grasp_phase_idx） ---
    for t in raw_signals:
        gp, method = grasp.detect_grasp_event(t)
        t.grasp_phase_idx = gp
        t.grasp_detected_by = method
        if gp is not None:
            grasp.analyze_grasp_signals(t)

    # --- 附加视频语义证据 ---
    from evidence.vision import attach_vision_evidence

    attach_vision_evidence(
        data_dir,
        raw_signals,
        segment_assignments=segment_assignments,
    )

    # 不再从失败任务终点反向“学习”盒子位置，避免循环污染。
    place_center = np.array(V5_RULES.default_place_center)
    place_tol = np.array(
        [V5_RULES.place_roi_xy_m, V5_RULES.place_roi_xy_m, V5_RULES.place_roi_z_m]
    )

    print(
        f"  place_center=({place_center[0]:.4f}, {place_center[1]:.4f}, {place_center[2]:.4f})"
    )
    print(f"  place_tol=({place_tol[0]:.4f}, {place_tol[1]:.4f}, {place_tol[2]:.4f})")

    return [
        judge_task(t, place_center=place_center)
        for t in sorted(raw_signals, key=lambda x: x.task_index)
    ]
