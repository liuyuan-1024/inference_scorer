"""单次 task 的语义判定与数据目录分析入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from config import (
    PLACE_ROI_XY_M,
    PLACE_ROI_Z_M,
    RETURN_MIN_EXCURSION_M,
    RETURN_MIN_PROGRESS_M,
    RETRACT_DIST_M,
    RETRACT_Z_RISE_M,
    WITHDRAW_HOME_XY_M,
    WITHDRAW_HOME_Z_M,
)
from events import build_action_event_chain, fused_confidence
import grasp_analysis
from models import TaskJudgment, TaskSignals
import motion_analysis

__all__ = [
    "analyze_data_dir",
    "judge_task",
]


# ===================================================================
# Task 综合判定
# ===================================================================


def judge_task(
    t: TaskSignals, *, place_center: np.ndarray, jc_baseline: float | None = None
) -> TaskJudgment:
    """对单个 task 做综合语义判定（不依赖首帧视频）。"""
    motion_analysis.analyze_motion_phases(t, place_center=place_center)

    has_motion = motion_analysis.detect_motion(t)
    has_grasp_attempt = grasp_analysis.detect_grasp_attempt(t, has_motion)

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
                t.grasp_transport_m >= 0.03
                or t.retract_dist_m >= RETRACT_DIST_M * 0.5
                or t.transport_to_place
            )
        )
        if has_place_phase and t.ee_at_place:
            ee = np.array(t.ee_at_place)
            place_at_box = bool(
                np.all(
                    np.abs(ee - place_center)
                    <= np.array([PLACE_ROI_XY_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M])
                )
            )

    # --- 动作完成 ---
    action_complete = has_motion and has_grasp_object and has_place_phase

    # --- 靠近质量 ---
    _, approach_quality = motion_analysis.classify_approach(t, has_motion)

    # --- 撤回分类 ---
    has_retract = (
        t.retract_dist_m >= RETRACT_DIST_M or t.retract_z_rise_m >= RETRACT_Z_RISE_M
    )
    has_withdraw = (
        t.max_home_excursion_m >= RETURN_MIN_EXCURSION_M
        and t.return_progress_m >= RETURN_MIN_PROGRESS_M
        and t.home_xy_error_m <= WITHDRAW_HOME_XY_M
        and t.home_z_error_m <= WITHDRAW_HOME_Z_M
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
    if t.grip_empty_close:
        evidence.append(f"夹爪闭合至{t.grip_closed_min:.3f}，判为空夹")
    if t.grasp_phase_idx is not None:
        evidence.append(
            f"抓取候选帧={t.grasp_phase_idx}({t.grasp_detected_by or '?'})"
        )
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
    confidence, vision_confidence, sensor_confidence = fused_confidence(t, events)
    if t.grip_empty_close or (t.vision.available and t.vision.object_present is False):
        confidence = max(confidence, min(vision_confidence, 0.88))

    return TaskJudgment(
        task_index=t.task_index,
        has_motion=has_motion,
        has_grasp_attempt=has_grasp_attempt,
        has_grasp_contact=has_grasp_contact,
        has_grasp_object=has_grasp_object,
        has_place_phase=has_place_phase,
        place_at_box=place_at_box,
        action_complete=action_complete,
        approach_quality=approach_quality,
        retract_semantic=retract_semantic,
        has_retract=has_retract,
        has_withdraw_home=has_withdraw,
        n_frames=t.n_frames,
        reason=reason,
        grip_release_detected=t.grip_release_detected,
        grasp_time_sec=t.grasp_time_sec,
        ee_path_m=t.ee_path_m,
        joint_delta_rad=t.joint_delta_rad,
        task_start_ee=t.task_start_ee,
        task_start_frame_index=t.task_start_frame_index,
        task_start_timestamp=t.task_start_timestamp,
        approach_z_drop_m=t.approach_z_drop_m,
        approach_xy_m=t.approach_xy_m,
        approach_align=t.approach_align,
        path_efficiency=t.path_efficiency,
        approach_frame_frac=t.approach_frame_frac,
        jc_at_grasp=t.jc_at_grasp,
        jc_grasp_rise=t.jc_grasp_rise,
        fz_spike_grasp=t.fz_spike_grasp,
        grasp_transport_m=t.grasp_transport_m,
        retract_dist_m=t.retract_dist_m,
        retract_z_rise_m=t.retract_z_rise_m,
        withdraw_home_dist_m=t.withdraw_home_dist_m,
        transport_to_place=t.transport_to_place,
        home_xy_error_m=t.home_xy_error_m,
        home_z_error_m=t.home_z_error_m,
        max_home_excursion_m=t.max_home_excursion_m,
        return_progress_m=t.return_progress_m,
        return_duration_sec=t.return_duration_sec,
        grip_final_val=t.grip_final_val,
        grip_empty_close=t.grip_empty_close,
        object_lifted=events.object_lifted,
        object_dropped=events.object_dropped,
        vision_available=t.vision.available,
        scenario=t.vision.scenario,
        object_present=t.vision.object_present,
        box_present=t.vision.box_present,
        object_motion_norm=t.vision.object_motion_after_grasp_norm,
        final_object_relation=t.vision.final_object_relation,
        confidence=confidence,
        vision_confidence=vision_confidence,
        sensor_confidence=sensor_confidence,
        state_feedback_timeout_rate=t.state_feedback_timeout_rate,
        events=events,
        evidence=evidence,
    )


# ===================================================================
# 工具: 学习 place_center 和 jc_baseline
# ===================================================================


def learn_place_center(signals: list[TaskSignals]) -> np.ndarray:
    """
    从各 task 的抓取后末端位置学习放置中心。

    取有负载运输的 task 的终点位置的中位数。
    """
    from config import GRASP_TRANSPORT_M

    place_ee = []
    for t in signals:
        ee = t.ee_at_place
        if not ee:
            continue
        if t.grasp_transport_m >= GRASP_TRANSPORT_M * 0.5:
            place_ee.append(ee)

    if place_ee:
        arr = np.array(place_ee)
        return np.median(arr, axis=0)

    from config import DEFAULT_PLACE_CENTER

    return np.array(DEFAULT_PLACE_CENTER)


def learn_place_tolerance(
    signals: list[TaskSignals], place_center: np.ndarray
) -> np.ndarray:
    """学习放置容差。"""
    from config import PLACE_ROI_XY_M, PLACE_ROI_Z_M

    diffs = []
    for t in signals:
        ee = t.ee_at_place
        if not ee or t.grasp_transport_m < 0.03:
            continue
        diffs.append(np.abs(np.array(ee) - place_center))

    if diffs:
        arr = np.array(diffs)
        spread = np.std(arr, axis=0)
        return np.maximum(spread * 2.5, [PLACE_ROI_XY_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M])

    return np.array([PLACE_ROI_XY_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M])


def learn_jc_baseline(signals: list[TaskSignals]) -> float:
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


def analyze_data_dir(data_dir: Path) -> list[TaskJudgment]:
    """完整分析一个 eval 目录。"""
    from loader import load_frame_to_segment, ingest_eval_log

    data_dir = data_dir.resolve()
    frame_seg = load_frame_to_segment(data_dir)
    if not frame_seg:
        print(f"无法读取 frame→segment 映射（{data_dir}）", file=sys.stderr)
        return []

    signals = ingest_eval_log(data_dir, frame_seg)
    if not signals:
        print(f"无法读取 eval_log（{data_dir}）", file=sys.stderr)
        return []

    raw_signals = list(signals.values())

    # --- 抓取事件检测（填充 grasp_phase_idx） ---
    for t in raw_signals:
        gp, method = grasp_analysis.detect_grasp_event(t)
        t.grasp_phase_idx = gp
        t.grasp_detected_by = method
        if gp is not None:
            grasp_analysis.analyze_grasp_signals(t)

    # --- 附加视频语义证据 ---
    from vision import attach_vision_evidence

    attach_vision_evidence(data_dir, raw_signals)

    # 不再从失败任务终点反向“学习”盒子位置，避免循环污染。
    from config import DEFAULT_PLACE_CENTER

    place_center = np.array(DEFAULT_PLACE_CENTER)
    place_tol = np.array([PLACE_ROI_XY_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M])

    print(
        f"  place_center=({place_center[0]:.4f}, {place_center[1]:.4f}, {place_center[2]:.4f})"
    )
    print(f"  place_tol=({place_tol[0]:.4f}, {place_tol[1]:.4f}, {place_tol[2]:.4f})")

    return [
        judge_task(t, place_center=place_center)
        for t in sorted(raw_signals, key=lambda x: x.task_index)
    ]
