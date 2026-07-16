"""
信号分析 — 运动学阶段划分、抓取检测、靠近/撤回质量判定。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from config import (
    APPROACH_ALIGN_MIN,
    APPROACH_XY_M,
    APPROACH_Z_DROP_M,
    FAST_APPROACH_FRAC,
    FZ_CONTACT_MIN,
    GRASP_PHASE_WINDOW,
    GRASP_TRANSPORT_M,
    GRIP_CLOSE_THRESHOLD,
    GRIP_OPEN_THRESHOLD,
    GRIP_PRE_CLOSE_MAX,
    GRIP_RELEASE_RISE,
    JC_CONTACT_DELTA,
    JC_GRASP_RISE_MIN,
    MIN_FRAMES,
    MOTION_EE_PATH_M,
    MOTION_JOINT_RAD,
    PATH_EFFICIENCY_MIN,
    PLACE_ROI_XY_M,
    PLACE_ROI_Z_M,
    RETRACT_DIST_M,
    RETRACT_Z_RISE_M,
    WITHDRAW_HOME_M,
    GRIP_EMPTY_CLOSED_MAX
)
from models import TaskJudgment, TaskSignals


# ===================================================================
# 抓取相位检测
# ===================================================================


def detect_grasp_event(t: TaskSignals) -> tuple[int | None, str | None]:
    """
    检测抓取事件发生的帧索引。

    检测策略（按优先级）：
    1. **夹爪闭合** — 检测 grip 从张开(>0.7) → 闭合(<0.5) 的首次跳变
    2. **关节电流尖峰** — 若无夹爪信号，检测 JC 的首次持续尖峰
    3. **轨迹最低点** — fallback，使用 z 最低点

    Returns
    -------
    (grasp_frame_idx, method)
        method: "grip" / "jc" / "z_min" / None
    """
    ee, jc, grip = t.ee_traj, t.jc_traj, t.grip_traj
    n = len(ee)
    if n < MIN_FRAMES:
        return None, None

    # --- 策略 1：夹爪闭合检测 ---
    valid_mask = ~np.isnan(grip)
    if valid_mask.sum() > 5:
        valid_grip = grip[valid_mask]
        valid_idx = np.where(valid_mask)[0]

        # 找首次 grip 从"张开"(>0.7) 下降到"闭合"(<0.5) 的帧并且下降幅度 > 0.3（排除噪声）
        for i in range(len(valid_grip) - 1):
            if (
                valid_grip[i] > GRIP_OPEN_THRESHOLD
                and valid_grip[i + 1] < GRIP_CLOSE_THRESHOLD
                and valid_grip[i] - valid_grip[i + 1] > 0.3
            ):
                frame = valid_idx[i + 1]
                return frame, "grip"

    # --- 策略 2：关节电流尖峰检测 ---
    # 忽略前 5 帧的初始化冲击（机器人复位/启动时 JC 可能短暂偏高）
    JC_IGNORE_FRAMES: int = 5
    if len(jc) > 15:
        # 用中段帧的 JC 中位数作为基线（避开启动和结束）
        mid_start = JC_IGNORE_FRAMES
        mid_end = max(mid_start + 10, n * 2 // 3)
        mid_jc = jc[mid_start:mid_end]
        if len(mid_jc) > 5:
            jc_baseline = float(np.median(mid_jc))
            jc_thresh = jc_baseline + JC_GRASP_RISE_MIN

            # 找首次持续尖峰：连续 3 帧超过阈值（从第 5 帧之后开始找）
            above = jc >= jc_thresh
            for i in range(JC_IGNORE_FRAMES, len(above) - 3):
                if above[i] and above[i + 1] and above[i + 2]:
                    return i, "jc"

    # --- 策略 3：fallback — z 最低点 ---
    if len(ee) > 0 and ee[:, 2].max() - ee[:, 2].min() > 0.02:
        z_min_idx = int(np.argmin(ee[:, 2]))
        return z_min_idx, "z_min"

    return None, None


def analyze_grasp_signals(t: TaskSignals) -> None:
    """
    在已知 grasp_phase_idx 的基础上，提取抓取窗口内的力/电流信号，并检测夹爪释放。
    """
    ee, jc, fz, grip = t.ee_traj, t.jc_traj, t.fz_traj, t.grip_traj
    gp = t.grasp_phase_idx
    n = len(ee)

    if gp is None:
        return

    # --- 抓取窗口 ---
    lo = max(0, gp - 3)
    hi = min(n, gp + GRASP_PHASE_WINDOW)
    win = slice(lo, hi)

    t.jc_at_grasp = float(jc[win].max()) if len(jc) else 0.0
    t.fz_spike_grasp = float(np.max(np.abs(fz[win]))) if len(fz) else 0.0

    # JC 相对上升
    pre_jc = float(np.mean(jc[:lo])) if lo > 0 else (float(jc[0]) if len(jc) else 0.0)
    t.jc_grasp_rise = t.jc_at_grasp - pre_jc
    t.jc_pre_close = pre_jc
    t.jc_close_rise = t.jc_grasp_rise

    # --- 抓取点 EE 位姿 ---
    t.ee_at_grasp = ee[gp].tolist() if gp < n else []

    # --- 运输距离：抓取后 EE 相对于抓取点的最大位移 ---
    if gp < n - 1:
        grasp_ee = ee[gp]
        post_ee = ee[gp + 1 :]
        if len(post_ee):
            dists = np.linalg.norm(post_ee - grasp_ee, axis=1)
            t.grasp_transport_m = float(dists.max())
            t.retract_dist_m = float(dists.max())
            t.retract_z_rise_m = float(post_ee[:, 2].max() - grasp_ee[2])
        t.withdraw_home_dist_m = float(np.linalg.norm(ee[-1] - ee[0]))
        t.ee_at_place = ee[-1].tolist()
    elif len(ee):
        t.ee_at_place = ee[-1].tolist()

    # --- 夹爪释放检测 ---
    _detect_grip_release(t)


def _detect_grip_release(t: TaskSignals) -> None:
    """
    检测夹爪释放：抓取后 grip 从闭合状态张开。
    """
    grip, gp = t.grip_traj, t.grasp_phase_idx
    if gp is None or len(grip) < 5:
        return

    valid_mask = ~np.isnan(grip)
    if valid_mask.sum() < 5:
        return

    # 抓取前的夹爪均值（排除前几帧的初始化动作）
    pre_win = grip[max(3, gp - 10) : gp + 1]
    pre_valid = pre_win[~np.isnan(pre_win)]
    if len(pre_valid) == 0:
        return
    pre_grip = float(pre_valid.mean())

    # 抓取后的夹爪最大值
    post_win = grip[gp : min(len(grip), gp + 40)]
    post_valid = post_win[~np.isnan(post_win)]
    if len(post_valid) == 0:
        return
    post_grip = float(post_valid.max())

    # 释放条件：抓取前处于闭合状态，抓取后明显张开
    if pre_grip < GRIP_PRE_CLOSE_MAX and post_grip > pre_grip + GRIP_RELEASE_RISE:
        # 排除最小值在第 0 帧（初始化张开）
        valid_grip = grip[valid_mask]
        valid_idx = np.where(valid_mask)[0]
        min_valid_idx = int(valid_idx[np.argmin(valid_grip)])
        if min_valid_idx > 2:
            t.grip_release_detected = True
            # 找首次张开到 > 0.5 的帧
            release_candidates = np.where(
                (grip > pre_grip + GRIP_RELEASE_RISE) & (np.arange(len(grip)) >= gp)
            )[0]
            if len(release_candidates):
                t.grip_release_frame = int(release_candidates[0])


# ===================================================================
# 运动学阶段分析
# ===================================================================


def analyze_motion_phases(t: TaskSignals, *, place_center: np.ndarray) -> None:
    """
    填充靠近 / 撤回阶段指标。

    以 grasp_phase_idx 为界：
    - 前半 → 靠近阶段 (approach)
    - 后半 → 撤回阶段 (retract)
    """
    ee = t.ee_traj
    n = len(ee)
    if n < 2:
        return

    grasp_idx = t.grasp_phase_idx
    if grasp_idx is None:
        # 无抓取点 → 把整个轨迹当"靠近"分析，但无法计算撤回
        pre_end = n
    else:
        pre_end = max(1, grasp_idx + 1)

    pre_ee = ee[:pre_end]
    ee_start = pre_ee[0]

    # --- 靠近阶段 ---
    # z 下降量
    z_min = float(pre_ee[:, 2].min())
    t.approach_z_drop_m = float(ee_start[2] - z_min)

    # z 最低点
    min_z_idx = int(np.argmin(pre_ee[:, 2]))
    ee_at_min_z = pre_ee[min_z_idx]

    # xy 位移
    disp_xy = ee_at_min_z[:2] - ee_start[:2]
    t.approach_xy_m = float(np.linalg.norm(disp_xy))

    # 方向对齐度量
    if t.approach_xy_m > 0.01:
        # 用 xy 位移方向作为对齐的代理（首帧→z最低点的方向）
        t.approach_align = min(1.0, t.approach_xy_m / 0.10)
    else:
        t.approach_align = 0.0

    # 路径效率
    path_len = float(np.linalg.norm(np.diff(pre_ee, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(ee_at_min_z - ee_start))
    t.path_efficiency = straight / path_len if path_len > 1e-6 else 0.0
    t.approach_frame_frac = min_z_idx / max(pre_end - 1, 1)

    # --- 撤回阶段 ---
    if grasp_idx is not None and grasp_idx < n - 1:
        grasp_ee = ee[grasp_idx]
        post_ee = ee[grasp_idx + 1 :]

        dists = np.linalg.norm(post_ee - grasp_ee, axis=1)
        t.retract_dist_m = float(dists.max()) if len(dists) else 0.0
        t.retract_z_rise_m = (
            float(post_ee[:, 2].max() - grasp_ee[2]) if len(post_ee) else 0.0
        )
        t.withdraw_home_dist_m = float(np.linalg.norm(ee[-1] - ee_start))

        # 运输方向对齐放置 ROI
        if len(post_ee) >= 2:
            to_place = place_center[:2] - grasp_ee[:2]
            post_disp = post_ee[-1][:2] - grasp_ee[:2]
            if np.linalg.norm(to_place) > 0.03 and np.linalg.norm(post_disp) > 0.04:
                align_place = float(
                    np.dot(post_disp, to_place)
                    / (np.linalg.norm(post_disp) * np.linalg.norm(to_place))
                )
                t.transport_to_place = (
                    align_place > 0.5 and t.retract_dist_m >= RETRACT_DIST_M * 0.8
                )


# ===================================================================
# 运动/靠近/抓取 判定
# ===================================================================


def detect_motion(t: TaskSignals) -> bool:
    """判定机械臂是否有效移动。"""
    if t.ee_path_m >= MOTION_EE_PATH_M:
        return True
    if t.joint_delta_rad >= MOTION_JOINT_RAD:
        return True
    return False


def classify_approach(t: TaskSignals, has_motion: bool) -> tuple[bool, str]:
    """分类靠近质量。返回 (has_approach, quality_label)。"""
    if not has_motion:
        return False, "none"

    z_ok = t.approach_z_drop_m >= APPROACH_Z_DROP_M
    xy_ok = t.approach_xy_m >= APPROACH_XY_M
    align_ok = t.approach_align >= APPROACH_ALIGN_MIN
    eff_ok = t.path_efficiency >= PATH_EFFICIENCY_MIN
    has_approach = z_ok and xy_ok and align_ok and eff_ok

    if has_approach:
        if t.approach_frame_frac <= FAST_APPROACH_FRAC and eff_ok:
            return True, "fast"
        return True, "slow"

    if has_motion and (not align_ok or not eff_ok):
        return False, "wander"
    return False, "none"


def detect_grasp_attempt(t: TaskSignals, has_motion: bool) -> bool:
    """
    运动学+语义抓取判定。

    有抓取帧索引（grasp_phase_idx 非 None）且满足基本运动条件即为尝试。
    """
    if not has_motion or t.grasp_phase_idx is None:
        return False

    # 夹爪闭合检测到的抓取 → 总是可信
    if t.grasp_detected_by == "grip":
        return True

    # JC/z_min 检测到的抓取 → 需要额外的运动学验证
    # 如果抓取点太靠前(<10帧)，很可能是初始化冲击
    if t.grasp_phase_idx < 10:
        return False

    z_ok = t.approach_z_drop_m >= APPROACH_Z_DROP_M * 0.6
    xy_ok = t.approach_xy_m >= APPROACH_XY_M * 0.6
    return z_ok or xy_ok


# ===================================================================
# Trial 综合判定
# ===================================================================


def judge_task_simple(
    t: TaskSignals, *, place_center: np.ndarray, jc_baseline: float
) -> TaskJudgment:
    """对单个 task 做综合语义判定（不依赖首帧视频）。"""
    analyze_motion_phases(t, place_center=place_center)

    has_motion = detect_motion(t)
    has_grasp_attempt = detect_grasp_attempt(t, has_motion)

    # --- 接触 / 抓取判定 ---
    abs_ok = t.jc_at_grasp >= jc_baseline + JC_CONTACT_DELTA
    rel_ok = t.jc_grasp_rise >= JC_GRASP_RISE_MIN
    fz_ok = t.fz_spike_grasp >= FZ_CONTACT_MIN
    transport_ok = t.grasp_transport_m >= GRASP_TRANSPORT_M
    has_load = abs_ok or rel_ok or fz_ok
    
    is_empty_grasp = False
    if t.grip_min_val < GRIP_EMPTY_CLOSED_MAX:
        is_empty_grasp = True

    if not has_grasp_attempt:
        has_grasp_contact = False
        has_grasp_object = False
    else:
        has_grasp_contact = has_load
        # 只要夹空了，不管电流多大，都认为 object = False
        has_grasp_object = (has_load or (transport_ok and has_load)) and not is_empty_grasp
    """""  
    if not has_grasp_attempt:
        has_grasp_contact = False
        has_grasp_object = False
    else:
        has_grasp_contact = has_load
        has_grasp_object = has_load or (transport_ok and has_load)
    """
    # --- 放置相位 ---
    has_place_phase = False
    place_at_box = False
    if has_grasp_object:
        has_place_phase = (
            t.grasp_transport_m >= 0.03
            or t.retract_dist_m >= RETRACT_DIST_M * 0.5
            or t.transport_to_place
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
    _, approach_quality = classify_approach(t, has_motion)

    # --- 撤回分类 ---
    has_retract = (
        t.retract_dist_m >= RETRACT_DIST_M or t.retract_z_rise_m >= RETRACT_Z_RISE_M
    )
    has_withdraw = t.withdraw_home_dist_m <= WITHDRAW_HOME_M

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

    reason = "；".join(parts) if parts else "信号不足"

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
        ee_path_m=t.ee_path_m,
        joint_delta_rad=t.joint_delta_rad,
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
    )


# ===================================================================
# 工具: 学习 place_center 和 jc_baseline
# ===================================================================


def learn_place_center(signals: list[TaskSignals]) -> np.ndarray:
    """
    从各 task 的抓取后末端位置学习放置中心。

    取有负载运输的 task 的终点位置的中位数。
    """
    from config import GRASP_TRANSPORT_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M

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
        gp, method = detect_grasp_event(t)
        t.grasp_phase_idx = gp
        t.grasp_detected_by = method
        if gp is not None:
            analyze_grasp_signals(t)

    # --- 学习放置 ROI ---
    place_center = learn_place_center(raw_signals)
    place_tol = learn_place_tolerance(raw_signals, place_center)

    # --- 学习 JC 基线 ---
    jc_baseline = learn_jc_baseline(raw_signals)

    print(
        f"  place_center=({place_center[0]:.4f}, {place_center[1]:.4f}, {place_center[2]:.4f})"
    )
    print(f"  place_tol=({place_tol[0]:.4f}, {place_tol[1]:.4f}, {place_tol[2]:.4f})")
    print(f"  jc_baseline={jc_baseline:.0f}")

    return [
        judge_task_simple(t, place_center=place_center, jc_baseline=jc_baseline)
        for t in sorted(raw_signals, key=lambda x: x.task_index)
    ]
