"""夹爪、关节电流和力信号的抓取事件分析。"""

from __future__ import annotations

import numpy as np

from config import (
    GRASP_PHASE_WINDOW,
    GRIP_CLOSE_THRESHOLD,
    GRIP_EMPTY_CLOSED_MAX,
    GRIP_OPEN_THRESHOLD,
    JC_GRASP_RISE_MIN,
    MIN_FRAMES,
)
from models import TaskSignals


def detect_grasp_event(task: TaskSignals) -> tuple[int | None, str | None]:
    """按夹爪反馈、关节电流、轨迹最低点的优先级寻找抓取帧。"""
    ee, joint_current, grip = task.ee_traj, task.jc_traj, task.grip_traj
    n_frames = len(ee)
    if n_frames < MIN_FRAMES:
        return None, None

    valid_mask = ~np.isnan(grip)
    if valid_mask.sum() > 5:
        valid_grip = grip[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        seen_open = valid_grip[0] > GRIP_OPEN_THRESHOLD
        for index in range(1, len(valid_grip)):
            seen_open = seen_open or valid_grip[index - 1] > GRIP_OPEN_THRESHOLD
            if valid_grip[index] < GRIP_CLOSE_THRESHOLD and seen_open:
                return int(valid_indices[index]), "grip"
        # 有可靠夹爪反馈却没有闭合时，不用弱信号伪造抓取事件。
        return None, None

    ignore_frames = 5
    if len(joint_current) > 15:
        baseline_end = max(ignore_frames + 10, n_frames * 2 // 3)
        baseline_window = joint_current[ignore_frames:baseline_end]
        if len(baseline_window) > 5:
            threshold = float(np.median(baseline_window)) + JC_GRASP_RISE_MIN
            above = joint_current >= threshold
            for index in range(ignore_frames, len(above) - 3):
                if above[index] and above[index + 1] and above[index + 2]:
                    return index, "jc"

    if len(ee) and np.ptp(ee[:, 2]) > 0.02:
        return int(np.argmin(ee[:, 2])), "z_min"
    return None, None


def analyze_grasp_signals(task: TaskSignals) -> None:
    """提取抓取窗口中的接触、运输、撤回和释放指标。"""
    ee = task.ee_traj
    joint_current = task.jc_traj
    force_z = task.fz_traj
    grip = task.grip_traj
    grasp_index = task.grasp_phase_idx
    n_frames = len(ee)
    if grasp_index is None:
        return

    if len(task.frame_times) == n_frames and n_frames:
        task.grasp_time_sec = float(
            task.frame_times[grasp_index] - task.frame_times[0]
        )
        task.grasp_timestamp = float(task.frame_times[grasp_index])
    else:
        task.grasp_time_sec = float(grasp_index) / 15.0

    window_start = max(0, grasp_index - 3)
    window_end = min(n_frames, grasp_index + GRASP_PHASE_WINDOW)
    grasp_window = slice(window_start, window_end)

    task.jc_at_grasp = (
        float(joint_current[grasp_window].max()) if len(joint_current) else 0.0
    )
    task.fz_spike_grasp = (
        float(np.max(np.abs(force_z[grasp_window]))) if len(force_z) else 0.0
    )

    pre_current = (
        float(np.mean(joint_current[:window_start]))
        if window_start > 0
        else (float(joint_current[0]) if len(joint_current) else 0.0)
    )
    task.jc_grasp_rise = task.jc_at_grasp - pre_current
    task.jc_pre_close = pre_current
    task.jc_close_rise = task.jc_grasp_rise

    if len(task.jc_joint_traj):
        baseline_end = max(1, window_start)
        baseline = np.median(task.jc_joint_traj[:baseline_end], axis=0)
        task.jc_contact_delta = float(
            np.max(np.abs(task.jc_joint_traj[grasp_window] - baseline))
        )
    else:
        task.jc_contact_delta = max(0.0, task.jc_grasp_rise)

    if len(force_z):
        baseline_end = max(1, window_start)
        baseline = float(np.median(force_z[:baseline_end]))
        task.fz_contact_delta = float(
            np.max(np.abs(force_z[grasp_window] - baseline))
        )

    task.ee_at_grasp = ee[grasp_index].tolist() if grasp_index < n_frames else []
    if grasp_index < n_frames - 1:
        grasp_ee = ee[grasp_index]
        post_ee = ee[grasp_index + 1 :]
        if len(post_ee):
            distances = np.linalg.norm(post_ee - grasp_ee, axis=1)
            task.grasp_transport_m = float(distances.max())
            task.retract_dist_m = float(distances.max())
            task.retract_z_rise_m = float(post_ee[:, 2].max() - grasp_ee[2])
        task.withdraw_home_dist_m = float(np.linalg.norm(ee[-1] - ee[0]))
        task.ee_at_place = ee[-1].tolist()
    elif len(ee):
        task.ee_at_place = ee[-1].tolist()

    detect_grip_release(task)
    if (
        task.grip_release_frame is not None
        and len(task.frame_times) == n_frames
        and task.grip_release_frame < n_frames
    ):
        task.release_timestamp = float(
            task.frame_times[task.grip_release_frame]
        )

    if len(grip):
        closed_end = task.grip_release_frame or min(len(grip), grasp_index + 40)
        closed = grip[grasp_index:closed_end]
        closed = closed[~np.isnan(closed)]
        if len(closed):
            task.grip_closed_min = float(closed.min())
            task.grip_empty_close = (
                task.grip_closed_min <= GRIP_EMPTY_CLOSED_MAX
            )


def detect_grip_release(task: TaskSignals) -> None:
    """寻找抓取后夹爪重新越过张开阈值的稳定释放事件。"""
    grip = task.grip_traj
    grasp_index = task.grasp_phase_idx
    if grasp_index is None or len(grip) < 5:
        return
    if (~np.isnan(grip)).sum() < 5:
        return

    candidates = np.where(
        (grip > GRIP_OPEN_THRESHOLD)
        & (np.arange(len(grip)) > grasp_index)
    )[0]
    if not len(candidates):
        return

    release_index = int(candidates[0])
    held = grip[grasp_index : min(release_index, grasp_index + 3)]
    if np.sum(held < GRIP_CLOSE_THRESHOLD) >= 2:
        task.grip_release_detected = True
        task.grip_release_frame = release_index


def detect_grasp_attempt(task: TaskSignals, has_motion: bool) -> bool:
    """结合抓取来源与靠近幅度，过滤初始化冲击等伪抓取。"""
    if not has_motion or task.grasp_phase_idx is None:
        return False
    if task.grasp_detected_by == "grip":
        return True
    if task.grasp_phase_idx < 10:
        return False

    from config import APPROACH_XY_M, APPROACH_Z_DROP_M

    return (
        task.approach_z_drop_m >= APPROACH_Z_DROP_M * 0.6
        or task.approach_xy_m >= APPROACH_XY_M * 0.6
    )
