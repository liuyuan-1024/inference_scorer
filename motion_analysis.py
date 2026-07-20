"""末端轨迹的靠近、运输、撤回和归位分析。"""

from __future__ import annotations

import numpy as np

from config import (
    APPROACH_ALIGN_MIN,
    APPROACH_XY_M,
    APPROACH_Z_DROP_M,
    FAST_APPROACH_FRAC,
    MOTION_EE_EXCURSION_M,
    MOTION_JOINT_EXCURSION_RAD,
    PATH_EFFICIENCY_MIN,
    RETRACT_DIST_M,
    VISION_WRIST_APPROACH_DROP_NORM,
    VISION_WRIST_NEAR_GRIPPER_NORM,
)
from models import TaskSignals


def analyze_motion_phases(
    task: TaskSignals, *, place_center: np.ndarray
) -> None:
    """以抓取帧为界，填充靠近、运输、撤回和归位指标。"""
    ee = task.ee_traj
    n_frames = len(ee)
    if n_frames < 2:
        return

    grasp_index = task.grasp_phase_idx
    approach_end = n_frames if grasp_index is None else max(1, grasp_index + 1)
    approach_ee = ee[:approach_end]
    start_ee = approach_ee[0]

    min_z_index = int(np.argmin(approach_ee[:, 2]))
    min_z_ee = approach_ee[min_z_index]
    task.approach_z_drop_m = float(start_ee[2] - approach_ee[:, 2].min())
    task.approach_xy_m = float(np.linalg.norm(min_z_ee[:2] - start_ee[:2]))

    path_length = float(
        np.linalg.norm(np.diff(approach_ee, axis=0), axis=1).sum()
    )
    straight_distance = float(np.linalg.norm(min_z_ee - start_ee))
    task.path_efficiency = (
        straight_distance / path_length if path_length > 1e-6 else 0.0
    )
    # 未标定物体三维坐标时，路径直线度仅作为保守代理。
    task.approach_align = task.path_efficiency
    task.approach_frame_frac = min_z_index / max(approach_end - 1, 1)

    # 归位原点始终是当前 task 排序后的第一帧，不能跨 task 共用。
    task_origin = ee[0].copy()
    task.task_start_ee = task_origin.tolist()
    if task.task_start_frame_index is None and len(task.frame_indices):
        task.task_start_frame_index = int(task.frame_indices[0])
    if task.task_start_timestamp is None and len(task.frame_times):
        task.task_start_timestamp = float(task.frame_times[0])

    home_delta = ee - task_origin
    home_distances = np.linalg.norm(home_delta, axis=1)
    max_home_index = int(np.argmax(home_distances))
    task.max_home_excursion_m = float(home_distances[max_home_index])
    task.home_xy_error_m = float(np.linalg.norm(home_delta[-1, :2]))
    task.home_z_error_m = float(abs(home_delta[-1, 2]))
    final_home_distance = float(home_distances[-1])
    task.return_progress_m = max(
        0.0, task.max_home_excursion_m - final_home_distance
    )
    if max_home_index < n_frames - 1:
        if len(task.frame_times) == n_frames:
            task.return_duration_sec = float(
                task.frame_times[-1] - task.frame_times[max_home_index]
            )
        else:
            task.return_duration_sec = float(
                n_frames - 1 - max_home_index
            ) / 15.0
    task.withdraw_home_dist_m = final_home_distance

    if grasp_index is None or grasp_index >= n_frames - 1:
        return

    grasp_ee = ee[grasp_index]
    post_ee = ee[grasp_index + 1 :]
    distances = np.linalg.norm(post_ee - grasp_ee, axis=1)
    task.retract_dist_m = float(distances.max()) if len(distances) else 0.0
    task.retract_z_rise_m = (
        float(post_ee[:, 2].max() - grasp_ee[2]) if len(post_ee) else 0.0
    )

    if len(post_ee) >= 2:
        to_place = place_center[:2] - grasp_ee[:2]
        transported = post_ee[-1][:2] - grasp_ee[:2]
        if np.linalg.norm(to_place) > 0.03 and np.linalg.norm(transported) > 0.04:
            alignment = float(
                np.dot(transported, to_place)
                / (np.linalg.norm(transported) * np.linalg.norm(to_place))
            )
            task.transport_to_place = (
                alignment > 0.5
                and task.retract_dist_m >= RETRACT_DIST_M * 0.8
            )


def detect_motion(task: TaskSignals) -> bool:
    """判断机械臂是否产生了足够的末端或关节位移。"""
    return (
        task.max_ee_excursion_m >= MOTION_EE_EXCURSION_M
        or task.max_joint_excursion_rad >= MOTION_JOINT_EXCURSION_RAD
    )


def classify_approach(
    task: TaskSignals, has_motion: bool
) -> tuple[bool, str]:
    """返回是否真正靠近目标，以及 fast / slow / wander / none。"""
    if not has_motion:
        return False, "none"

    if task.vision.wrist_available and task.vision.object_present is not False:
        visually_directed = (
            task.vision.wrist_approach_drop_norm
            >= VISION_WRIST_APPROACH_DROP_NORM
        )
        visually_near = (
            task.vision.wrist_min_object_gripper_norm
            <= VISION_WRIST_NEAR_GRIPPER_NORM
        )
        has_approach = visually_directed or (
            visually_near and task.grasp_phase_idx is not None
        )
        if has_approach:
            if (
                task.grasp_time_sec <= 5.0
                and task.path_efficiency >= PATH_EFFICIENCY_MIN
            ):
                return True, "fast"
            return True, "slow"
        return False, "wander"

    z_ok = task.approach_z_drop_m >= APPROACH_Z_DROP_M
    xy_ok = task.approach_xy_m >= APPROACH_XY_M
    align_ok = task.approach_align >= APPROACH_ALIGN_MIN
    efficient = task.path_efficiency >= PATH_EFFICIENCY_MIN
    has_approach = z_ok and xy_ok and align_ok and efficient
    if has_approach:
        if task.approach_frame_frac <= FAST_APPROACH_FRAC:
            return True, "fast"
        return True, "slow"
    if not align_ok or not efficient:
        return False, "wander"
    return False, "none"
