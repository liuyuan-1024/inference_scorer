"""
评分映射 — 将 TaskJudgment 语义判定映射为 Excel 分值。

每个评分函数严格对应一个维度 (S1~S5)，输入 TaskJudgment，输出 int。
"""

from __future__ import annotations

import numpy as np

from config import PLACE_ROI_XY_M, PLACE_ROI_Z_M, WITHDRAW_HOME_M, RETRACT_DIST_M
from models import TaskJudgment


def approach_frame_count(t: TaskJudgment, fps: float = 15.0) -> float:
    """估算靠近阶段所用时间（秒）。"""
    if t.n_frames < 2:
        return 999.0
    pre_grasp_frames = t.n_frames
    return t.approach_frame_frac * pre_grasp_frames / fps


def score_S1_positioning(t: TaskJudgment, fps: float = 15.0) -> int:
    """
    S1定位评分 (0-4):
      0 = 10s内几乎没有动作
      1 = 无规律乱动
      2 = 定位偏移
      3 = 缓慢靠近目标（5-10s）
      4 = 5s内快速准确定位
    """
    if not t.has_motion:
        return 0
    if t.ee_path_m < 0.05 and t.joint_delta_rad < 0.3:
        return 0

    if t.approach_quality == "fast":
        return 4
    if t.approach_quality == "slow":
        approach_time = approach_frame_count(t, fps)
        return 3 if approach_time <= 10.0 else 2
    if t.approach_quality == "wander":
        return 1

    # approach_quality == "none"
    # 有部分靠近迹象但不足
    if t.approach_z_drop_m >= 0.02 and t.approach_xy_m >= 0.03:
        return 2
    if t.ee_path_m >= 0.10:
        return 1
    return 0


def score_S2_grasping(t: TaskJudgment) -> int:
    """
    S2抓取评分 (0-3):
      0 = 未抓取
      1 = 抓取失败（抓空）
      2 = 抬起掉落
      3 = 稳定抓取
    """
    if not t.has_grasp_attempt:
        return 0
    if not t.has_grasp_contact:
        return 1
    if t.has_grasp_object:
        if not t.has_place_phase and t.grasp_transport_m < 0.08:
            return 2
        return 3
    else:
        if t.grasp_transport_m < 0.03:
            return 2
        return 1


def score_S3_transport(t: TaskJudgment) -> int:
    """
    S3搬运评分 (0-2):
      0 = 未搬运
      1 = 搬运未到位
      2 = 到达目标上方
    """
    if not t.has_grasp_object:
        return 0
    if t.grasp_transport_m < 0.05:
        return 0
    if t.transport_to_place or (t.retract_dist_m >= 0.10 and t.has_place_phase):
        return 2
    return 1


def _distance_to_box_edge(ee_pos: list[float], place_center: list[float]) -> float:
    """
    计算 EE 末端到放置 ROI 边缘的最短距离。

    盒子的 ROIs 范围:
      x: [center_x - tol_x, center_x + tol_x]
      y: [center_y - tol_y, center_y + tol_y]
      z: [center_z - tol_z, center_z + tol_z]

    返回距离（米），负值表示在盒内。
    """
    center = np.array(place_center[:3])
    pos = np.array(ee_pos[:3])
    tols = np.array([PLACE_ROI_XY_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M])

    # 各轴距离（带符号）
    half_dist = np.abs(pos - center) - tols
    # 在盒内的轴 half_dist <= 0
    if np.all(half_dist <= 0):
        return -float(np.linalg.norm(np.maximum(half_dist, 0)))  # 在盒内，负值
    return float(np.linalg.norm(np.maximum(half_dist, 0)))


def score_S4_placing(t: TaskJudgment) -> int:
    """
    S4投放评分 (0-3):
      0 = 无投放
      1 = 落于盒外（距 ROI > 2cm）
      2 = 落于盒边（距 ROI ≤ 2cm）
      3 = 准确入盒
    """
    has_release = t.grip_release_detected

    if not has_release and not t.has_place_phase:
        return 0

    if t.place_at_box:
        return 3

    # 有放置动作但不在 ROI 内 → 区分盒边/盒外
    # 如果 place_at_box 为 False，但 EE 末端距离 ROI 边缘较近 → 盒边
    if has_release and t.retract_dist_m > 0:
        # 尝试用抓取点/终点的位置推断距盒距离
        # 这里简化：用 has_place_phase 和 transport 判断
        if t.transport_to_place and t.retract_dist_m >= 0.08:
            return 2  # 运到附近但未精确入盒
    return 1


def score_S5_return(t: TaskJudgment) -> int:
    """
    S5归位评分 (0-2):
      0 = 未归位
      1 = 归位偏差
      2 = 成功归位
    """
    if t.withdraw_home_dist_m <= WITHDRAW_HOME_M:
        return 2
    if not t.has_retract:
        if t.withdraw_home_dist_m < 0.20:
            return 1
        return 0
    if t.withdraw_home_dist_m < 0.20:
        return 1
    return 1 if t.retract_dist_m >= RETRACT_DIST_M else 0


def compute_all_scores(
    judgments: list[TaskJudgment], fps: float = 15.0
) -> dict[int, dict[str, int]]:
    """对所有 task 计算各维度评分。"""
    results: dict[int, dict[str, int]] = {}
    for j in judgments:
        results[j.task_index] = {
            "S1定位": score_S1_positioning(j, fps),
            "S2抓取": score_S2_grasping(j),
            "S3搬运": score_S3_transport(j),
            "S4投放": score_S4_placing(j),
            "S5归位": score_S5_return(j),
        }
    return results
