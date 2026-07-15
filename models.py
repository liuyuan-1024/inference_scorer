"""
数据结构定义 — TaskSignals（原始信号）和 TaskJudgment（语义判定）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TaskSignals:
    """单个 task 的原始传感器信号。"""

    task_index: int
    n_frames: int = 0

    # --- 运动学 ---
    ee_path_m: float = 0.0
    joint_delta_rad: float = 0.0
    ee_start: list[float] = field(default_factory=list)
    ee_traj: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    # --- 力/电流 ---
    fz_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    jc_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    jc_max: float = 0.0

    # --- 夹爪 ---
    grip_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    grip_min_val: float = 1.0
    grip_max_val: float = 0.0

    # --- 抓取相位（由 detector 填充） ---
    grasp_phase_idx: int | None = None
    """抓取发生帧索引（None=未检测到抓取）。"""
    grasp_detected_by: str | None = None
    """抓取检测方式: "grip" / "jc" / "z_min" / None。"""

    # --- 抓取窗口信号 ---
    jc_at_grasp: float = 0.0
    fz_spike_grasp: float = 0.0
    jc_grasp_rise: float = 0.0
    jc_pre_close: float = 0.0
    jc_close_rise: float = 0.0

    # --- 运输/放置 ---
    grasp_transport_m: float = 0.0
    ee_at_grasp: list[float] = field(default_factory=list)
    ee_at_place: list[float] = field(default_factory=list)

    # --- 夹爪释放检测 ---
    grip_release_detected: bool = False
    grip_release_frame: int | None = None

    # --- 阶段指标（由 analyze_motion_phases 填充） ---
    approach_align: float = 0.0
    approach_z_drop_m: float = 0.0
    approach_xy_m: float = 0.0
    path_efficiency: float = 0.0
    approach_frame_frac: float = 1.0
    retract_dist_m: float = 0.0
    retract_z_rise_m: float = 0.0
    withdraw_home_dist_m: float = float("inf")
    transport_to_place: bool = False


@dataclass
class TaskJudgment:
    """单个 task 的语义判定结果。"""

    task_index: int
    has_motion: bool
    has_grasp_attempt: bool
    has_grasp_contact: bool
    has_grasp_object: bool
    has_place_phase: bool
    place_at_box: bool
    action_complete: bool
    approach_quality: str  # "none" / "wander" / "slow" / "fast"
    retract_semantic: str  # "none" / "lift_only" / "withdraw_home" / "transport_place"
    has_retract: bool
    has_withdraw_home: bool
    n_frames: int
    reason: str

    # --- 夹爪释放 ---
    grip_release_detected: bool = False

    # --- 原始指标（用于评分参考） ---
    ee_path_m: float = 0.0
    joint_delta_rad: float = 0.0
    approach_z_drop_m: float = 0.0
    approach_xy_m: float = 0.0
    approach_align: float = 0.0
    path_efficiency: float = 0.0
    approach_frame_frac: float = 1.0
    jc_at_grasp: float = 0.0
    jc_grasp_rise: float = 0.0
    fz_spike_grasp: float = 0.0
    grasp_transport_m: float = 0.0
    retract_dist_m: float = 0.0
    retract_z_rise_m: float = 0.0
    withdraw_home_dist_m: float = float("inf")
    transport_to_place: bool = False
