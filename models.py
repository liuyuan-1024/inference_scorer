"""
数据结构定义 — TaskSignals（原始信号）和 TaskJudgment（语义判定）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VisionEvidence:
    """单个 task 的轻量视频语义证据。"""

    available: bool = False
    camera: str = ""
    sample_count: int = 0
    scenario: str = "unknown"
    object_present: bool | None = None
    box_present: bool | None = None
    object_detection_rate: float = 0.0
    box_detection_rate: float = 0.0
    object_motion_norm: float = 0.0
    object_motion_after_grasp_norm: float = 0.0
    object_box_distance_drop_norm: float = 0.0
    moved_toward_box: bool = False
    final_object_relation: str = "unknown"
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


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
    frame_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    duration_sec: float = 0.0
    max_ee_excursion_m: float = 0.0
    max_joint_excursion_rad: float = 0.0
    joint_traj: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))

    # --- 力/电流 ---
    fz_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    jc_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    jc_joint_traj: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))
    jc_max: float = 0.0
    jc_contact_delta: float = 0.0
    fz_contact_delta: float = 0.0

    # --- 夹爪 ---
    grip_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    grip_min_val: float = 1.0
    grip_max_val: float = 0.0
    grip_closed_min: float = 1.0
    grip_final_val: float = float("nan")
    grip_empty_close: bool = False

    # --- 抓取相位（由 detector 填充） ---
    grasp_phase_idx: int | None = None
    """抓取发生帧索引（None=未检测到抓取）。"""
    grasp_detected_by: str | None = None
    """抓取检测方式: "grip" / "jc" / "z_min" / None。"""
    grasp_time_sec: float = float("inf")

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
    home_xy_error_m: float = float("inf")
    home_z_error_m: float = float("inf")
    max_home_excursion_m: float = 0.0
    return_progress_m: float = 0.0
    return_duration_sec: float = float("inf")

    # --- 视频语义 ---
    vision: VisionEvidence = field(default_factory=VisionEvidence)


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
    grasp_time_sec: float = float("inf")

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
    home_xy_error_m: float = float("inf")
    home_z_error_m: float = float("inf")
    max_home_excursion_m: float = 0.0
    return_progress_m: float = 0.0
    return_duration_sec: float = float("inf")
    grip_final_val: float = float("nan")
    grip_empty_close: bool = False

    # --- 视频语义与审计 ---
    vision_available: bool = False
    scenario: str = "unknown"
    object_present: bool | None = None
    box_present: bool | None = None
    object_motion_norm: float = 0.0
    final_object_relation: str = "unknown"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
