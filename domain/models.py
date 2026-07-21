"""领域数据结构：原始信号、统一评分证据和阶段结果。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VisionEvidence:
    """单个 task 的多视角视频语义证据。"""

    available: bool = False
    camera: str = ""
    cameras: list[str] = field(default_factory=list)
    chest_available: bool = False
    wrist_available: bool = False
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
    final_inside_ratio: float = 0.0
    object_settled_after_release: bool = False
    wrist_object_near_gripper: bool = False
    wrist_object_retained: bool = False
    wrist_drop_detected: bool = False
    wrist_observation_rate: float = 0.0
    wrist_acquire_rate: float = 0.0
    wrist_early_hold_rate: float = 0.0
    wrist_late_hold_rate: float = 0.0
    wrist_approach_drop_norm: float = 0.0
    wrist_min_object_gripper_norm: float = float("inf")
    temporal_alignment_sec: float = float("inf")
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class ActionEventChain:
    """融合传感器和多视角视频得到的可审计动作事件链。"""

    close_detected: bool = False
    contact_detected: bool = False
    object_between_jaws: bool = False
    object_acquired: bool = False
    object_lifted: bool = False
    object_retained: bool = False
    object_dropped: bool = False
    transported: bool = False
    reached_box: bool = False
    release_detected: bool = False
    object_settled: bool = False
    placed_inside: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class TaskEvidence:
    """单个 task 从原始测量到评分语义的唯一证据对象。"""

    task_index: int
    n_frames: int = 0

    # --- 运动学 ---
    ee_path_m: float = 0.0
    joint_delta_rad: float = 0.0
    task_start_ee: list[float] = field(default_factory=list)
    task_start_frame_index: int | None = None
    task_start_timestamp: float | None = None
    ee_traj: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    frame_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    frame_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    image_times: np.ndarray = field(default_factory=lambda: np.empty(0))
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
    grip_fully_closed: bool = False
    grip_feedback_age_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    grip_feedback_received_traj: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    grip_observation_source: str = ""
    desired_grip_traj: np.ndarray = field(default_factory=lambda: np.empty(0))

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
    grasp_timestamp: float | None = None
    release_timestamp: float | None = None

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

    # --- 数据质量 / 执行反馈 ---
    state_feedback_timeout_rate: float = 0.0
    response_tracking_error_p95: float = 0.0
    image_state_diff_p95_sec: float = float("inf")
    gripper_udp_age_p95_sec: float = float("inf")
    sensor_confidence: float = 0.0

    # --- 视频语义 ---
    vision: VisionEvidence = field(default_factory=VisionEvidence)

    # --- 融合事件与评分语义（由 analysis 填充） ---
    events: ActionEventChain = field(default_factory=ActionEventChain)
    has_motion: bool = False
    has_grasp_attempt: bool = False
    has_grasp_contact: bool = False
    has_grasp_object: bool = False
    has_place_phase: bool = False
    place_at_box: bool = False
    action_complete: bool = False
    approach_quality: str = "none"
    retract_semantic: str = "none"
    has_retract: bool = False
    has_withdraw_home: bool = False
    reason: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StageResult:
    """一个评分阶段的完整、可审计输出。"""

    stage: str
    score: int
    max_score: int
    confidence: float
    needs_review: bool
    evidence: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    review_reason: str | None = None
    rule_version: str = "v5"

    def to_dict(self) -> dict:
        result = {
            "score": self.score,
            "max_score": self.max_score,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "evidence": list(self.evidence),
            "missing_evidence": list(self.missing_evidence),
            "rule_version": self.rule_version,
        }
        if self.review_reason:
            result["review_reason"] = self.review_reason
        return result


@dataclass(frozen=True)
class TaskScore:
    """一个 task 的五阶段评分结果。"""

    task_index: int
    stages: dict[str, StageResult]

    @property
    def total(self) -> int:
        return sum(stage.score for stage in self.stages.values())


@dataclass(frozen=True)
class ScoringRun:
    """与 Excel、JSON 等输出格式无关的一次评分运行。"""

    data_dir: str
    evidence: tuple[TaskEvidence, ...]
    tasks: dict[int, TaskScore]
    rule_version: str = "v5"
