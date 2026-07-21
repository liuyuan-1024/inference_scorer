"""机械臂抓取投放评分规则 v5 的唯一事实来源。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.models import TaskEvidence


@dataclass(frozen=True)
class ConfidencePolicy:
    sensor_weight: float
    vision_weight: float
    missing_vision_penalty: float
    max_confidence: float
    review_threshold: float = 0.70
    feedback_timeout_threshold: float = 0.20
    feedback_penalty: float = 0.08
    sync_penalty: float = 0.08
    modality_conflict_penalty: float = 0.0
    unknown_relation_penalty: float = 0.0


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    max_score: int
    modalities: tuple[str, ...]
    confidence: ConfidencePolicy


@dataclass(frozen=True)
class RuleDecision:
    score: int
    strength: float
    reason: str
    review_reason: str | None = None
    missing_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class V5RuleSet:
    version: str = "v5"

    # 运动与接近
    motion_joint_excursion_rad: float = 0.017453292519943295
    motion_ee_excursion_m: float = 0.01
    approach_z_drop_m: float = 0.04
    approach_xy_m: float = 0.05
    approach_align_min: float = 0.30
    path_efficiency_min: float = 0.30
    fast_approach_frac: float = 0.40
    positioning_fast_sec: float = 5.0
    positioning_slow_sec: float = 10.0

    # 抓取、接触和搬运
    grasp_phase_window: int = 8
    fz_contact_delta_min: float = 0.8
    jc_contact_delta: int = 800
    jc_grasp_rise_min: float = 900.0
    grasp_transport_m: float = 0.06
    grasp_min_lift_m: float = 0.02
    grasp_lift_m: float = 0.05
    transport_partial_m: float = 0.05
    transport_direction_target_min_m: float = 0.03
    transport_direction_motion_min_m: float = 0.04
    transport_direction_alignment_min: float = 0.50
    transport_direction_retract_ratio: float = 0.80
    retract_dist_m: float = 0.06
    retract_z_rise_m: float = 0.035
    place_fallback_transport_m: float = 0.03
    place_fallback_retract_ratio: float = 0.50

    # 归位
    withdraw_home_xy_m: float = 0.10
    withdraw_home_z_m: float = 0.05
    return_min_excursion_m: float = 0.08
    return_min_progress_m: float = 0.03
    return_max_sec: float = 10.0
    return_grip_open_required: bool = True

    # 放置区域
    place_roi_xy_m: float = 0.06
    place_roi_z_m: float = 0.05
    place_roi_edge_m: float = 0.02
    default_place_center: tuple[float, float, float] = (-0.077, 0.512, 0.168)

    # 夹爪
    grip_open_threshold: float = 0.70
    grip_close_threshold: float = 0.50
    grip_release_rise: float = 0.15
    grip_pre_close_max: float = 0.30
    grip_empty_closed_max: float = 0.10
    grip_reliable_sample_min: int = 5
    grasp_current_ignore_frames: int = 5
    grasp_z_range_fallback_m: float = 0.02
    grasp_fallback_min_phase_index: int = 10
    grasp_fallback_approach_ratio: float = 0.60

    # 视觉证据提取与语义判定
    vision_sample_stride: int = 6
    vision_width: int = 320
    vision_height: int = 180
    vision_min_object_pixels: int = 45
    vision_min_box_pixels: int = 250
    vision_present_rate: float = 0.35
    vision_object_motion_norm: float = 0.02
    vision_toward_box_norm: float = 0.03
    vision_event_window_sec: float = 1.0
    vision_settle_window_sec: float = 0.6
    vision_settle_motion_norm: float = 0.012
    vision_settle_release_delay_sec: float = 0.15
    vision_settle_min_frames: int = 3
    vision_box_inner_margin_frac: float = 0.08
    vision_inside_ratio: float = 0.65
    vision_wrist_gripper_x_norm: float = 0.50
    vision_wrist_gripper_y_norm: float = 0.82
    vision_wrist_near_gripper_norm: float = 0.28
    vision_wrist_retain_rate: float = 0.55
    vision_wrist_acquire_window_sec: float = 1.0
    vision_wrist_acquire_rate: float = 0.50
    vision_wrist_reliable_observation_rate: float = 0.40
    vision_wrist_drop_far_min_frames: int = 3
    vision_wrist_drop_far_rate: float = 0.75
    vision_wrist_release_guard_sec: float = 0.20
    vision_wrist_approach_drop_norm: float = 0.08

    # 数据质量
    gripper_udp_fresh_sec: float = 0.08
    image_state_sync_good_sec: float = 0.04
    tracking_error_good_rad: float = 1.0
    min_frames: int = 3
    default_fps: float = 15.0

    def decide(self, stage: str, task: TaskEvidence) -> RuleDecision:
        evaluators = {
            "S1定位": self._decide_s1,
            "S2抓取": self._decide_s2,
            "S3搬运": self._decide_s3,
            "S4投放": self._decide_s4,
            "S5归位": self._decide_s5,
        }
        return evaluators[stage](task)

    def _decide_s1(self, task: TaskEvidence) -> RuleDecision:
        if not task.has_motion:
            return RuleDecision(0, 0.90, "所有关节/EE均未达到1°/1cm运动阈值")
        if task.vision.object_present is False:
            return RuleDecision(1, 0.90, "视频确认场景中没有目标物，机械臂仍发生移动")
        if task.has_grasp_object:
            if task.grasp_time_sec <= self.positioning_fast_sec:
                return RuleDecision(4, 0.82, "稳定抓取反证快速精准到位")
            if task.grasp_time_sec <= self.positioning_slow_sec:
                return RuleDecision(3, 0.80, "稳定抓取反证已到位但耗时较长")
            return RuleDecision(2, 0.72, "到达目标但耗时超过10秒")
        if task.approach_quality in {"fast", "slow"} or task.has_grasp_attempt:
            return RuleDecision(2, 0.74, "到达目标附近，但没有可靠接触")
        return RuleDecision(1, 0.78, "有移动，但没有形成朝目标靠近的多视角证据")

    def _decide_s2(self, task: TaskEvidence) -> RuleDecision:
        if task.vision.object_present is False:
            return RuleDecision(0, 0.92, "视频确认无目标物")
        if not task.has_grasp_attempt:
            return RuleDecision(0, 0.88, "无真实UDP闭合事件")
        if task.grip_fully_closed and not task.events.object_acquired:
            return RuleDecision(1, 0.90, "夹爪完全闭合，且没有物体获取或接触证据")
        if task.has_grasp_object:
            return RuleDecision(3, 0.88, "获取、抬升、保持和运输均成立")
        if task.events.object_lifted and task.events.object_dropped:
            return RuleDecision(2, 0.80, "确认物体抬起后在正常释放前掉落")
        if task.events.object_lifted:
            return RuleDecision(
                1,
                0.62,
                "确认短暂抬起，但没有可靠的提前掉落或稳定保持证据",
                review_reason="抓取结果介于失败和稳定抓取之间，需要复核",
            )
        if not task.vision.available:
            return RuleDecision(
                1,
                0.82,
                "执行闭合但无稳定抬升/保持证据",
                review_reason="仅有传感器证据",
                missing_evidence=("抓取过程视觉证据",),
            )
        return RuleDecision(1, 0.82, "执行闭合但无稳定抬升/保持证据")

    def _decide_s3(self, task: TaskEvidence) -> RuleDecision:
        if task.vision.object_present is False:
            return RuleDecision(0, 0.92, "无目标物")
        if task.vision.box_present is False:
            return RuleDecision(0, 0.92, "OOD场景无盒子，v5按未搬运至目标记0分")
        if not task.has_grasp_object:
            return RuleDecision(0, 0.84, "无稳定抓取")
        if task.transport_to_place and task.vision.final_object_relation in {
            "inside",
            "edge",
        }:
            return RuleDecision(2, 0.86, "物体到达盒口区域")
        if (
            task.transport_to_place
            or task.grasp_transport_m >= self.transport_partial_m
        ):
            return RuleDecision(1, 0.78, "向盒子方向搬运但未到位")
        return RuleDecision(0, 0.80, "抓取后未形成有效水平搬运")

    def _decide_s4(self, task: TaskEvidence) -> RuleDecision:
        if task.vision.object_present is False:
            return RuleDecision(0, 0.92, "无目标物")
        if task.vision.box_present is False:
            return RuleDecision(0, 0.92, "OOD场景无盒子，按未投放记0分")
        if not task.grip_release_detected:
            return RuleDecision(0, 0.90, "无真实UDP张开释放事件")
        if not task.has_grasp_object:
            return RuleDecision(0, 0.84, "没有稳定夹持物体，张开动作不构成投放")
        if task.place_at_box:
            return RuleDecision(3, 0.88, "物体释放后稳定落于盒内")
        if task.vision.final_object_relation == "edge" and task.events.object_settled:
            return RuleDecision(2, 0.84, "物体释放后稳定落于盒边")
        if task.vision.final_object_relation == "outside":
            return RuleDecision(1, 0.86, "物体释放后落于盒外")
        if task.vision.final_object_relation in {"inside", "edge"}:
            return RuleDecision(
                1,
                0.58,
                "物体经过盒口区域，但释放后未观察到稳定落点",
                review_reason="需要连续观察释放后的稳定状态",
                missing_evidence=("释放后物体稳定落点",),
            )
        return RuleDecision(
            1,
            0.45,
            "检测到释放，但最终落点不可见",
            review_reason="必须查看释放后的连续视频",
            missing_evidence=("释放后物体稳定落点",),
        )

    def _decide_s5(self, task: TaskEvidence) -> RuleDecision:
        if (
            task.max_home_excursion_m < self.return_min_excursion_m
            or task.return_progress_m < self.return_min_progress_m
        ):
            return RuleDecision(0, 0.88, "未形成明确的离开后归位动作")
        grip_known = not math.isnan(task.grip_final_val)
        grip_ok = (
            task.grip_final_val > self.grip_open_threshold
            if self.return_grip_open_required and grip_known
            else True
        )
        position_ok = (
            task.home_xy_error_m <= self.withdraw_home_xy_m
            and task.home_z_error_m <= self.withdraw_home_z_m
        )
        time_ok = task.return_duration_sec <= self.return_max_sec
        if position_ok and time_ok and grip_ok:
            return RuleDecision(
                2,
                0.92 if grip_known else 0.74,
                "位置、时间和夹爪状态均满足归位标准",
                missing_evidence=() if grip_known else ("归位后夹爪状态",),
            )
        reasons = []
        if not position_ok:
            reasons.append("终点位置超差")
        if not time_ok:
            reasons.append("归位超过10秒")
        if not grip_ok:
            reasons.append("归位后夹爪未张开")
        return RuleDecision(
            1,
            0.88 if grip_known else 0.70,
            "、".join(reasons),
            missing_evidence=() if grip_known else ("归位后夹爪状态",),
        )


V5_RULES = V5RuleSet()
RULE_VERSION = V5_RULES.version

V5_STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        "S1定位",
        "定位与接近",
        4,
        ("motion", "vision", "quality"),
        ConfidencePolicy(0.55, 0.45, 0.15, 0.94),
    ),
    StageSpec(
        "S2抓取",
        "抓取与抬起",
        3,
        ("gripper", "contact", "vision", "quality"),
        ConfidencePolicy(
            0.40,
            0.60,
            0.20,
            0.93,
            modality_conflict_penalty=0.10,
        ),
    ),
    StageSpec(
        "S3搬运",
        "搬运至目标",
        2,
        ("motion", "vision", "quality"),
        ConfidencePolicy(
            0.25,
            0.75,
            0.28,
            0.93,
            unknown_relation_penalty=0.10,
        ),
    ),
    StageSpec(
        "S4投放",
        "投放释放",
        3,
        ("gripper", "vision", "quality"),
        ConfidencePolicy(
            0.15,
            0.85,
            0.35,
            0.93,
            unknown_relation_penalty=0.16,
        ),
    ),
    StageSpec(
        "S5归位",
        "机械臂归位",
        2,
        ("motion", "gripper", "quality"),
        ConfidencePolicy(1.0, 0.0, 0.0, 0.96),
    ),
)

V5_STAGE_BY_KEY = {spec.key: spec for spec in V5_STAGE_SPECS}

RULES_TEXT = """
============================================================
  机械臂抓取模型 自动评分规则（v5）
============================================================
S1定位: 0=未动；1=方向错误；2=到达附近；3=5~10秒到位；4=5秒内到位
S2抓取: 0=未抓取；1=抓取失败；2=抬起掉落；3=稳定抓取
S3搬运: 0=未搬运；1=搬运未到位；2=到达盒口上方
S4投放: 0=未释放；1=盒外；2=盒边；3=盒内
S5归位: 0=未归位；1=归位偏差；2=位置、时间和夹爪状态均达标

所有量化阈值、阶段模态权重和复核阈值均定义于 V5RuleSet。
============================================================
"""
