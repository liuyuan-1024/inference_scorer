"""多模态动作事件链融合。

这里只负责“发生了什么”，不负责轨迹特征提取，也不负责映射 Excel 分数。
"""

from __future__ import annotations

import numpy as np

from config import (
    FZ_CONTACT_DELTA_MIN,
    GRASP_LIFT_M,
    GRASP_TRANSPORT_M,
    JC_CONTACT_DELTA,
    VISION_OBJECT_MOTION_NORM,
)
from models import ActionEventChain, TaskSignals


def build_action_event_chain(
    task: TaskSignals, *, has_grasp_attempt: bool
) -> tuple[ActionEventChain, bool]:
    """按事件先后融合夹爪、传感器、腕部和胸前视频证据。"""
    jc_ok = task.jc_contact_delta >= JC_CONTACT_DELTA
    fz_ok = task.fz_contact_delta >= FZ_CONTACT_DELTA_MIN
    sensor_contact = jc_ok or fz_ok
    wrist_contact = (
        task.vision.wrist_available
        and task.vision.wrist_object_near_gripper
    )
    contact = bool(
        has_grasp_attempt
        and not task.grip_empty_close
        and (wrist_contact or sensor_contact)
    )

    chest_object_motion = (
        task.vision.chest_available
        and task.vision.object_motion_after_grasp_norm
        >= VISION_OBJECT_MOTION_NORM
    )
    retained = bool(
        contact
        and (
            task.vision.wrist_object_retained
            if task.vision.wrist_available
            else chest_object_motion or sensor_contact
        )
    )
    # 单目画面中的水平运动也可能是推挤；没有可靠深度时，
    # “抬起”必须由 EE 的真实 z 上升确认。
    lifted = bool(contact and task.retract_z_rise_m >= 0.02)
    dropped = bool(
        contact
        and (
            task.vision.wrist_drop_detected
            or (lifted and task.vision.wrist_available and not retained)
        )
    )
    stable_lift = task.retract_z_rise_m >= GRASP_LIFT_M
    transported = bool(
        retained
        and not dropped
        and task.grasp_transport_m >= GRASP_TRANSPORT_M
    )
    has_grasp_object = bool(
        contact and retained and lifted and stable_lift and transported and not dropped
    )
    reached_box = bool(
        transported
        and task.vision.moved_toward_box
        and task.vision.final_object_relation in {"inside", "edge"}
    )
    release = bool(task.grip_release_detected)
    settled = bool(release and task.vision.object_settled_after_release)
    placed_inside = bool(
        has_grasp_object
        and release
        and settled
        and task.vision.final_object_relation == "inside"
    )

    notes = [
        f"接触证据: wrist={wrist_contact}, jc={jc_ok}, fz={fz_ok}",
        f"保持={retained}, 抬升={lifted}, 掉落={dropped}",
    ]
    events = ActionEventChain(
        close_detected=has_grasp_attempt,
        contact_detected=contact,
        object_between_jaws=wrist_contact,
        object_lifted=lifted,
        object_retained=retained,
        object_dropped=dropped,
        transported=transported,
        reached_box=reached_box,
        release_detected=release,
        object_settled=settled,
        placed_inside=placed_inside,
        notes=notes,
    )
    return events, has_grasp_object


def fused_confidence(
    task: TaskSignals, events: ActionEventChain
) -> tuple[float, float, float]:
    """返回 overall / vision / sensor 置信度，模态冲突时主动降级。"""
    vision_confidence = task.vision.confidence if task.vision.available else 0.0
    sensor_confidence = task.sensor_confidence or 0.45
    if task.vision.available:
        overall = 0.56 * vision_confidence + 0.44 * sensor_confidence
    else:
        overall = sensor_confidence * 0.82

    sensor_contact = (
        task.jc_contact_delta >= JC_CONTACT_DELTA
        or task.fz_contact_delta >= FZ_CONTACT_DELTA_MIN
    )
    visual_contact = events.object_between_jaws
    if task.vision.wrist_available and sensor_contact != visual_contact:
        overall -= 0.10
    if task.state_feedback_timeout_rate >= 0.20:
        overall -= 0.06
    return (
        float(np.clip(overall, 0.25, 0.93)),
        float(np.clip(vision_confidence, 0.0, 0.95)),
        float(np.clip(sensor_confidence, 0.0, 0.98)),
    )
