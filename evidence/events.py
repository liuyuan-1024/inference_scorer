"""多模态动作事件链融合。

这里只负责“发生了什么”，不负责轨迹特征提取，也不负责映射 Excel 分数。
"""

from __future__ import annotations

from domain.models import ActionEventChain, TaskEvidence
from domain.rules_v5 import V5_RULES


def build_action_event_chain(
    task: TaskEvidence, *, has_grasp_attempt: bool
) -> tuple[ActionEventChain, bool]:
    """按事件先后融合夹爪、传感器、腕部和胸前视频证据。"""
    jc_ok = task.jc_contact_delta >= V5_RULES.jc_contact_delta
    fz_ok = task.fz_contact_delta >= V5_RULES.fz_contact_delta_min
    sensor_contact = jc_ok or fz_ok
    wrist_acquired = (
        task.vision.wrist_available and task.vision.wrist_object_near_gripper
    )
    wrist_reliable = (
        task.vision.wrist_available
        and task.vision.wrist_observation_rate
        >= V5_RULES.vision_wrist_reliable_observation_rate
    )
    empty_close_candidate = bool(
        task.grip_fully_closed and not wrist_acquired and not sensor_contact
    )
    contact = bool(
        has_grasp_attempt
        and not empty_close_candidate
        and (wrist_acquired or sensor_contact)
    )

    chest_object_motion = (
        task.vision.chest_available
        and task.vision.object_motion_after_grasp_norm
        >= V5_RULES.vision_object_motion_norm
    )
    acquired = bool(
        contact
        and (
            wrist_acquired if wrist_reliable else sensor_contact and chest_object_motion
        )
    )
    # EE 上升只说明夹爪抬起；必须先确认物体已进入夹爪，才能称为物体抬起。
    lifted = bool(acquired and task.retract_z_rise_m >= V5_RULES.grasp_min_lift_m)
    retained = bool(
        lifted
        and (
            task.vision.wrist_object_retained if wrist_reliable else chest_object_motion
        )
    )
    # 掉落必须是“已获取并已抬起”之后，腕部视角明确观察到释放前脱离。
    # 不再用单纯的后段漏检或未保持来推断掉落。
    dropped = bool(lifted and wrist_reliable and task.vision.wrist_drop_detected)
    stable_lift = task.retract_z_rise_m >= V5_RULES.grasp_lift_m
    transported = bool(
        retained
        and not dropped
        and task.grasp_transport_m >= V5_RULES.grasp_transport_m
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
        f"接触证据: wrist={wrist_acquired}, jc={jc_ok}, fz={fz_ok}",
        f"获取={acquired}, 抬升={lifted}, 保持={retained}, 掉落={dropped}",
    ]
    events = ActionEventChain(
        close_detected=has_grasp_attempt,
        contact_detected=contact,
        object_between_jaws=wrist_acquired,
        object_acquired=acquired,
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
