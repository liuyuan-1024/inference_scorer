"""
数据加载 — 读取 eval_log.jsonl、machine_flow.jsonl、task_segments.jsonl 等。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from domain.models import TaskEvidence
from domain.rules_v5 import V5_RULES


def _as_float(value, default: float = float("nan")) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def load_frame_to_segment(data_dir: Path) -> dict[int, int]:
    """
    读取 machine_flow.jsonl，建立 frame_index → task_segment_index 映射。

    备用方案：从 trajectory.parquet 读取。
    """
    # 保留旧 API 供外部调用；分段解析的唯一实现在 inputs.segments。
    from inputs.segments import RecordedTaskSegmentResolver

    return RecordedTaskSegmentResolver().resolve(data_dir).state_frame_to_segment


def load_eval_log(data_dir: Path) -> list[dict]:
    """
    读取 eval_log.jsonl 并返回所有 eval_inference_frame 条目的原始字典列表。
    """
    p = data_dir / "eval_log.jsonl"
    if not p.is_file() or p.stat().st_size < 200:
        return []

    records: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") != "eval_inference_frame":
                continue
            records.append(o)
    return records


def ingest_eval_log(
    data_dir: Path, frame_seg: dict[int, int]
) -> dict[int, TaskEvidence]:
    """解析 eval_log.jsonl，按 task 聚合信号。"""
    records = load_eval_log(data_dir)
    if not records:
        return {}

    by_seg: dict[int, dict] = {}
    for o in records:
        fi = int(o["frame_index"])
        seg = frame_seg.get(fi)
        if seg is None:
            continue

        rs = o["checkpoint_input"]["robot_state"]
        ee_pose = rs.get("end_effector_pose") or {}
        ee = (ee_pose.get("left_arm") or [0, 0, 0])[:3]
        grip_state = rs.get("gripper_state") or {}
        grip_udp = o.get("gripper_udp_feedback") or rs.get("gripper_udp_feedback") or {}
        grip = grip_udp.get("left", grip_state.get("left_gripper"))
        wrench = (rs.get("end_effector_wrench_6d") or {}).get("left_arm") or [0] * 6
        fz = float(wrench[2]) if len(wrench) > 2 else 0.0
        jc_list = (rs.get("joint_current") or {}).get("left_arm") or []
        jc = max((abs(float(x)) for x in jc_list), default=0.0)
        jp_list = (rs.get("joint_position") or {}).get("left_arm") or []
        timestamps = o.get("timestamps") or {}
        state_time = _as_float(
            timestamps.get("state"), _as_float(o.get("wall_time"), float(fi))
        )
        image_time = _as_float(timestamps.get("image"), state_time)
        image_state_diff = abs(
            _as_float(timestamps.get("image_state_diff_sec"), np.nan)
        )

        desired_grip = np.nan
        feedback_rows: list[dict] = []
        for action_step in o.get("action_steps") or []:
            desired = (
                (action_step.get("desired_state") or {}).get("gripper_state") or {}
            ).get("left_gripper")
            if desired is not None:
                desired_grip = float(desired)
            feedback_rows.append(action_step.get("feedback") or {})

        if seg not in by_seg:
            by_seg[seg] = {
                "ee": [],
                "grip": [],
                "jc": [],
                "jc_joint": [],
                "fz": [],
                "jp": [],
                "frame_index": [],
                "state_time": [],
                "image_time": [],
                "image_state_diff": [],
                "grip_age": [],
                "grip_received": [],
                "grip_source": [],
                "desired_grip": [],
                "feedback": [],
            }
        d = by_seg[seg]
        d["ee"].append(ee)
        d["grip"].append(grip if grip is not None else np.nan)
        d["jc"].append(jc)
        d["jc_joint"].append(jc_list)
        d["fz"].append(fz)
        d["jp"].append(jp_list)
        d["frame_index"].append(fi)
        d["state_time"].append(state_time)
        d["image_time"].append(image_time)
        d["image_state_diff"].append(image_state_diff)
        d["grip_age"].append(_as_float(grip_udp.get("age_sec"), np.nan))
        d["grip_received"].append(bool(grip_udp.get("received", grip is not None)))
        d["grip_source"].append(
            str(grip_state.get("observation_source") or grip_udp.get("source") or "")
        )
        d["desired_grip"].append(desired_grip)
        d["feedback"].extend(feedback_rows)

    out: dict[int, TaskEvidence] = {}
    for seg, d in by_seg.items():
        # 日志不保证物理行序严格递增；task 原点必须取该 segment
        # 帧号最小的状态，而不是文件中偶然最先出现的状态。
        order = np.argsort(np.asarray(d["frame_index"]), kind="stable")
        per_frame_keys = (
            "ee",
            "grip",
            "jc",
            "jc_joint",
            "fz",
            "jp",
            "frame_index",
            "state_time",
            "image_time",
            "image_state_diff",
            "grip_age",
            "grip_received",
            "grip_source",
            "desired_grip",
        )
        for key in per_frame_keys:
            d[key] = [d[key][index] for index in order]

        ee = np.array(d["ee"], dtype=float)
        grip = np.array(d["grip"], dtype=float)
        jc = np.array(d["jc"], dtype=float)
        fz = np.array(d["fz"], dtype=float)
        frame_indices = np.array(d["frame_index"], dtype=int)
        frame_times = np.array(d["state_time"], dtype=float)
        image_times = np.array(d["image_time"], dtype=float)
        grip_age = np.array(d["grip_age"], dtype=float)
        grip_received = np.array(d["grip_received"], dtype=bool)
        desired_grip = np.array(d["desired_grip"], dtype=float)
        jp = (
            np.array(d["jp"], dtype=float)
            if d["jp"] and d["jp"][0]
            else np.empty((0, 7))
        )
        jc_joint = (
            np.array(d["jc_joint"], dtype=float)
            if d["jc_joint"] and d["jc_joint"][0]
            else np.empty((0, 7))
        )

        t = TaskEvidence(task_index=seg, n_frames=len(ee))
        t.ee_traj = ee
        t.fz_traj = fz
        t.jc_traj = jc
        t.jc_joint_traj = jc_joint
        t.grip_traj = grip
        t.joint_traj = jp
        t.frame_indices = frame_indices
        t.frame_times = frame_times
        t.image_times = image_times
        t.grip_feedback_age_traj = grip_age
        t.grip_feedback_received_traj = grip_received
        t.desired_grip_traj = desired_grip
        t.grip_observation_source = next(
            (source for source in d["grip_source"] if source), ""
        )
        if len(frame_times) >= 2:
            t.duration_sec = float(frame_times[-1] - frame_times[0])

        if len(ee):
            t.task_start_ee = ee[0].tolist()
            t.task_start_frame_index = int(frame_indices[0])
            if np.isfinite(frame_times[0]):
                t.task_start_timestamp = float(frame_times[0])
        if len(ee) >= 2:
            t.ee_path_m = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())
            t.max_ee_excursion_m = float(np.linalg.norm(ee - ee[0], axis=1).max())
        if len(jp) >= 2:
            t.joint_delta_rad = float(np.abs(np.diff(jp, axis=0)).sum())
            t.max_joint_excursion_rad = float(np.abs(jp - jp[0]).max())

        t.jc_max = float(jc.max()) if len(jc) else 0.0

        # 夹爪统计
        if len(grip) > 0:
            valid = grip[~np.isnan(grip)]
            if len(valid) > 0:
                t.grip_min_val = float(valid.min())
                t.grip_max_val = float(valid.max())
                t.grip_final_val = float(valid[-1])

        # 数据质量只影响置信度，不直接改变动作语义。
        feedback = d["feedback"]
        if feedback:
            received = [
                bool(item.get("state_feedback_received", False)) for item in feedback
            ]
            t.state_feedback_timeout_rate = 1.0 - float(np.mean(received))
            tracking_errors = [
                float(item["response_tracking_error_max"])
                for item in feedback
                if item.get("response_tracking_error_max") is not None
            ]
            if tracking_errors:
                t.response_tracking_error_p95 = float(
                    np.percentile(tracking_errors, 95)
                )

        valid_sync = np.array(d["image_state_diff"], dtype=float)
        valid_sync = valid_sync[np.isfinite(valid_sync)]
        if len(valid_sync):
            t.image_state_diff_p95_sec = float(np.percentile(valid_sync, 95))
        valid_age = grip_age[np.isfinite(grip_age)]
        if len(valid_age):
            t.gripper_udp_age_p95_sec = float(np.percentile(valid_age, 95))

        udp_rate = float(grip_received.mean()) if len(grip_received) else 0.0
        udp_freshness = (
            max(0.0, 1.0 - t.gripper_udp_age_p95_sec / V5_RULES.gripper_udp_fresh_sec)
            if np.isfinite(t.gripper_udp_age_p95_sec)
            else 0.0
        )
        sync_quality = (
            max(
                0.0,
                1.0 - t.image_state_diff_p95_sec / V5_RULES.image_state_sync_good_sec,
            )
            if np.isfinite(t.image_state_diff_p95_sec)
            else 0.0
        )
        feedback_quality = 1.0 - min(1.0, t.state_feedback_timeout_rate)
        tracking_quality = max(
            0.0, 1.0 - t.response_tracking_error_p95 / V5_RULES.tracking_error_good_rad
        )
        t.sensor_confidence = float(
            np.clip(
                0.30 * udp_rate
                + 0.20 * udp_freshness
                + 0.25 * sync_quality
                + 0.15 * feedback_quality
                + 0.10 * tracking_quality,
                0.0,
                0.98,
            )
        )

        out[seg] = t
    return out


def load_meta(data_dir: Path) -> dict:
    """读取 meta.json。"""
    p = data_dir / "meta.json"
    if p.is_file():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_frame_timestamps(data_dir: Path) -> list[dict]:
    """读取 videos/frame_timestamps.jsonl。"""
    p = data_dir / "videos" / "frame_timestamps.jsonl"
    if not p.is_file():
        return []
    records: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_fps(data_dir: Path, default: float = V5_RULES.default_fps) -> float:
    """从 meta.json 获取帧率。"""
    meta = load_meta(data_dir)
    return float(meta.get("fps", default))
