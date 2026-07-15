"""
数据加载 — 读取 eval_log.jsonl、machine_flow.jsonl、task_segments.jsonl 等。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models import TaskSignals


def load_frame_to_segment(data_dir: Path) -> dict[int, int]:
    """
    读取 machine_flow.jsonl，建立 frame_index → task_segment_index 映射。

    备用方案：从 trajectory.parquet 读取。
    """
    mapping: dict[int, int] = {}
    mf = data_dir / "machine_flow.jsonl"
    if mf.is_file():
        with mf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if o.get("event") == "chunk_inference" and "frame_index" in o:
                    seg = o.get("task_segment_index")
                    if seg is not None:
                        mapping[int(o["frame_index"])] = int(seg)

    if mapping:
        return mapping

    # 备用：从 trajectory.parquet 读取
    pq = data_dir / "trajectory_data" / "trajectory.parquet"
    if pq.is_file():
        import pandas as pd

        df = pd.read_parquet(
            pq, columns=["inference_frame_index", "task_segment_index"]
        )
        for fi, seg in zip(df["inference_frame_index"], df["task_segment_index"]):
            mapping[int(fi)] = int(seg)
    return mapping


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
) -> dict[int, TaskSignals]:
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
        grip = (rs.get("gripper_state") or {}).get("left_gripper")
        wrench = (rs.get("end_effector_wrench_6d") or {}).get("left_arm") or [0] * 6
        fz = float(wrench[2]) if len(wrench) > 2 else 0.0
        jc_list = (rs.get("joint_current") or {}).get("left_arm") or []
        jc = max((abs(float(x)) for x in jc_list), default=0.0)
        jp_list = (rs.get("joint_position") or {}).get("left_arm") or []

        if seg not in by_seg:
            by_seg[seg] = {"ee": [], "grip": [], "jc": [], "fz": [], "jp": []}
        d = by_seg[seg]
        d["ee"].append(ee)
        d["grip"].append(grip if grip is not None else np.nan)
        d["jc"].append(jc)
        d["fz"].append(fz)
        d["jp"].append(jp_list)

    out: dict[int, TaskSignals] = {}
    for seg, d in by_seg.items():
        ee = np.array(d["ee"], dtype=float)
        grip = np.array(d["grip"], dtype=float)
        jc = np.array(d["jc"], dtype=float)
        fz = np.array(d["fz"], dtype=float)
        jp = (
            np.array(d["jp"], dtype=float)
            if d["jp"] and d["jp"][0]
            else np.empty((0, 7))
        )

        t = TaskSignals(task_index=seg, n_frames=len(ee))
        t.ee_traj = ee
        t.fz_traj = fz
        t.jc_traj = jc
        t.grip_traj = grip

        if len(ee):
            t.ee_start = ee[0].tolist()
        if len(ee) >= 2:
            t.ee_path_m = float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum())
        if len(jp) >= 2:
            t.joint_delta_rad = float(np.abs(np.diff(jp, axis=0)).sum())

        t.jc_max = float(jc.max()) if len(jc) else 0.0

        # 夹爪统计
        if len(grip) > 0:
            valid = grip[~np.isnan(grip)]
            if len(valid) > 0:
                t.grip_min_val = float(valid.min())
                t.grip_max_val = float(valid.max())

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


def get_fps(data_dir: Path, default: float = 15.0) -> float:
    """从 meta.json 获取帧率。"""
    meta = load_meta(data_dir)
    return float(meta.get("fps", default))
