#!/usr/bin/env python3
"""从 eval_log.jsonl + trial 首帧图像自动判定真机 trial 抓取语义。

阶段语义：定位（靠近目标）→ 移动 → 抓取 → 撤回/运输 → 放置。

数据源（每个 action-step_* / test 目录）：
  - eval_log.jsonl       逐帧 robot_state（EE FK、wrench_6d、joint_current；**不用夹爪反馈**）
  - task_segments.jsonl  trial 起止帧 + 首帧 video_frame_index
  - machine_flow.jsonl   frame_index -> task_segment_index 映射
  - eval_videos/cam_mid  首帧红玩具 centroid → cell_3x3（定位参考）

用法：
  # 单个 eval 目录
  python3 tools/real_robot/judge_real_robot_trial_semantics.py \\
    --test-dir research/ablation/runs/real/00002/.../010000/action-step_10

  # 扫描整个 real 目录
  python3 tools/real_robot/judge_real_robot_trial_semantics.py \\
    --root research/ablation/runs/real --out-dir research/ablation/runs/real/_analysis

  # 打印判定规则说明
  python3 tools/real_robot/judge_real_robot_trial_semantics.py --explain
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools._repo import ensure_repo_on_path

ensure_repo_on_path()
from tools.real_robot.analyze_infer_trial_pose_distribution import (  # noqa: E402
    parse_task_segments_jsonl,
    read_video_frame,
    find_cam_mid_video,
)
from tools.real_robot.analyze_real_dataset_pose_distribution import (  # noqa: E402
    detect_container,
    detect_red_object,
    grid_cell,
    grid_cell_3x3,
)

import cv2

# ---------------------------------------------------------------------------
# 判定阈值 — 全部从 eval_log.jsonl 逐帧读取并在 trial 内聚合（ingest_eval_log）
#
# 日志行：type=eval_inference_frame
#   checkpoint_input.robot_state:
#     end_effector_pose.left_arm[:3]  EE 位置 xyz (m)
#     end_effector_wrench_6d.left_arm 末端六维力/力矩（fz 作接触代理）
#     joint_current.left_arm[7]       关节电流 (mA)，低 z 窗口尖峰作负载代理
#     joint_position.left_arm[7]      关节角 (rad)，帧间 |Δ| 累加
#     gripper_state — 仅记录，**不参与判定**（多为策略指令回显，非传感器反馈）
#
# trial 切段：machine_flow.jsonl 的 frame_index→task_segment_index（或 trajectory parquet）
# 首帧玩具方向（approach 对齐）另读 cam_mid 视频，其余标量均来自 eval_log。
# ---------------------------------------------------------------------------

# --- 移动 ---
MOTION_EE_PATH_M = 0.20
# eval_log: end_effector_pose.left_arm[:3] 逐帧差分 L2 范数之和 → ee_path_m
# has_motion: ee_path_m ≥ 0.20m（区分 trial 内几乎不动 vs 有实质位移）

MOTION_JOINT_RAD = 0.15
# eval_log: joint_position.left_arm 逐帧 Σ|Δq_i| → joint_delta_rad
# EE 缺失时的 OR 备选：joint_delta_rad ≥ 0.15rad

# --- 抓取 / 放置（EE 相位 + wrench_6d + joint_current；不依赖夹爪）---
FZ_CONTACT_MIN = 5.0
# eval_log: 抓取相位窗口内 max|fz|；≥5N 作接触代理

GRASP_PHASE_WINDOW = 8
# 以 argmin(EE.z) 为抓取相位锚点，前后若干帧统计 jc/fz

JC_CONTACT_DELTA = 400
# 抓取窗口内 max(jc) ≥ batch_jc_p25 + 400 → 绝对负载代理

JC_GRASP_RISE_MIN = 600
# 抓取窗口 jc 尖峰 − 抓取前均值；≥600 作相对负载代理
JC_CLOSE_RISE_MIN = JC_GRASP_RISE_MIN  # 兼容旧名 / crosswalk 导入

GRASP_TRANSPORT_M = 0.06
# 抓取相位后 EE 累计位移；≥0.06m 且负载信号 → 抓上并带走

PLACE_ROI_XY_M = 0.06
PLACE_ROI_Z_M = 0.05
# 抓取后 EE 是否进入放置 ROI（纯运动学，不用夹爪张开）

# 兼容旧字段名（CSV 列）；语义已改为 kinematic phase，非夹爪
GRIP_CLOSE = 0.25
GRIP_OPEN = 0.55

MIN_FRAMES = 3
# eval_log: 该 trial 内 eval_inference_frame 计数 n_frames ≥ 3 才参与判定

# --- 靠近（关爪前段，EE 来自 eval_log）---
APPROACH_Z_DROP_M = 0.04
# eval_log: ee_start[2] − min(pre_close EE z) → approach_z_drop_m ≥ 0.04m

APPROACH_XY_M = 0.05
# eval_log: ‖pre_close EE_xy − ee_start_xy‖ → approach_xy_m ≥ 0.05m

APPROACH_ALIGN_MIN = 0.30
# eval_log EE 位移方向 · 首帧玩具推断方向 → approach_align ≥ 0.30

PATH_EFFICIENCY_MIN = 0.30
# eval_log: 关爪前 EE 直线距离/路径长 → path_efficiency ≥ 0.30

FAST_APPROACH_FRAC = 0.40
# eval_log: argmin(pre_close z) 所在帧 / 关爪前总帧数 ≤ 0.40 → fast

# --- 撤回 / 抬升（关爪后段，EE 来自 eval_log）---
RETRACT_DIST_M = 0.06
# eval_log: 关爪后 max‖EE − ee_at_close‖ → retract_dist_m ≥ 0.06m

RETRACT_Z_RISE_M = 0.035
# eval_log: 关爪后 max(EE_z) − ee_at_close_z → retract_z_rise_m ≥ 0.035m

WITHDRAW_HOME_M = 0.12
# eval_log: ‖EE_last − ee_start‖ → withdraw_home_dist_m ≤ 0.12m

SEMANTIC_LABELS = {
    "no_motion": "几乎不移动",
    "motion_no_grasp": "有移动但无抓取动作",
    "grasp_attempt_fail": "尝试抓取但失败（抓空/无接触）",
    "grasp_no_place": "抓上有接触但未完成放置张开",
    "place_miss": "放置张开但不在黄盒 ROI",
    "action_complete_ok": "动作链完整且落点正确（自动口径）",
    "insufficient_log": "日志不足，无法判定",
}

APPROACH_LABELS = {
    "none": "未靠近目标",
    "wander": "无序运动偏离目标",
    "slow": "缓慢移动靠近",
    "fast": "快速准确定位",
    "no_toy": "首帧未检测到玩具（无法判定位）",
}

RETRACT_LABELS = {
    "none": "无撤回/抬升",
    "lift_only": "关爪后抬升但未回 home",
    "withdraw_home": "撤回至起始附近",
    "transport_place": "运输至放置区（非简单撤回）",
}


@dataclass
class TrialFirstFrame:
    trial_index: int
    toy_detected: bool = False
    red_x_norm: float = float("nan")
    red_y_norm: float = float("nan")
    cell: str | None = None
    cell_3x3: str | None = None
    box_color: str | None = None
    video_frame_index: int | None = None


@dataclass
class TrialSignals:
    trial_index: int
    n_frames: int = 0
    ee_path_m: float = 0.0
    joint_delta_rad: float = 0.0
    grip_min: float = 1.0
    grip_max: float = 0.0
    jc_max: float = 0.0
    jc_at_close: float = 0.0
    jc_pre_close: float = 0.0
    jc_close_rise: float = 0.0
    grasp_transport_m: float = 0.0
    ee_start: list[float] = field(default_factory=list)
    ee_at_place_open: list[float] = field(default_factory=list)
    ee_traj: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    fz_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    jc_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    grip_traj: np.ndarray = field(default_factory=lambda: np.empty(0))
    grasp_phase_idx: int | None = None
    close_idx: int | None = None
    jc_at_grasp: float = 0.0
    fz_spike_grasp: float = 0.0
    jc_grasp_rise: float = 0.0
    ee_at_grasp: list[float] = field(default_factory=list)
    ee_at_place: list[float] = field(default_factory=list)
    # 阶段指标（由 analyze_motion_phases 填充）
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
class TrialJudgment:
    trial_index: int
    outcome: str
    semantic: str
    has_motion: bool
    has_grasp_close: bool  # 兼容 CSV 列名；语义 = has_grasp_attempt（运动学抓取相位）
    has_grasp_contact: bool
    has_grasp_object: bool
    has_place_open: bool  # 兼容 CSV 列名；语义 = has_place_phase（抓取后放置运动）
    place_at_box: bool
    action_complete: bool
    ee_path_m: float
    joint_delta_rad: float
    grip_min: float
    grip_max: float
    jc_at_close: float
    n_frames: int
    reason: str
    jc_at_grasp: float = 0.0
    fz_spike_grasp: float = 0.0
    jc_close_rise: float = 0.0
    jc_grasp_rise: float = 0.0
    grasp_transport_m: float = 0.0
    # 首帧 + 阶段语义
    toy_detected: bool = False
    cell_3x3: str | None = None
    red_x_norm: float = float("nan")
    red_y_norm: float = float("nan")
    has_approach: bool = False
    approach_quality: str = "none"
    approach_semantic: str = APPROACH_LABELS["none"]
    motion_semantic: str = ""
    grasp_semantic: str = ""
    retract_semantic: str = RETRACT_LABELS["none"]
    has_retract: bool = False
    has_withdraw_home: bool = False
    phase_chain: str = ""

    def to_dict(self) -> dict:
        return {
            "trial_index": self.trial_index,
            "outcome": self.outcome,
            "semantic": self.semantic,
            "has_motion": self.has_motion,
            "has_grasp_close": self.has_grasp_close,
            "has_grasp_contact": self.has_grasp_contact,
            "has_grasp_object": self.has_grasp_object,
            "has_place_open": self.has_place_open,
            "place_at_box": self.place_at_box,
            "action_complete": self.action_complete,
            "ee_path_m": round(self.ee_path_m, 4),
            "joint_delta_rad": round(self.joint_delta_rad, 4),
            "grip_min": round(self.grip_min, 4),
            "grip_max": round(self.grip_max, 4),
            "jc_at_close": round(self.jc_at_close, 1),
            "jc_at_grasp": round(self.jc_at_grasp, 1),
            "fz_spike_grasp": round(self.fz_spike_grasp, 2),
            "jc_close_rise": round(self.jc_close_rise, 1),
            "jc_grasp_rise": round(self.jc_grasp_rise, 1),
            "grasp_transport_m": round(self.grasp_transport_m, 4),
            "signal_mode": "no_gripper",
            "n_frames": self.n_frames,
            "reason": self.reason,
            "toy_detected": self.toy_detected,
            "cell_3x3": self.cell_3x3 or "",
            "red_x_norm": round(self.red_x_norm, 4) if self.toy_detected else "",
            "red_y_norm": round(self.red_y_norm, 4) if self.toy_detected else "",
            "has_approach": self.has_approach,
            "approach_quality": self.approach_quality,
            "approach_semantic": self.approach_semantic,
            "motion_semantic": self.motion_semantic,
            "grasp_semantic": self.grasp_semantic,
            "retract_semantic": self.retract_semantic,
            "has_retract": self.has_retract,
            "has_withdraw_home": self.has_withdraw_home,
            "phase_chain": self.phase_chain,
        }


def print_rules() -> None:
    """打印判定逻辑（供人工核对）。"""
    text = f"""
# 真机 Trial 基础语义自动判定规则（无夹爪反馈）

## 输入
- `eval_log.jsonl` 中 `type=eval_inference_frame` 的逐帧记录
- 左臂字段：`end_effector_pose.left_arm`（xyz）、`end_effector_wrench_6d.left_arm`（fz 接触代理）、
  `joint_current.left_arm`、`joint_position.left_arm`
- **不使用** `gripper_state` 作判定（多为策略指令回显）

## 阶段判定（按顺序）

| 阶段 | 条件 | 阈值 | 对应 Excel Rubric 语义 |
|------|------|------|------------------------|
| **0. 首帧定位参考** | task_boundary start 帧 | cam_mid 红玩具 centroid → cell_3x3 | 目标在哪个格位 |
| **1. 能否动** | `has_motion` | EE 轨迹位移 ≥ {MOTION_EE_PATH_M}m **或** 关节 L1 变化 ≥ {MOTION_JOINT_RAD}rad | 「几乎不移动」vs「有移动」 |
| **2. 靠近/定位** | `has_approach` + `approach_quality` | z 下降 ≥ {APPROACH_Z_DROP_M}m 且 xy 位移 ≥ {APPROACH_XY_M}m，方向对齐 ≥ {APPROACH_ALIGN_MIN} | 「快速定位 / 缓慢靠近 / 无序偏离」 |
| **3. 抓取相位** | `has_grasp_attempt` | argmin(EE.z) 前段满足靠近条件 | 「没有抓取动作」vs「尝试抓取」 |
| **4. 是否夹到物体** | `has_grasp_contact` / `has_grasp_object` | 见下「抓取判据」 | 「抓空」vs「抓上」 |
| **5. 撤回/运输** | `has_retract` / `retract_semantic` | 抓取相位后离抓取点 ≥ {RETRACT_DIST_M}m 或 z 抬升 ≥ {RETRACT_Z_RISE_M}m | 「撤回 / 运输至放置区」 |
| **6. 放置相位** | `has_place_phase` | 抓上后 EE 后段位移 ≥ 0.03m 或朝放置 ROI 运输 | 放置阶段 |
| **7. 落点** | `place_at_box` | 末段 EE 落在放置 ROI 内 | 「放回成功」 |

### 靠近/定位细则（首帧 → 期望 EE 方向）
- 图像左侧（red_x_norm 小）→ 期望 EE x 减小；图像下方（red_y_norm 大）→ 期望 EE y 减小
- `fast`：在前 {FAST_APPROACH_FRAC:.0%} 靠近帧内到达 z 最低点，且路径效率 ≥ {PATH_EFFICIENCY_MIN}
- `slow`：满足靠近条件但较慢或路径效率偏低
- `wander`：有移动但方向不对齐或路径效率 < {PATH_EFFICIENCY_MIN}

### 撤回细则
- `lift_only`：抓取相位后抬升/离开抓取点，但未回到起始附近
- `withdraw_home`：末端 EE 距起始 ≤ {WITHDRAW_HOME_M}m
- `transport_place`：抓取后 EE 朝放置 ROI 移动（非简单撤回）

- joint_current 基线：全 trial 的 jc_max 的 25 分位数（同 batch 内估计）
- 放置 ROI：同 batch 内「抓上且后段运输」trial 的末段 EE 中位数 ± 2.5σ（下限 {PLACE_ROI_XY_M}m）

### 抓取判据（EE + wrench + joint_current，不依赖夹爪）

1. **抓取相位** `has_grasp_attempt`：`argmin(EE.z)` 且靠近条件（z↓/xy 位移）成立
2. **接触/负载** `has_grasp_contact`（满足任一）：
   - 绝对：`max(jc|抓取窗口) ≥ batch_jc_p25 + {JC_CONTACT_DELTA}`
   - 相对：`抓取窗口 jc 尖峰 − 抓取前均值 ≥ {JC_GRASP_RISE_MIN}`
   - 力传感：`max|fz|（抓取窗口）≥ {FZ_CONTACT_MIN}N`
3. **抓上并带走** `has_grasp_object`：`has_grasp_attempt` 且（`has_grasp_contact` 或（抓取后运输 ≥{GRASP_TRANSPORT_M}m **且** 负载信号））

`joint_current.left_arm` 取 7 关节 |I| 最大值；夹到刚性物体时抓取窗口电流相对抓取前明显抬升。

## Outcome 映射（第一个不满足的阶段决定）

```
无 has_motion            → no_motion          （几乎不移动）
有 motion, 无抓取相位     → motion_no_grasp    （有移动但无抓取）
有抓取相位, 无抓上        → grasp_attempt_fail （抓空）
有抓上, 无放置相位        → grasp_no_place
有放置相位, ROI 外        → place_miss
全部通过                 → action_complete_ok
帧数 < {MIN_FRAMES}      → insufficient_log
```

## 注意
- 本脚本 **不依赖 Excel**；与人工 Rubric 在「抓空/无动作」上通常一致，但
  `action_complete_ok` ≠ Excel `full_success`（后者还要求人工认定抓稳/放准）。
- `action_complete` = has_motion ∧ has_grasp_object ∧ has_place_phase（不要求 ROI）。
- CSV 列 `has_grasp_close` / `has_place_open` 为兼容别名，分别映射抓取相位/放置相位。
"""
    print(text.strip())


def load_trial_first_frames(
    test_dir: Path, *, skip_video: bool = False
) -> dict[int, TrialFirstFrame]:
    """从 task_boundary start 对应视频首帧提取红玩具位置。"""
    if skip_video:
        return {}

    out: dict[int, TrialFirstFrame] = {}
    seg_path = test_dir / "task_segments.jsonl"
    video_path = find_cam_mid_video(test_dir)
    if not seg_path.is_file() or video_path is None:
        return out

    try:
        starts = parse_task_segments_jsonl(seg_path)
    except Exception:
        return out

    for spec in starts:
        trial = int(spec["trial_index"])
        frame_idx = int(spec.get("video_frame_index") or 0)
        ff = TrialFirstFrame(trial_index=trial, video_frame_index=frame_idx)
        img_rgb = read_video_frame(video_path, frame_idx)
        if img_rgb is None:
            out[trial] = ff
            continue

        h, w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        red = detect_red_object(img_bgr)
        box = detect_container(img_bgr)
        if red:
            ff.toy_detected = True
            ff.red_x_norm = red[0] / w
            ff.red_y_norm = red[1] / h
            ff.cell = grid_cell(ff.red_x_norm, ff.red_y_norm)
            ff.cell_3x3 = grid_cell_3x3(ff.red_x_norm, ff.red_y_norm)
        if box:
            ff.box_color = box[3]
        out[trial] = ff
    return out


def expected_approach_dir_xy(red_x_norm: float, red_y_norm: float) -> np.ndarray:
    """由首帧玩具图像坐标推断 EE 应移动的 xy 方向（启发式，无标定）。"""
    # 图像左=玩具更靠近机械臂基座侧 → EE x 减小；图像下=桌面远端 → EE y 减小
    dx = -1.0 if red_x_norm < 0.30 else (-0.6 if red_x_norm < 0.45 else -0.3)
    dy = -0.8 if red_y_norm > 0.50 else (-0.3 if red_y_norm > 0.38 else 0.2)
    v = np.array([dx, dy], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([-1.0, -0.5])


def analyze_motion_phases(
    t: TrialSignals,
    ff: TrialFirstFrame | None,
    *,
    place_center: np.ndarray,
) -> None:
    """填充 TrialSignals 上的靠近/撤回阶段指标。"""
    ee = t.ee_traj
    grip = t.grip_traj
    if len(ee) < 2:
        return

    close_idx = t.grasp_phase_idx if t.grasp_phase_idx is not None else t.close_idx
    if close_idx is None and len(ee) > 0:
        close_idx = int(np.argmin(ee[:, 2]))
    if close_idx is None:
        pre_end = len(ee)
    else:
        pre_end = max(1, close_idx + 1)

    pre_ee = ee[:pre_end]
    ee_start = pre_ee[0]
    min_z_idx = int(np.argmin(pre_ee[:, 2]))
    ee_at_min_z = pre_ee[min_z_idx]

    t.approach_z_drop_m = float(ee_start[2] - ee_at_min_z[2])
    disp_xy = ee_at_min_z[:2] - ee_start[:2]
    t.approach_xy_m = float(np.linalg.norm(disp_xy))

    if ff and ff.toy_detected:
        exp_dir = expected_approach_dir_xy(ff.red_x_norm, ff.red_y_norm)
        if t.approach_xy_m > 1e-4:
            t.approach_align = float(np.dot(disp_xy / t.approach_xy_m, exp_dir))
        else:
            t.approach_align = 0.0
    elif t.approach_xy_m > 1e-4:
        # 无首帧时：任意显著 xy 位移且 z 下降即弱靠近
        t.approach_align = 0.5

    path_len = float(np.linalg.norm(np.diff(pre_ee, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(ee_at_min_z - ee_start))
    t.path_efficiency = straight / path_len if path_len > 1e-6 else 0.0
    t.approach_frame_frac = min_z_idx / max(pre_end - 1, 1)

    if close_idx is not None and close_idx < len(ee) - 1:
        close_ee = ee[close_idx]
        post_ee = ee[close_idx + 1 :]
        dists = np.linalg.norm(post_ee - close_ee, axis=1)
        t.retract_dist_m = float(dists.max()) if len(dists) else 0.0
        t.retract_z_rise_m = (
            float(post_ee[:, 2].max() - close_ee[2]) if len(post_ee) else 0.0
        )
        t.withdraw_home_dist_m = float(np.linalg.norm(ee[-1] - ee_start))

        if len(post_ee) >= 2:
            to_place = place_center[:2] - close_ee[:2]
            post_disp = post_ee[-1][:2] - close_ee[:2]
            if np.linalg.norm(to_place) > 0.03 and np.linalg.norm(post_disp) > 0.04:
                align_place = float(
                    np.dot(post_disp, to_place)
                    / (np.linalg.norm(post_disp) * np.linalg.norm(to_place))
                )
                t.transport_to_place = (
                    align_place > 0.5 and t.retract_dist_m >= RETRACT_DIST_M * 0.8
                )


def classify_approach(
    t: TrialSignals, ff: TrialFirstFrame | None, has_motion: bool
) -> tuple[bool, str, str]:
    if ff and not ff.toy_detected:
        if has_motion:
            return False, "no_toy", APPROACH_LABELS["no_toy"]
        return False, "none", APPROACH_LABELS["none"]

    z_ok = t.approach_z_drop_m >= APPROACH_Z_DROP_M
    xy_ok = t.approach_xy_m >= APPROACH_XY_M
    align_ok = t.approach_align >= APPROACH_ALIGN_MIN
    eff_ok = t.path_efficiency >= PATH_EFFICIENCY_MIN

    has_approach = z_ok and xy_ok and align_ok and eff_ok
    if not has_motion:
        return False, "none", APPROACH_LABELS["none"]
    if has_approach:
        if t.approach_frame_frac <= FAST_APPROACH_FRAC and eff_ok:
            return True, "fast", APPROACH_LABELS["fast"]
        return True, "slow", APPROACH_LABELS["slow"]
    if has_motion and (not align_ok or not eff_ok):
        return False, "wander", APPROACH_LABELS["wander"]
    return False, "none", APPROACH_LABELS["none"]


def classify_retract(
    t: TrialSignals,
    *,
    has_grasp_attempt: bool,
    has_grasp_contact: bool,
    has_place_phase: bool,
    place_ok: bool,
) -> tuple[bool, bool, str]:
    if not has_grasp_attempt:
        return False, False, RETRACT_LABELS["none"]

    has_retract = (
        t.retract_dist_m >= RETRACT_DIST_M or t.retract_z_rise_m >= RETRACT_Z_RISE_M
    )
    has_withdraw = t.withdraw_home_dist_m <= WITHDRAW_HOME_M

    if has_grasp_contact and has_place_phase and (place_ok or t.transport_to_place):
        return has_retract, has_withdraw, RETRACT_LABELS["transport_place"]
    if has_withdraw and has_retract:
        return True, True, RETRACT_LABELS["withdraw_home"]
    if has_retract:
        return True, False, RETRACT_LABELS["lift_only"]
    return False, False, RETRACT_LABELS["none"]


def build_phase_chain(
    *,
    has_motion: bool,
    approach_quality: str,
    has_grasp_attempt: bool,
    has_grasp_contact: bool,
    has_grasp_object: bool,
    has_retract: bool,
    retract_semantic: str,
    has_place_phase: bool,
    place_ok: bool,
) -> str:
    parts: list[str] = []
    if not has_motion:
        parts.append("静止")
    elif approach_quality == "fast":
        parts.append("快速定位")
    elif approach_quality == "slow":
        parts.append("缓慢靠近")
    elif approach_quality == "wander":
        parts.append("无序偏离")
    elif approach_quality == "no_toy":
        parts.append("移动(无首帧)")
    else:
        parts.append("移动未靠近")

    if has_grasp_attempt:
        parts.append("抓上" if has_grasp_contact or has_grasp_object else "抓空")
    else:
        parts.append("无抓取相位")

    if has_grasp_contact or has_grasp_object:
        if has_retract:
            if "运输" in retract_semantic:
                parts.append("运输")
            elif "撤回" in retract_semantic:
                parts.append("撤回")
            else:
                parts.append("抬升")
        if has_place_phase:
            parts.append("放置成功" if place_ok else "放置偏")
        else:
            parts.append("未放置")
    elif has_grasp_attempt and has_retract:
        parts.append("抓空后抬升")

    return " → ".join(parts)


def load_task_segments(test_dir: Path) -> list[dict]:
    p = test_dir / "task_segments.jsonl"
    if not p.is_file():
        return []
    trials: list[dict] = []
    cur: dict | None = None
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") != "task_boundary":
                continue
            if o.get("event") == "start":
                cur = {"trial_index": int(o["task_segment_index"])}
            elif o.get("event") == "end" and cur is not None:
                trials.append(cur)
                cur = None
    return trials


def load_frame_to_segment(test_dir: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    mf = test_dir / "machine_flow.jsonl"
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
    pq = test_dir / "trajectory_data" / "trajectory.parquet"
    if pq.is_file():
        import pandas as pd

        df = pd.read_parquet(
            pq, columns=["inference_frame_index", "task_segment_index"]
        )
        for fi, seg in zip(df["inference_frame_index"], df["task_segment_index"]):
            mapping[int(fi)] = int(seg)
    return mapping


def ingest_eval_log(
    test_dir: Path, frame_seg: dict[int, int]
) -> dict[int, TrialSignals]:
    by_seg: dict[int, dict] = {}
    p = test_dir / "eval_log.jsonl"
    if not p.is_file() or p.stat().st_size < 200:
        return {}

    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") != "eval_inference_frame":
                continue
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

    out: dict[int, TrialSignals] = {}
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

        t = TrialSignals(trial_index=seg, n_frames=len(ee))
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

        t.grip_min = float(np.nanmin(grip)) if len(grip) else 1.0
        t.grip_max = float(np.nanmax(grip)) if len(grip) else 0.0
        t.jc_max = float(jc.max()) if len(jc) else 0.0

        if len(ee) > 0:
            gp = int(np.argmin(ee[:, 2]))
            t.grasp_phase_idx = gp
            t.close_idx = gp
            t.ee_at_grasp = ee[gp].tolist()

            lo = max(0, gp - 3)
            hi = min(len(ee), gp + GRASP_PHASE_WINDOW)
            win = slice(lo, hi)
            t.jc_at_grasp = float(jc[win].max()) if len(jc) else 0.0
            t.fz_spike_grasp = float(np.max(np.abs(fz[win]))) if len(fz) else 0.0
            pre_jc = (
                float(np.mean(jc[:lo])) if lo > 0 else float(jc[0]) if len(jc) else 0.0
            )
            t.jc_grasp_rise = t.jc_at_grasp - pre_jc
            t.jc_at_close = t.jc_at_grasp
            t.jc_pre_close = pre_jc
            t.jc_close_rise = t.jc_grasp_rise

            if gp < len(ee) - 1:
                grasp_ee = ee[gp]
                post_ee = ee[gp + 1 :]
                t.grasp_transport_m = (
                    float(np.max(np.linalg.norm(post_ee - grasp_ee, axis=1)))
                    if len(post_ee)
                    else 0.0
                )
                t.retract_dist_m = t.grasp_transport_m
                if len(post_ee):
                    t.retract_z_rise_m = float(post_ee[:, 2].max() - grasp_ee[2])
                t.withdraw_home_dist_m = float(np.linalg.norm(ee[-1] - ee[0]))
                t.ee_at_place = ee[-1].tolist()
                t.ee_at_place_open = t.ee_at_place
            elif len(ee):
                t.ee_at_place = ee[-1].tolist()
                t.ee_at_place_open = t.ee_at_place

        out[seg] = t
    return out


def learn_place_roi(place_ee_list: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not place_ee_list:
        return (
            np.array([-0.077, 0.512, 0.168]),
            np.array([PLACE_ROI_XY_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M]),
        )
    arr = np.array(place_ee_list)
    center = np.median(arr, axis=0)
    spread = np.std(arr, axis=0)
    tol = np.maximum(spread * 2.5, [PLACE_ROI_XY_M, PLACE_ROI_XY_M, PLACE_ROI_Z_M])
    return center, tol


def detect_grasp_attempt(t: TrialSignals, has_motion: bool) -> bool:
    """运动学抓取相位：EE 到达 z 最低点且有关爪前靠近迹象。"""
    if not has_motion or t.grasp_phase_idx is None:
        return False
    z_ok = t.approach_z_drop_m >= APPROACH_Z_DROP_M * 0.6
    xy_ok = t.approach_xy_m >= APPROACH_XY_M * 0.6
    return z_ok or xy_ok


def detect_place_phase(t: TrialSignals, has_grasp_object: bool) -> bool:
    """抓取后是否进入放置运动阶段（纯 EE，不用夹爪张开）。"""
    if not has_grasp_object:
        return False
    return (
        t.grasp_transport_m >= 0.03
        or t.retract_dist_m >= RETRACT_DIST_M * 0.5
        or t.transport_to_place
    )


def detect_grasp_contact(t: TrialSignals, jc_baseline: float) -> tuple[bool, bool, str]:
    """从 EE + wrench + joint_current 判定是否夹到物体（不依赖夹爪）。"""
    abs_ok = t.jc_at_grasp >= jc_baseline + JC_CONTACT_DELTA
    rel_ok = t.jc_grasp_rise >= JC_GRASP_RISE_MIN
    fz_ok = t.fz_spike_grasp >= FZ_CONTACT_MIN
    transport_ok = t.grasp_transport_m >= GRASP_TRANSPORT_M
    load_ok = abs_ok or rel_ok or fz_ok

    has_contact = load_ok
    has_object = has_contact or (transport_ok and load_ok)

    parts: list[str] = []
    if abs_ok:
        parts.append(f"jc={t.jc_at_grasp:.0f}≥baseline+{JC_CONTACT_DELTA}")
    if rel_ok:
        parts.append(f"Δjc={t.jc_grasp_rise:.0f}≥{JC_GRASP_RISE_MIN}")
    if fz_ok:
        parts.append(f"|fz|={t.fz_spike_grasp:.1f}≥{FZ_CONTACT_MIN}N")
    if transport_ok:
        parts.append(f"运输={t.grasp_transport_m:.3f}m")
    detail = (
        "；".join(parts)
        if parts
        else (
            f"jc={t.jc_at_grasp:.0f} Δjc={t.jc_grasp_rise:.0f} "
            f"|fz|={t.fz_spike_grasp:.1f} 运输={t.grasp_transport_m:.3f}m 未达阈值"
        )
    )
    return has_contact, has_object, detail


def detect_motion(t: TrialSignals) -> tuple[bool, str]:
    """判定机械臂是否「有效移动」。"""
    if t.ee_path_m >= MOTION_EE_PATH_M:
        return True, f"EE 轨迹 {t.ee_path_m:.3f}m ≥ {MOTION_EE_PATH_M}m"
    if t.joint_delta_rad >= MOTION_JOINT_RAD:
        return (
            True,
            f"关节累计变化 {t.joint_delta_rad:.3f}rad ≥ {MOTION_JOINT_RAD}rad（EE 位移不足）",
        )
    return (
        False,
        f"EE 轨迹 {t.ee_path_m:.3f}m、关节变化 {t.joint_delta_rad:.3f}rad 均未达阈值",
    )


def judge_trial(
    t: TrialSignals,
    *,
    place_center: np.ndarray,
    place_tol: np.ndarray,
    jc_baseline: float,
    first_frame: TrialFirstFrame | None = None,
) -> TrialJudgment:
    analyze_motion_phases(t, first_frame, place_center=place_center)

    if t.n_frames < MIN_FRAMES:
        return TrialJudgment(
            trial_index=t.trial_index,
            outcome="insufficient_log",
            semantic=SEMANTIC_LABELS["insufficient_log"],
            has_motion=False,
            has_grasp_close=False,
            has_grasp_contact=False,
            has_grasp_object=False,
            has_place_open=False,
            place_at_box=False,
            action_complete=False,
            ee_path_m=t.ee_path_m,
            joint_delta_rad=t.joint_delta_rad,
            grip_min=t.grip_min,
            grip_max=t.grip_max,
            jc_at_close=t.jc_at_close,
            n_frames=t.n_frames,
            reason=f"infer 帧数 {t.n_frames} < {MIN_FRAMES}",
            toy_detected=bool(first_frame and first_frame.toy_detected),
            cell_3x3=first_frame.cell_3x3 if first_frame else None,
            red_x_norm=first_frame.red_x_norm if first_frame else float("nan"),
            red_y_norm=first_frame.red_y_norm if first_frame else float("nan"),
        )

    has_motion, motion_reason = detect_motion(t)
    has_grasp_attempt = detect_grasp_attempt(t, has_motion)
    has_grasp_contact, has_grasp_object, grasp_detail = detect_grasp_contact(
        t, jc_baseline
    )
    if not has_grasp_attempt:
        has_grasp_contact = False
        has_grasp_object = False
    has_place_phase = detect_place_phase(t, has_grasp_object)

    place_ok = False
    place_ee = t.ee_at_place or t.ee_at_place_open
    if has_place_phase and place_ee:
        ee = np.array(place_ee)
        place_ok = bool(np.all(np.abs(ee - place_center) <= place_tol))

    action_complete = has_motion and has_grasp_object and has_place_phase
    # 兼容 CSV 列名
    has_grasp_close = has_grasp_attempt
    has_place_open = has_place_phase

    has_approach, approach_quality, approach_semantic = classify_approach(
        t, first_frame, has_motion
    )
    has_retract, has_withdraw, retract_semantic = classify_retract(
        t,
        has_grasp_attempt=has_grasp_attempt,
        has_grasp_contact=has_grasp_contact,
        has_place_phase=has_place_phase,
        place_ok=place_ok,
    )

    if not has_motion:
        motion_semantic = "几乎不移动"
    elif approach_quality == "wander":
        motion_semantic = "无序运动偏离目标"
    elif has_approach:
        motion_semantic = approach_semantic
    else:
        motion_semantic = "有移动但未有效靠近"

    if not has_grasp_attempt:
        grasp_semantic = "无抓取相位"
    elif has_grasp_contact or has_grasp_object:
        grasp_semantic = "抓取有接触/负载"
    else:
        grasp_semantic = "尝试抓取但抓空"

    phase_chain = build_phase_chain(
        has_motion=has_motion,
        approach_quality=approach_quality,
        has_grasp_attempt=has_grasp_attempt,
        has_grasp_contact=has_grasp_contact,
        has_grasp_object=has_grasp_object,
        has_retract=has_retract,
        retract_semantic=retract_semantic,
        has_place_phase=has_place_phase,
        place_ok=place_ok,
    )

    ff_note = ""
    if first_frame and first_frame.toy_detected:
        ff_note = f"首帧玩具 cell={first_frame.cell_3x3} ({first_frame.red_x_norm:.2f},{first_frame.red_y_norm:.2f})；"

    if not has_motion:
        outcome = "no_motion"
        reason = f"{ff_note}{motion_reason}"
    elif not has_grasp_attempt:
        if approach_quality == "wander":
            outcome = "motion_no_grasp"
            reason = (
                f"{ff_note}{motion_reason}；靠近对齐={t.approach_align:.2f}、"
                f"路径效率={t.path_efficiency:.2f}，判为无序偏离；未达抓取相位"
            )
        elif not has_approach:
            outcome = "motion_no_grasp"
            reason = (
                f"{ff_note}{motion_reason}；z下降={t.approach_z_drop_m:.3f}m、"
                f"xy位移={t.approach_xy_m:.3f}m，未有效靠近"
            )
        else:
            outcome = "motion_no_grasp"
            reason = f"{ff_note}{approach_semantic}但未达抓取相位（z↓={t.approach_z_drop_m:.3f}m）"
    elif not has_grasp_object:
        outcome = "grasp_attempt_fail"
        reason = f"{ff_note}{approach_semantic}→抓取相位但无负载信号（{grasp_detail}）"
    elif not has_place_phase:
        outcome = "grasp_no_place"
        retract_note = f"；{retract_semantic}" if has_retract else ""
        reason = (
            f"{ff_note}有接触但抓取后运输不足"
            f"（transport={t.grasp_transport_m:.3f}m）{retract_note}"
        )
    elif not place_ok:
        outcome = "place_miss"
        reason = f"{ff_note}放置阶段末段 EE={place_ee} 不在 ROI"
    else:
        outcome = "action_complete_ok"
        reason = f"{ff_note}{phase_chain}"

    return TrialJudgment(
        trial_index=t.trial_index,
        outcome=outcome,
        semantic=SEMANTIC_LABELS[outcome],
        has_motion=has_motion,
        has_grasp_close=has_grasp_close,
        has_grasp_contact=has_grasp_contact,
        has_grasp_object=has_grasp_object,
        has_place_open=has_place_open,
        place_at_box=place_ok,
        action_complete=action_complete,
        ee_path_m=t.ee_path_m,
        joint_delta_rad=t.joint_delta_rad,
        grip_min=t.grip_min,
        grip_max=t.grip_max,
        jc_at_close=t.jc_at_close,
        jc_at_grasp=t.jc_at_grasp,
        fz_spike_grasp=t.fz_spike_grasp,
        jc_close_rise=t.jc_close_rise,
        jc_grasp_rise=t.jc_grasp_rise,
        grasp_transport_m=t.grasp_transport_m,
        n_frames=t.n_frames,
        reason=reason,
        toy_detected=bool(first_frame and first_frame.toy_detected),
        cell_3x3=first_frame.cell_3x3 if first_frame else None,
        red_x_norm=first_frame.red_x_norm if first_frame else float("nan"),
        red_y_norm=first_frame.red_y_norm if first_frame else float("nan"),
        has_approach=has_approach,
        approach_quality=approach_quality,
        approach_semantic=approach_semantic,
        motion_semantic=motion_semantic,
        grasp_semantic=grasp_semantic,
        retract_semantic=retract_semantic,
        has_retract=has_retract,
        has_withdraw_home=has_withdraw,
        phase_chain=phase_chain,
    )


def analyze_test_dir(
    test_dir: Path, *, skip_video: bool = False
) -> list[TrialJudgment]:
    test_dir = test_dir.resolve()
    frame_seg = load_frame_to_segment(test_dir)
    if not frame_seg:
        return []
    signals = ingest_eval_log(test_dir, frame_seg)
    if not signals:
        return []

    first_frames = load_trial_first_frames(test_dir, skip_video=skip_video)

    raw_signals = list(signals.values())
    place_ee = []
    for t in raw_signals:
        ee = t.ee_at_place or t.ee_at_place_open
        if not ee:
            continue
        load = (
            t.jc_grasp_rise >= JC_GRASP_RISE_MIN
            or t.fz_spike_grasp >= FZ_CONTACT_MIN
            or t.jc_at_grasp >= 3000
        )
        if t.grasp_transport_m >= GRASP_TRANSPORT_M * 0.5 and load:
            place_ee.append(ee)
    place_center, place_tol = learn_place_roi(place_ee)
    jc_vals = [t.jc_max for t in raw_signals if t.jc_max > 0]
    jc_baseline = float(np.percentile(jc_vals, 25)) if jc_vals else 3000.0

    return [
        judge_trial(
            t,
            place_center=place_center,
            place_tol=place_tol,
            jc_baseline=jc_baseline,
            first_frame=first_frames.get(t.trial_index),
        )
        for t in sorted(signals.values(), key=lambda x: x.trial_index)
    ]


def discover_test_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for elog in sorted(root.rglob("eval_log.jsonl")):
        if "_batch_staging" in elog.parts:
            continue
        if elog.stat().st_size < 200:
            continue
        td = elog.parent.resolve()
        key = str(td)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(td)
    return dirs


def print_summary(test_dir: Path, judgments: list[TrialJudgment]) -> None:
    print(f"\n{'=' * 72}")
    print(f"TEST: {test_dir}")
    print(f"{'=' * 72}")
    if not judgments:
        print(
            "  （无 trial：缺少 eval_log.jsonl 或 task_segments / machine_flow 映射）"
        )
        return

    n = len(judgments)
    oc = Counter(j.outcome for j in judgments)
    motion_n = sum(1 for j in judgments if j.has_motion)

    print(f"  trials={n}  能动={motion_n}/{n} ({motion_n / n:.1%})")
    approach_n = sum(1 for j in judgments if j.has_approach)
    retract_n = sum(1 for j in judgments if j.has_retract)
    toy_n = sum(1 for j in judgments if j.toy_detected)
    print(
        f"  首帧检测到玩具={toy_n}/{n}  有效靠近={approach_n}/{n}  抓取后撤回={retract_n}/{n}"
    )
    aq = Counter(j.approach_quality for j in judgments)
    print("  靠近/定位分布:", ", ".join(f"{k}={v}" for k, v in aq.most_common()))
    print("  outcome 分布:")
    for k, v in oc.most_common():
        print(f"    {k:22} {v:3} ({v / n:.1%})  — {SEMANTIC_LABELS.get(k, k)}")

    print("\n  trial 明细:")
    for j in judgments:
        flag = "✓动" if j.has_motion else "✗静"
        cell = j.cell_3x3 or "?"
        print(
            f"    #{j.trial_index:2d} [{flag}] cell={cell:6} {j.outcome:22} | "
            f"{j.phase_chain}"
        )
        print(
            f"         └ 定位:{j.approach_semantic} | 抓取:{j.grasp_semantic} | "
            f"撤回:{j.retract_semantic} | {j.reason}"
        )


def write_outputs(out_dir: Path, all_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trial_semantics_judgment.csv"
    json_path = out_dir / "trial_semantics_judgment.json"

    if not all_rows:
        return

    import csv

    fields = list(all_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    summary = {
        "n_trials": len(all_rows),
        "outcome_counts": dict(Counter(r["outcome"] for r in all_rows)),
        "motion_rate": sum(1 for r in all_rows if r["has_motion"]) / len(all_rows),
        "trials": all_rows,
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test-dir", type=Path, help="单个 eval 目录（含 eval_log.jsonl）"
    )
    parser.add_argument(
        "--root", type=Path, help="扫描根目录（默认 research/ablation/runs/real）"
    )
    parser.add_argument("--out-dir", type=Path, help="批量模式 CSV/JSON 输出目录")
    parser.add_argument("--explain", action="store_true", help="打印判定规则后退出")
    parser.add_argument(
        "--quiet", action="store_true", help="批量模式不打印每个 test 明细"
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="跳过首帧视频解析（不做定位/格位判定，仅 EE 轨迹语义）",
    )
    args = parser.parse_args()

    if args.explain:
        print_rules()
        return 0

    if args.test_dir:
        test_dirs = [args.test_dir.resolve()]
    elif args.root:
        test_dirs = discover_test_dirs(args.root.resolve())
    else:
        default_root = (
            Path(__file__).resolve().parents[2] / "research/ablation/runs/real"
        )
        test_dirs = discover_test_dirs(default_root)

    if not test_dirs:
        print("未找到 eval_log.jsonl", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for td in test_dirs:
        judgments = analyze_test_dir(td, skip_video=args.skip_video)
        if not args.quiet or args.test_dir:
            print_summary(td, judgments)
        for j in judgments:
            row = j.to_dict()
            row["test_dir"] = str(td)
            all_rows.append(row)

    if all_rows:
        n = len(all_rows)
        oc = Counter(r["outcome"] for r in all_rows)
        motion_rate = sum(1 for r in all_rows if r["has_motion"]) / n
        print(f"\n{'=' * 72}")
        print(f"GLOBAL: {len(test_dirs)} tests, {n} trials, 能动率={motion_rate:.1%}")
        for k, v in oc.most_common():
            print(f"  {k:22} {v:4} ({v / n:.1%})")

    if args.out_dir and all_rows:
        write_outputs(args.out_dir.resolve(), all_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
