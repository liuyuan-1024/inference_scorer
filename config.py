"""
配置文件 — 所有可调阈值、维度定义、Excel 映射。

修改这里的数值即可调参，无需改动核心逻辑。
"""

from __future__ import annotations

# =========================== Excel 评分维度定义 ===========================

# 每个维度: (data_row, max_score, name)
SCORE_DIMS: list[tuple[int, int, str]] = [
    (6, 4, "S1定位"),
    (11, 3, "S2抓取"),
    (15, 2, "S3搬运"),
    (18, 3, "S4投放"),
    (22, 2, "S5归位"),
]

# 维度名 → Excel 数据行号
DIM_ROW: dict[str, int] = {
    "S1定位": 6,
    "S2抓取": 11,
    "S3搬运": 15,
    "S4投放": 18,
    "S5归位": 22,
}

# Trial → Excel 列字母
TASK_TO_COL: dict[int, str] = {1: "E", 2: "F", 3: "G", 4: "H", 5: "I", 6: "J"}

# 列字母 → openpyxl 列索引
COL_INDEX: dict[str, int] = {"E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}

# =========================== Excel 文件结构 ===========================

# 评分 Excel 文件名（每个 data-dir 下固定叫这个名字）
EXCEL_FILENAME: str = "机械臂抓取模型反馈评分模型.xlsx"

# Sheet 名称
EXCEL_SHEET_NAME: str = "推理 (2)"


# =========================== 运动检测阈值 ===========================

# v5: 任一关节相对初始位置变化 ≥ 1° → 有移动
MOTION_JOINT_EXCURSION_RAD: float = 0.017453292519943295
# EE 相对初始位置移动 ≥ 1cm → 有移动
MOTION_EE_EXCURSION_M: float = 0.01

# =========================== 抓取接触阈值 ===========================

# 抓取相位前后帧数（窗口）
GRASP_PHASE_WINDOW: int = 8
# 抓取窗口 Fz 相对抓取前中位数变化 ≥ 0.8N → 接触候选
FZ_CONTACT_DELTA_MIN: float = 0.8
# 单关节电流相对抓取前中位数变化 ≥ 800 → 接触候选
JC_CONTACT_DELTA: int = 800
# JC fallback 仅在没有可靠夹爪反馈时启用
JC_GRASP_RISE_MIN: float = 900
# 抓取后 EE 累计位移 ≥ 0.06m 且负载 → 抓上并带走
GRASP_TRANSPORT_M: float = 0.06
# v5 稳定抓取要求物体抬离桌面 ≥ 5cm
GRASP_LIFT_M: float = 0.05

# =========================== 靠近阶段阈值 ===========================

# z 下降 ≥ 0.04m
APPROACH_Z_DROP_M: float = 0.04
# xy 位移 ≥ 0.05m
APPROACH_XY_M: float = 0.05
# 方向对齐 ≥ 0.30
APPROACH_ALIGN_MIN: float = 0.30
# 路径效率 ≥ 0.30
PATH_EFFICIENCY_MIN: float = 0.30
# 前 40% 帧到达 z 最低点 → fast
FAST_APPROACH_FRAC: float = 0.40

# =========================== 撤回/归位阈值 ===========================

# 抓取后离抓取点 ≥ 0.06m
RETRACT_DIST_M: float = 0.06
# z 抬升 ≥ 0.035m
RETRACT_Z_RISE_M: float = 0.035
# v5: 归位水平误差 ≤ 10cm、垂直误差 ≤ 5cm
WITHDRAW_HOME_XY_M: float = 0.10
WITHDRAW_HOME_Z_M: float = 0.05
# 必须先离开初始位置，再至少向初始位置回撤 3cm
RETURN_MIN_EXCURSION_M: float = 0.08
RETURN_MIN_PROGRESS_M: float = 0.03
RETURN_MAX_SEC: float = 10.0
# 评分细则 E23 写“张开”，与关键术语中的“闭合”冲突；此处显式选择 E23。
RETURN_GRIP_OPEN_REQUIRED: bool = True

# =========================== 放置 ROI 阈值 ===========================

# 放置 ROI xy 容差
PLACE_ROI_XY_M: float = 0.06
# 放置 ROI z 容差
PLACE_ROI_Z_M: float = 0.05
# "落于盒边" 距 ROI 边缘的最大距离
PLACE_ROI_EDGE_M: float = 0.02

# =========================== 夹爪检测阈值 ===========================

# 夹爪"打开"阈值（值 > 此值视为张开）
GRIP_OPEN_THRESHOLD: float = 0.70
# 夹爪"闭合"阈值（值 < 此值视为闭合）
GRIP_CLOSE_THRESHOLD: float = 0.50
# 夹爪释放检测：抓取后张开幅度 ≥ 此值
GRIP_RELEASE_RISE: float = 0.15
# 夹爪释放检测：抓取前必须闭合程度
GRIP_PRE_CLOSE_MAX: float = 0.30
# 夹爪完全闭合到此阈值通常表示空夹
GRIP_EMPTY_CLOSED_MAX: float = 0.10

# =========================== 轻量视觉阈值 ===========================

VISION_SAMPLE_STRIDE: int = 6
VISION_WIDTH: int = 320
VISION_HEIGHT: int = 180
VISION_MIN_OBJECT_PIXELS: int = 45
VISION_MIN_BOX_PIXELS: int = 250
VISION_PRESENT_RATE: float = 0.35
VISION_OBJECT_MOTION_NORM: float = 0.02
VISION_TOWARD_BOX_NORM: float = 0.03

# =========================== 通用阈值 ===========================

MIN_FRAMES: int = 3  # task 最少帧数
DEFAULT_FPS: float = 15.0  # 默认视频帧率


# =========================== 放置 ROI 默认估计 ===========================

# 当无法从数据学习时使用的默认放置中心
DEFAULT_PLACE_CENTER: list[float] = [-0.077, 0.512, 0.168]

# =========================== 评分细则（用于 print_rules） ===========================

RULES_TEXT: str = """
============================================================
  机械臂抓取模型 自动评分规则（v5 融合实现）
============================================================

S1定位 (0-4):
  0 = 所有关节变化 <1° 且 EE 位移 <1cm
  1 = 有移动但未形成有效靠近证据，或无目标物时仍移动
  2 = 向目标区域靠近但未准确到位
  3 = 5-10s 内到达可抓取区域
  4 = 5s 内快速到达可抓取区域
  注：缺少相机厘米级标定时，S1=3/4 自动标记人工复核

S2抓取 (0-3):
  0 = 无闭合事件，或视频确认没有目标物
  1 = 执行闭合但空夹/物体未随动
  2 = 有接触和短暂抬升，但未满足 5cm 稳定抓取
  3 = 非空夹 + 物体随动 + 抬升≥5cm + 稳定运输

S3搬运 (0-2):
  0 = 无稳定抓取、无搬运，或 OOD 场景无盒子
  1 = 物体向盒子移动但未到盒口
  2 = 物体到达盒口区域且未掉落

S4投放 (0-3):
  0 = 未稳定抓取/未释放，或 OOD 场景无盒子
  1 = 释放后物体在盒外
  2 = 物体与盒边相交/卡边
  3 = 物体整体位于盒内

S5归位 (0-2):
  0 = 未形成“离开初始位置→返回”的明确轨迹
  1 = 有返回动作，但位置/时间/夹爪状态至少一项不达标
  2 = xy≤10cm、z≤5cm、10s内完成且夹爪张开

每个分数同时输出 confidence、evidence、needs_review，并写入 Excel 批注。
============================================================
"""
