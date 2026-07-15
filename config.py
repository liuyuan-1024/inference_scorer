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

# EE 轨迹 ≥ 0.20m → 有移动
MOTION_EE_PATH_M: float = 0.20
# 关节累计 ≥ 0.15rad → 有移动（EE 缺失时备用）
MOTION_JOINT_RAD: float = 0.15

# =========================== 抓取接触阈值 ===========================

# 抓取相位前后帧数（窗口）
GRASP_PHASE_WINDOW: int = 8
# 抓取窗口 max|fz| ≥ 5N → 接触
FZ_CONTACT_MIN: float = 5.0
# max(jc) ≥ jc_p25 + 400 → 绝对负载
JC_CONTACT_DELTA: int = 400
# jc 尖峰 − 抓取前均值 ≥ 600 → 相对负载
JC_GRASP_RISE_MIN: float = 600
# 抓取后 EE 累计位移 ≥ 0.06m 且负载 → 抓上并带走
GRASP_TRANSPORT_M: float = 0.06

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
# 末端距起始 ≤ 0.12m → 归位
WITHDRAW_HOME_M: float = 0.12

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

# =========================== 通用阈值 ===========================

MIN_FRAMES: int = 3  # task 最少帧数
DEFAULT_FPS: float = 15.0  # 默认视频帧率


# =========================== 放置 ROI 默认估计 ===========================

# 当无法从数据学习时使用的默认放置中心
DEFAULT_PLACE_CENTER: list[float] = [-0.077, 0.512, 0.168]

# =========================== 评分细则（用于 print_rules） ===========================

RULES_TEXT: str = """
============================================================
  机械臂抓取模型 自动评分规则（v5）
============================================================

S1定位 (0-4):
  0 = 10s内几乎没有动作         → has_motion=False
  1 = 无规律乱动               → has_motion=True, approach_quality=wander/none
  2 = 定位偏移                 → approach_quality=slow 且 >10s 或效率低
  3 = 缓慢靠近目标（5-10s）    → approach_quality=slow, 5-10s
  4 = 5s内快速准确定位         → approach_quality=fast

S2抓取 (0-3):
  0 = 未抓取                   → has_grasp_attempt=False
  1 = 抓取失败（抓空）          → has_grasp_attempt=True, has_grasp_contact=False
  2 = 抬起掉落                 → has_grasp_contact=True, has_grasp_object=False
  3 = 稳定抓取                 → has_grasp_object=True (接触+运输)

S3搬运 (0-2):
  0 = 未搬运                   → grasp_transport_m < 0.05m
  1 = 搬运未到位               → grasp_transport_m ≥ 0.05m 但未到放置区
  2 = 到达目标上方             → transport_to_place=True 且 retract_dist_m ≥ 0.10m

S4投放 (0-3):
  0 = 无投放                   → has_place_phase=False
  1 = 落于盒外                 → has_place_phase=True, place_at_box=False (距ROI>2cm)
  2 = 落于盒边                 → place_at_box=False 但距 ROI ≤ 2cm
  3 = 准确入盒                 → place_at_box=True

S5归位 (0-2):
  0 = 未归位                   → has_retract=False
  1 = 归位偏差                 → has_retract=True, withdraw_home_dist_m > 0.12m
  2 = 成功归位                 → withdraw_home_dist_m ≤ 0.12m
============================================================
"""
