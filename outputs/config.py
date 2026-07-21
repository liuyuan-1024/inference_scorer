"""Excel 文件名与工作表布局配置。"""

from __future__ import annotations

DIM_ROW: dict[str, int] = {
    "S1定位": 6,
    "S2抓取": 11,
    "S3搬运": 15,
    "S4投放": 18,
    "S5归位": 22,
}

TASK_TO_COLUMN: dict[int, int] = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10}

EXCEL_FILENAME = "机械臂抓取模型反馈评分模型.xlsx"
EXCEL_SHEET_NAME = "推理 (2)"
