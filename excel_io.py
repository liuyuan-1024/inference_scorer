"""
Excel 读写 — 读模板、写评分、保存结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

import openpyxl

from config import (
    COL_INDEX,
    DIM_ROW,
    EXCEL_FILENAME,
    EXCEL_SHEET_NAME,
    RULES_TEXT,
    TASK_TO_COL,
)


def fill_excel(
    excel_path: Path,
    output_path: Path,
    scores: dict[int, dict[str, int]],
) -> None:
    """将评分写入 Excel（仅填写 S1~S5 分值，不动其他单元格）。"""
    wb = openpyxl.load_workbook(str(excel_path))
    ws = wb[EXCEL_SHEET_NAME]

    # 写入评分
    for task_idx, dim_scores in sorted(scores.items()):
        col_letter = TASK_TO_COL.get(task_idx)
        if col_letter is None:
            print(f"  ⚠ 跳过 task {task_idx}：无对应 Excel 列", file=sys.stderr)
            continue
        col_idx = COL_INDEX[col_letter]

        for dim_name, score in dim_scores.items():
            row = DIM_ROW.get(dim_name)
            if row is None:
                continue
            ws.cell(row, col_idx).value = score

    wb.save(str(output_path))
    print(f"  已保存: {output_path}")


def print_rules() -> None:
    """打印评分规则。"""
    print(RULES_TEXT)
