#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inputs.eval_log import load_meta
from outputs.config import EXCEL_FILENAME
from outputs.console import (
    print_evidence_summary,
    print_review_items,
    print_score_table,
    print_verbose_table,
)
from outputs.excel import fill_excel, print_rules
from outputs.json import save_summary
from pipeline import evaluate_data_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="自动评分 — 读取推理评测数据，填写机械臂抓取模型反馈评分模型.xlsx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("0870000/action-step_70"),
        help="data 目录（含 eval_log.jsonl）",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Excel 模板路径，指定后将复制到 data-dir 后再修改（默认: data-dir 下必须有 机械臂抓取模型反馈评分模型.xlsx）",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="打印评分规则后退出",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细分析信息",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="只计算评分并输出 JSON，不要求或修改 Excel",
    )

    args = parser.parse_args()

    if args.explain:
        print_rules()
        return 0

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"错误: 目录不存在 {data_dir}", file=sys.stderr)
        return 1

    excel_path = data_dir / EXCEL_FILENAME

    if args.no_excel and args.excel:
        print("错误: --no-excel 与 --excel 不能同时使用", file=sys.stderr)
        return 1
    if args.excel:
        # 从指定模板复制到 data_dir
        template = args.excel.resolve()
        if not template.is_file():
            print(f"错误: 模板文件不存在 {template}", file=sys.stderr)
            return 1
        import shutil

        shutil.copy2(str(template), str(excel_path))
        print(f"  已从模板复制: {template} → {excel_path}")
    elif not args.no_excel and not excel_path.is_file():
        print(
            f"错误: {data_dir} 下没有找到 {EXCEL_FILENAME}\n"
            f"      请用 --excel 指定模板路径",
            file=sys.stderr,
        )
        return 1

    print(f"评测目录: {data_dir}")
    print(
        f"Excel 文件: {excel_path.name}" if not args.no_excel else "Excel 输出: 已禁用"
    )

    # --- Step 1: 分析 task ---
    print("\n正在分析 task 数据...")
    run = evaluate_data_dir(data_dir)
    if not run.evidence:
        print("错误: 未获取到有效 task 判定", file=sys.stderr)
        return 1

    print(f"\n共 {len(run.evidence)} 个 task:")
    print_evidence_summary(run.evidence)

    if args.verbose:
        print_verbose_table(run.evidence)

    # --- Step 2: 计算评分 ---
    print("\n正在计算评分...")
    print_score_table(run.tasks)
    print_review_items(run.tasks)

    # --- Step 3: 写入 Excel ---
    meta = load_meta(data_dir)
    metadata = {
        **meta,
        "video_path": str(data_dir / "videos"),
    }
    if not args.no_excel:
        print("\n正在写入 Excel...")
        fill_excel(
            excel_path,
            excel_path,
            run.tasks,
            metadata=metadata,
        )

    # --- Step 4: 输出摘要 JSON ---
    summary_path = save_summary(data_dir, run)
    print(f"\n摘要已保存: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
