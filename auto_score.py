#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config import DEFAULT_FPS, EXCEL_FILENAME
from analysis import analyze_data_dir
from excel_io import fill_excel, print_rules
from scoring import compute_all_scores


def print_judgments(judgments: list) -> None:
    """打印各 task 的判定摘要。"""
    for j in judgments:
        flag = "✓动" if j.has_motion else "✗静"
        grasp = (
            "抓"
            if j.has_grasp_object
            else (
                "触" if j.has_grasp_contact else ("试" if j.has_grasp_attempt else "无")
            )
        )
        place = "放" if j.has_place_phase else "—"
        ret = "归" if j.has_withdraw_home else ("撤" if j.has_retract else "—")
        print(
            f"  #{j.task_index:2d} [{flag}] 靠近:{j.approach_quality:6} "
            f"抓取:{grasp} 放置:{place} 撤回:{ret} "
            f"| {j.reason}"
        )


def print_verbose_table(judgments: list) -> None:
    """打印详细指标表。"""
    print(f"\n{'─' * 90}")
    header = (
        f"{'#':>3} {'EE_path':>8} {'J_delta':>7} {'z_drop':>7} {'xy_move':>7} "
        f"{'align':>6} {'eff':>5} {'frac':>5} {'jc_grasp':>8} {'jc_rise':>7} "
        f"{'fz':>5} {'transport':>9} {'retract':>7} {'withdraw':>8}"
    )
    print(header)
    print(f"{'─' * 90}")
    for j in judgments:
        print(
            f"{j.task_index:>3} {j.ee_path_m:>8.3f} {j.joint_delta_rad:>7.3f} "
            f"{j.approach_z_drop_m:>7.4f} {j.approach_xy_m:>7.4f} "
            f"{j.approach_align:>6.2f} {j.path_efficiency:>5.2f} {j.approach_frame_frac:>5.2f} "
            f"{j.jc_at_grasp:>8.0f} {j.jc_grasp_rise:>7.0f} "
            f"{j.fz_spike_grasp:>5.1f} {j.grasp_transport_m:>9.4f} "
            f"{j.retract_dist_m:>7.4f} {j.withdraw_home_dist_m:>8.4f}"
        )


def print_score_table(scores: dict[int, dict[str, int]]) -> None:
    """打印评分表（行为维度，列为 task，与 Excel 布局一致）。"""
    DIMS = ["S1定位", "S2抓取", "S3搬运", "S4投放", "S5归位"]
    task_ids = sorted(scores.keys())
    col_width = 8

    # 表头
    header = f"{'维度':>6}  "
    header += "  ".join(f"{f'test{t}':>6}" for t in task_ids)
    header += f"  {'满分':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    # 每行一个维度
    max_scores = {"S1定位": 4, "S2抓取": 3, "S3搬运": 2, "S4投放": 3, "S5归位": 2}
    for dim in DIMS:
        line = f"{dim:>6}  "
        for t in task_ids:
            line += f"{scores[t][dim]:>6}  "
        line += f"{max_scores[dim]:>6}"
        print(line)

    # 总分行
    total_line = f"{'总分':>6}  "
    for t in task_ids:
        total_line += f"{sum(scores[t].values()):>6}  "
    total_line += f"{'14':>6}"
    print(total_line)


def save_summary(data_dir: Path, judgments: list, scores: dict) -> None:
    """保存评分摘要 JSON。"""
    summary_path = data_dir / "auto_score_summary.json"
    summary = {
        "data_dir": str(data_dir),
        "n_tasks": len(judgments),
        "n_scored": len(scores),
        "scores": {},
        "judgments": [
            {
                "task_index": j.task_index,
                "has_motion": j.has_motion,
                "has_grasp_object": j.has_grasp_object,
                "has_place_phase": j.has_place_phase,
                "place_at_box": j.place_at_box,
                "approach_quality": j.approach_quality,
                "reason": j.reason,
            }
            for j in judgments
        ],
    }
    for tid, ds in scores.items():
        summary["scores"][str(tid)] = {"scores": ds, "total": sum(ds.values())}

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n摘要已保存: {summary_path}")


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
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="视频帧率（默认 15）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细分析信息",
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

    if args.excel:
        # 从指定模板复制到 data_dir
        template = args.excel.resolve()
        if not template.is_file():
            print(f"错误: 模板文件不存在 {template}", file=sys.stderr)
            return 1
        import shutil

        shutil.copy2(str(template), str(excel_path))
        print(f"  已从模板复制: {template} → {excel_path}")
    elif not excel_path.is_file():
        print(
            f"错误: {data_dir} 下没有找到 {EXCEL_FILENAME}\n"
            f"      请用 --excel 指定模板路径",
            file=sys.stderr,
        )
        return 1

    print(f"评测目录: {data_dir}")
    print(f"Excel 文件: {excel_path.name}")

    # --- Step 1: 分析 task ---
    print("\n正在分析 task 数据...")
    judgments = analyze_data_dir(data_dir)
    if not judgments:
        print("错误: 未获取到有效 task 判定", file=sys.stderr)
        return 1

    print(f"\n共 {len(judgments)} 个 task:")
    print_judgments(judgments)

    if args.verbose:
        print_verbose_table(judgments)

    # --- Step 2: 计算评分 ---
    print("\n正在计算评分...")
    scores = compute_all_scores(judgments, args.fps)
    print_score_table(scores)

    # --- Step 3: 写入 Excel ---
    print("\n正在写入 Excel...")
    fill_excel(excel_path, excel_path, scores)

    # --- Step 4: 输出摘要 JSON ---
    save_summary(data_dir, judgments, scores)

    # 汇总统计
    totals = [sum(scores[t].values()) for t in sorted(scores.keys())]
    print(f"\n{'=' * 50}")
    print(f"完成! 共评分 {len(scores)} 个 task")
    if totals:
        print(f"   平均总分: {np.mean(totals):.1f} / 14.0")
        print(f"   最高分: {max(totals)}  最低分: {min(totals)}")
    print(f"   输出文件: {excel_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
