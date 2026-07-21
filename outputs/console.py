"""终端可读的证据、评分和复核信息。"""

from domain.models import TaskEvidence, TaskScore
from domain.rules_v5 import V5_STAGE_BY_KEY, V5_STAGE_SPECS


def print_evidence_summary(evidence: tuple[TaskEvidence, ...]) -> None:
    """打印各 task 的判定摘要。"""
    for item in evidence:
        flag = "✓动" if item.has_motion else "✗静"
        grasp = (
            "抓"
            if item.has_grasp_object
            else (
                "触"
                if item.has_grasp_contact
                else ("试" if item.has_grasp_attempt else "无")
            )
        )
        place = "放" if item.has_place_phase else "—"
        returning = (
            "归" if item.has_withdraw_home else ("撤" if item.has_retract else "—")
        )
        print(
            f"  #{item.task_index:2d} [{flag}] 靠近:{item.approach_quality:6} "
            f"抓取:{grasp} 放置:{place} 撤回:{returning} | {item.reason}"
        )


def print_verbose_table(evidence: tuple[TaskEvidence, ...]) -> None:
    """打印详细指标表。"""
    print(f"\n{'─' * 90}")
    header = (
        f"{'#':>3} {'EE_path':>8} {'J_delta':>7} {'z_drop':>7} {'xy_move':>7} "
        f"{'align':>6} {'eff':>5} {'frac':>5} {'jc_grasp':>8} {'jc_rise':>7} "
        f"{'fz':>5} {'transport':>9} {'retract':>7} {'withdraw':>8}"
    )
    print(header)
    print(f"{'─' * 90}")
    for item in evidence:
        print(
            f"{item.task_index:>3} {item.ee_path_m:>8.3f} "
            f"{item.joint_delta_rad:>7.3f} {item.approach_z_drop_m:>7.4f} "
            f"{item.approach_xy_m:>7.4f} {item.approach_align:>6.2f} "
            f"{item.path_efficiency:>5.2f} {item.approach_frame_frac:>5.2f} "
            f"{item.jc_at_grasp:>8.0f} {item.jc_grasp_rise:>7.0f} "
            f"{item.fz_spike_grasp:>5.1f} {item.grasp_transport_m:>9.4f} "
            f"{item.retract_dist_m:>7.4f} {item.withdraw_home_dist_m:>8.4f}"
        )


def print_score_table(tasks: dict[int, TaskScore]) -> None:
    """打印评分表（行为维度，列为 task，与 Excel 布局一致）。"""
    dimensions = [spec.key for spec in V5_STAGE_SPECS]
    task_ids = sorted(tasks)
    header = f"{'维度':>6}  "
    header += "  ".join(f"{f'test{task_id}':>6}" for task_id in task_ids)
    header += f"  {'满分':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for dimension in dimensions:
        line = f"{dimension:>6}  "
        for task_id in task_ids:
            line += f"{tasks[task_id].stages[dimension].score:>6}  "
        line += f"{V5_STAGE_BY_KEY[dimension].max_score:>6}"
        print(line)

    total_line = f"{'总分':>6}  "
    for task_id in task_ids:
        total_line += f"{tasks[task_id].total:>6}  "
    total_line += f"{'14':>6}"
    print(total_line)


def print_review_items(tasks: dict[int, TaskScore]) -> int:
    """打印需要人工复核的具体 task、维度和原因。"""
    review_items = []
    for task_index in sorted(tasks):
        for dimension, result in tasks[task_index].stages.items():
            if result.needs_review:
                review_items.append((task_index, dimension, result))

    if not review_items:
        print("\n无需人工复核。")
        return 0

    print(f"\n待人工复核明细（{len(review_items)} 项）:")
    for task_index, dimension, result in review_items:
        reason = result.review_reason or "评分置信度不足"
        print(
            f"  - test{task_index} / {dimension} "
            f"(置信度 {result.confidence:.0%}): {reason}"
        )
    return len(review_items)
