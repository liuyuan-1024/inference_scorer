"""类型化评分结果的 JSON 输出适配器。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from domain.models import ScoringRun


def _task_evidence(item) -> dict:
    """提取适合审计和人工阅读的 task 级证据。"""
    return {
        "has_motion": item.has_motion,
        "has_grasp_attempt": item.has_grasp_attempt,
        "has_grasp_contact": item.has_grasp_contact,
        "has_grasp_object": item.has_grasp_object,
        "object_lifted": item.events.object_lifted,
        "object_dropped": item.events.object_dropped,
        "has_place_phase": item.has_place_phase,
        "place_at_box": item.place_at_box,
        "approach_quality": item.approach_quality,
        "scenario": item.vision.scenario,
        "vision_available": item.vision.available,
        "object_present": item.vision.object_present,
        "box_present": item.vision.box_present,
        "final_object_relation": item.vision.final_object_relation,
        "vision_confidence": item.vision.confidence,
        "sensor_confidence": item.sensor_confidence,
        "state_feedback_timeout_rate": item.state_feedback_timeout_rate,
        "events": asdict(item.events),
        "observations": item.evidence,
        "reason": item.reason,
    }


def build_summary(run: ScoringRun) -> dict:
    """按 task 组织阶段评分、置信度和证据。"""
    evidence_by_task = {item.task_index: item for item in run.evidence}
    return {
        "data_dir": run.data_dir,
        "n_tasks": len(run.evidence),
        "n_scored": len(run.tasks),
        "rule_version": run.rule_version,
        "review_count": sum(
            int(result.needs_review)
            for task in run.tasks.values()
            for result in task.stages.values()
        ),
        "tasks": [
            {
                "task_index": task_id,
                "total_score": task.total,
                "max_score": sum(result.max_score for result in task.stages.values()),
                "needs_review": any(
                    result.needs_review for result in task.stages.values()
                ),
                "stages": {
                    name: result.to_dict() for name, result in task.stages.items()
                },
                "task_evidence": _task_evidence(evidence_by_task[task_id]),
            }
            for task_id, task in sorted(run.tasks.items())
        ],
    }


def save_summary(data_dir: Path, run: ScoringRun) -> Path:
    """保存自动评分摘要并返回输出路径。"""
    output = data_dir / "auto_score_summary.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(build_summary(run), stream, ensure_ascii=False, indent=2)
    return output
