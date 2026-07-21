import unittest
from unittest.mock import patch

from domain import models
from domain.models import (
    ScoringRun,
    StageResult,
    TaskEvidence,
    TaskScore,
)
from domain.rules_v5 import V5_RULES, V5_STAGE_SPECS
from outputs.json import build_summary
from pipeline import evaluate_data_dir
from inputs.segments import TaskSegmentAssignments
from stages.common import stage_confidence
from stages.scoring import compute_task_scores


def evidence(task_index: int = 1) -> TaskEvidence:
    return TaskEvidence(
        task_index=task_index,
        n_frames=10,
        sensor_confidence=0.9,
        has_motion=False,
        has_grasp_attempt=False,
        has_grasp_contact=False,
        has_grasp_object=False,
        has_place_phase=False,
        place_at_box=False,
        action_complete=False,
        approach_quality="none",
        retract_semantic="none",
        has_retract=False,
        has_withdraw_home=False,
        reason="test",
    )


class ScoringPipelineTests(unittest.TestCase):
    def test_task_evidence_is_the_only_task_domain_model(self) -> None:
        self.assertFalse(hasattr(models, "TaskSignals"))
        self.assertFalse(hasattr(models, "TaskJudgment"))

    def test_all_v5_stages_return_typed_results(self) -> None:
        task_score = compute_task_scores([evidence()])[1]

        self.assertEqual(list(task_score.stages), [s.key for s in V5_STAGE_SPECS])
        self.assertTrue(
            all(isinstance(v, StageResult) for v in task_score.stages.values())
        )
        self.assertEqual(
            task_score.total,
            sum(result.score for result in task_score.stages.values()),
        )
        self.assertEqual(task_score.stages["S1定位"].rule_version, "v5")

    def test_pipeline_has_no_excel_dependency(self) -> None:
        item = evidence()
        task_score = TaskScore(
            task_index=1,
            stages={
                "S1定位": StageResult(
                    stage="S1定位",
                    score=0,
                    max_score=4,
                    confidence=0.9,
                    needs_review=False,
                )
            },
        )
        with (
            patch("pipeline.analyze_data_dir", return_value=[item]),
            patch(
                "pipeline.compute_task_scores",
                return_value={1: task_score},
            ),
        ):
            run = evaluate_data_dir(__import__("pathlib").Path("missing-excel"))

        self.assertIsInstance(run, ScoringRun)
        self.assertEqual(run.tasks[1].stages["S1定位"].score, 0)
        self.assertEqual(run.rule_version, "v5")

        summary = build_summary(run)
        task_summary = summary["tasks"][0]
        self.assertEqual(task_summary["task_index"], 1)
        self.assertEqual(task_summary["total_score"], 0)
        self.assertEqual(task_summary["max_score"], 4)
        self.assertEqual(task_summary["stages"]["S1定位"]["score"], 0)
        self.assertEqual(task_summary["stages"]["S1定位"]["max_score"], 4)
        self.assertIn("task_evidence", task_summary)
        self.assertNotIn("scores", summary)
        self.assertNotIn("evidence", summary)

    def test_pipeline_accepts_replaceable_segment_resolver(self) -> None:
        class FakeResolver:
            def resolve(self, data_dir):
                return TaskSegmentAssignments(
                    state_frame_to_segment={7: 3},
                    video_ranges={"cam_mid": {3: (10, 20)}},
                    source="fake",
                )

        resolver = FakeResolver()
        with patch("pipeline.analyze_data_dir", return_value=[]) as analyze:
            evaluate_data_dir(
                __import__("pathlib").Path("replaceable-segments"),
                segment_resolver=resolver,
            )

        analyze.assert_called_once()
        self.assertIs(analyze.call_args.kwargs["segment_resolver"], resolver)

    def test_v5_rules_own_score_boundaries(self) -> None:
        item = evidence()
        item.has_motion = True
        item.has_grasp_object = True
        item.grasp_time_sec = V5_RULES.positioning_fast_sec
        self.assertEqual(V5_RULES.decide("S1定位", item).score, 4)
        item.grasp_time_sec += 0.001
        self.assertEqual(V5_RULES.decide("S1定位", item).score, 3)

    def test_each_stage_has_independent_confidence_policy(self) -> None:
        item = evidence()
        item.vision.available = False
        s4_without_vision = stage_confidence("S4投放", item, 0.9)
        s5_without_vision = stage_confidence("S5归位", item, 0.9)
        item.vision.available = True
        item.vision.confidence = 0.9
        s4_with_vision = stage_confidence("S4投放", item, 0.9)
        s5_with_vision = stage_confidence("S5归位", item, 0.9)

        self.assertGreater(s4_with_vision, s4_without_vision)
        self.assertEqual(s5_with_vision, s5_without_vision)


if __name__ == "__main__":
    unittest.main()
