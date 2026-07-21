import unittest

from domain.models import ActionEventChain, TaskEvidence
from stages.grasping import score_grasping
from stages.placing import score_placing
from stages.returning import score_return


def evidence(**overrides) -> TaskEvidence:
    task = TaskEvidence(task_index=1, n_frames=30, sensor_confidence=0.9)
    for field in (
        "grip_fully_closed",
        "max_home_excursion_m",
        "return_progress_m",
        "home_xy_error_m",
        "home_z_error_m",
        "return_duration_sec",
        "grip_final_val",
        "grip_release_detected",
    ):
        if field in overrides:
            setattr(task, field, overrides.pop(field))
    if "vision_available" in overrides:
        task.vision.available = overrides.pop("vision_available")
    if "object_present" in overrides:
        task.vision.object_present = overrides.pop("object_present")
    if "final_object_relation" in overrides:
        task.vision.final_object_relation = overrides.pop("final_object_relation")

    events = ActionEventChain(
        object_lifted=overrides.pop("object_lifted", False),
        object_dropped=overrides.pop("object_dropped", False),
    )
    values = {
        "has_motion": True,
        "has_grasp_attempt": True,
        "has_grasp_contact": False,
        "has_grasp_object": False,
        "has_place_phase": False,
        "place_at_box": False,
        "action_complete": False,
        "approach_quality": "none",
        "retract_semantic": "none",
        "has_retract": False,
        "has_withdraw_home": False,
        "reason": "",
        "events": events,
    }
    for key, value in values.items():
        setattr(task, key, value)
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


class ScoringTests(unittest.TestCase):
    def test_empty_close_is_failed_grasp_not_stable_grasp(self) -> None:
        task = evidence(
            grip_fully_closed=True,
            vision_available=True,
            object_present=True,
        )
        self.assertEqual(score_grasping(task).score, 1)

    def test_fully_closed_gripper_does_not_override_visual_acquisition(self) -> None:
        task = evidence(
            grip_fully_closed=True,
            has_grasp_object=True,
            vision_available=True,
            object_present=True,
        )
        task.events.object_acquired = True
        self.assertEqual(score_grasping(task).score, 3)

    def test_no_object_is_not_a_grasp(self) -> None:
        task = evidence(object_present=False, vision_available=True)
        self.assertEqual(score_grasping(task).score, 0)

    def test_lift_without_confirmed_drop_remains_failed_grasp(self) -> None:
        task = evidence(
            object_lifted=True,
            object_dropped=False,
            vision_available=True,
        )
        self.assertEqual(score_grasping(task).score, 1)

    def test_confirmed_lift_then_drop_scores_two(self) -> None:
        task = evidence(
            object_lifted=True,
            object_dropped=True,
            vision_available=True,
        )
        self.assertEqual(score_grasping(task).score, 2)

    def test_near_start_without_return_motion_is_not_full_return(self) -> None:
        task = evidence(
            max_home_excursion_m=0.09,
            return_progress_m=0.005,
            home_xy_error_m=0.01,
            home_z_error_m=0.01,
            return_duration_sec=2.0,
            grip_final_val=0.95,
        )
        self.assertEqual(score_return(task).score, 0)

    def test_explicit_return_meeting_all_thresholds_scores_two(self) -> None:
        task = evidence(
            max_home_excursion_m=0.20,
            return_progress_m=0.15,
            home_xy_error_m=0.04,
            home_z_error_m=0.02,
            return_duration_sec=5.0,
            grip_final_val=0.95,
        )
        self.assertEqual(score_return(task).score, 2)

    def test_inside_without_settling_is_not_accurate_placement(self) -> None:
        task = evidence(
            has_grasp_object=True,
            grip_release_detected=True,
            final_object_relation="inside",
            place_at_box=False,
            vision_available=True,
        )
        self.assertEqual(score_placing(task).score, 1)


if __name__ == "__main__":
    unittest.main()
