import unittest

from models import TaskJudgment
from scoring import detail_S2_grasping, detail_S4_placing, detail_S5_return


def judgment(**overrides) -> TaskJudgment:
    values = {
        "task_index": 1,
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
        "n_frames": 30,
        "reason": "",
    }
    values.update(overrides)
    return TaskJudgment(**values)


class ScoringTests(unittest.TestCase):
    def test_empty_close_is_failed_grasp_not_stable_grasp(self) -> None:
        task = judgment(grip_empty_close=True, vision_available=True, object_present=True)
        self.assertEqual(detail_S2_grasping(task)["score"], 1)

    def test_no_object_is_not_a_grasp(self) -> None:
        task = judgment(object_present=False, vision_available=True)
        self.assertEqual(detail_S2_grasping(task)["score"], 0)

    def test_near_start_without_return_motion_is_not_full_return(self) -> None:
        task = judgment(
            max_home_excursion_m=0.09,
            return_progress_m=0.005,
            home_xy_error_m=0.01,
            home_z_error_m=0.01,
            return_duration_sec=2.0,
            grip_final_val=0.95,
        )
        self.assertEqual(detail_S5_return(task)["score"], 0)

    def test_explicit_return_meeting_all_thresholds_scores_two(self) -> None:
        task = judgment(
            max_home_excursion_m=0.20,
            return_progress_m=0.15,
            home_xy_error_m=0.04,
            home_z_error_m=0.02,
            return_duration_sec=5.0,
            grip_final_val=0.95,
        )
        self.assertEqual(detail_S5_return(task)["score"], 2)

    def test_inside_without_settling_is_not_accurate_placement(self) -> None:
        task = judgment(
            has_grasp_object=True,
            grip_release_detected=True,
            final_object_relation="inside",
            place_at_box=False,
            vision_available=True,
        )
        self.assertEqual(detail_S4_placing(task)["score"], 1)


if __name__ == "__main__":
    unittest.main()
