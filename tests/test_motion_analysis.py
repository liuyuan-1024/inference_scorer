import unittest

import numpy as np

from models import TaskSignals
from motion_analysis import analyze_motion_phases


class TaskReturnOriginTests(unittest.TestCase):
    @staticmethod
    def analyze(trajectory: np.ndarray) -> TaskSignals:
        task = TaskSignals(task_index=1, n_frames=len(trajectory))
        task.ee_traj = trajectory
        # 即使对象中残留了别的原点，分析时也必须使用当前 task 第一帧。
        task.task_start_ee = [0.0, 0.0, 0.0]
        analyze_motion_phases(task, place_center=np.zeros(3))
        return task

    def test_return_is_relative_to_this_task_first_frame(self) -> None:
        task = self.analyze(
            np.array(
                [
                    [1.00, 2.00, 0.50],
                    [1.20, 2.00, 0.50],
                    [1.03, 2.04, 0.52],
                ]
            )
        )
        self.assertEqual(task.task_start_ee, [1.0, 2.0, 0.5])
        self.assertAlmostEqual(task.max_home_excursion_m, 0.20)
        self.assertAlmostEqual(task.home_xy_error_m, 0.05)
        self.assertAlmostEqual(task.home_z_error_m, 0.02)
        expected_final_distance = np.linalg.norm([0.03, 0.04, 0.02])
        self.assertAlmostEqual(
            task.return_progress_m, 0.20 - expected_final_distance
        )

    def test_absolute_robot_position_does_not_change_return_metrics(self) -> None:
        relative = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.20, 0.00, 0.00],
                [0.03, 0.04, 0.02],
            ]
        )
        first = self.analyze(relative)
        second = self.analyze(relative + np.array([5.0, -3.0, 1.2]))
        self.assertAlmostEqual(
            first.max_home_excursion_m, second.max_home_excursion_m
        )
        self.assertAlmostEqual(
            first.home_xy_error_m, second.home_xy_error_m
        )
        self.assertAlmostEqual(
            first.home_z_error_m, second.home_z_error_m
        )
        self.assertAlmostEqual(
            first.return_progress_m, second.return_progress_m
        )


if __name__ == "__main__":
    unittest.main()
