import unittest

import numpy as np

from analysis import detect_grasp_event
from models import TaskSignals


class DetectGraspEventTests(unittest.TestCase):
    def make_task(self, grip: list[float], jc: list[float] | None = None) -> TaskSignals:
        n = len(grip)
        task = TaskSignals(task_index=1, n_frames=n)
        task.ee_traj = np.zeros((n, 3), dtype=float)
        task.grip_traj = np.array(grip, dtype=float)
        task.jc_traj = np.array(jc or [100.0] * n, dtype=float)
        return task

    def test_detects_gradual_close(self) -> None:
        task = self.make_task([0.9, 0.85, 0.75, 0.65, 0.55, 0.48, 0.35])
        self.assertEqual(detect_grasp_event(task), (5, "grip"))

    def test_reliable_open_grip_does_not_fallback_to_current(self) -> None:
        task = self.make_task(
            [0.95] * 20,
            [100.0] * 8 + [2000.0] * 5 + [100.0] * 7,
        )
        self.assertEqual(detect_grasp_event(task), (None, None))

    def test_current_fallback_when_grip_is_unavailable(self) -> None:
        task = self.make_task(
            [float("nan")] * 20,
            [100.0] * 8 + [2000.0] * 5 + [100.0] * 7,
        )
        frame, method = detect_grasp_event(task)
        self.assertEqual(method, "jc")
        self.assertIsNotNone(frame)


if __name__ == "__main__":
    unittest.main()
