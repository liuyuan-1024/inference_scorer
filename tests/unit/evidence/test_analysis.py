import unittest

import numpy as np

from domain.models import TaskEvidence, VisionEvidence
from evidence.events import build_action_event_chain
from evidence.grasp import detect_grasp_event


class DetectGraspEventTests(unittest.TestCase):
    def make_task(
        self, grip: list[float], jc: list[float] | None = None
    ) -> TaskEvidence:
        n = len(grip)
        task = TaskEvidence(task_index=1, n_frames=n)
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


class ActionEventChainTests(unittest.TestCase):
    def test_chest_motion_without_z_lift_is_not_pickup(self) -> None:
        task = TaskEvidence(task_index=1, n_frames=20)
        task.jc_contact_delta = 1200
        task.retract_z_rise_m = 0.005
        task.grasp_transport_m = 0.08
        task.vision = VisionEvidence(
            available=True,
            chest_available=True,
            wrist_available=True,
            object_motion_after_grasp_norm=0.10,
            wrist_object_near_gripper=True,
            wrist_object_retained=True,
        )
        events, has_object = build_action_event_chain(task, has_grasp_attempt=True)
        self.assertTrue(events.contact_detected)
        self.assertFalse(events.object_lifted)
        self.assertFalse(has_object)

    def test_stable_lift_and_retention_is_grasped_object(self) -> None:
        task = TaskEvidence(task_index=1, n_frames=20)
        task.jc_contact_delta = 1200
        task.retract_z_rise_m = 0.06
        task.grasp_transport_m = 0.08
        task.vision = VisionEvidence(
            available=True,
            chest_available=True,
            wrist_available=True,
            object_motion_after_grasp_norm=0.10,
            wrist_object_near_gripper=True,
            wrist_object_retained=True,
        )
        events, has_object = build_action_event_chain(task, has_grasp_attempt=True)
        self.assertTrue(events.object_lifted)
        self.assertTrue(events.object_retained)
        self.assertTrue(events.transported)
        self.assertTrue(has_object)

    def test_gripper_lift_without_visual_acquisition_is_failed_grasp(self) -> None:
        task = TaskEvidence(task_index=1, n_frames=20)
        task.jc_contact_delta = 1200
        task.retract_z_rise_m = 0.06
        task.grasp_transport_m = 0.10
        task.vision = VisionEvidence(
            available=True,
            chest_available=True,
            wrist_available=True,
            wrist_observation_rate=1.0,
            wrist_object_near_gripper=False,
            object_motion_after_grasp_norm=0.10,
        )
        events, has_object = build_action_event_chain(task, has_grasp_attempt=True)
        self.assertTrue(events.contact_detected)
        self.assertFalse(events.object_acquired)
        self.assertFalse(events.object_lifted)
        self.assertFalse(events.object_dropped)
        self.assertFalse(has_object)

    def test_drop_requires_acquisition_then_lift_then_visual_departure(self) -> None:
        task = TaskEvidence(task_index=1, n_frames=20)
        task.jc_contact_delta = 1200
        task.retract_z_rise_m = 0.04
        task.grasp_transport_m = 0.05
        task.vision = VisionEvidence(
            available=True,
            wrist_available=True,
            wrist_observation_rate=1.0,
            wrist_object_near_gripper=True,
            wrist_object_retained=False,
            wrist_drop_detected=True,
        )
        events, has_object = build_action_event_chain(task, has_grasp_attempt=True)
        self.assertTrue(events.object_acquired)
        self.assertTrue(events.object_lifted)
        self.assertTrue(events.object_dropped)
        self.assertFalse(has_object)

    def test_loss_of_retention_alone_is_not_a_drop(self) -> None:
        task = TaskEvidence(task_index=1, n_frames=20)
        task.jc_contact_delta = 1200
        task.retract_z_rise_m = 0.04
        task.vision = VisionEvidence(
            available=True,
            wrist_available=True,
            wrist_observation_rate=1.0,
            wrist_object_near_gripper=True,
            wrist_object_retained=False,
            wrist_drop_detected=False,
        )
        events, _ = build_action_event_chain(task, has_grasp_attempt=True)
        self.assertTrue(events.object_lifted)
        self.assertFalse(events.object_dropped)


if __name__ == "__main__":
    unittest.main()
