import unittest

import numpy as np

from video_io import DecodedFrames
from vision import analyze_chest_frames, analyze_wrist_frames


class VisionEvidenceTests(unittest.TestCase):
    def frame(self, *, object_xy: tuple[int, int] | None, with_box: bool) -> np.ndarray:
        frame = np.full((180, 320, 3), 150, dtype=np.uint8)
        if with_box:
            frame[55:145, 185:295] = [150, 110, 5]
        if object_xy is not None:
            x, y = object_xy
            frame[y : y + 16, x : x + 18] = [120, 8, 12]
        return frame

    @staticmethod
    def chest_evidence(
        frames: np.ndarray, *, grasp_fraction: float | None = None
    ):
        return analyze_chest_frames(
            DecodedFrames(
                frames=frames,
                timestamps=np.full(len(frames), np.nan),
                frame_indices=np.arange(len(frames)),
            ),
            grasp_fraction=grasp_fraction,
        )

    def test_detects_box_without_object(self) -> None:
        frames = np.stack([self.frame(object_xy=None, with_box=True) for _ in range(8)])
        evidence = self.chest_evidence(frames)
        self.assertEqual(evidence.scenario, "box_without_object")
        self.assertFalse(evidence.object_present)
        self.assertTrue(evidence.box_present)

    def test_tracks_object_into_box(self) -> None:
        frames = []
        for x in np.linspace(40, 220, 12).astype(int):
            frames.append(self.frame(object_xy=(int(x), 90), with_box=True))
        evidence = self.chest_evidence(
            np.stack(frames), grasp_fraction=0.2
        )
        self.assertEqual(evidence.scenario, "normal")
        self.assertTrue(evidence.moved_toward_box)
        self.assertEqual(evidence.final_object_relation, "inside")

    def test_wrist_detects_object_retained_between_jaws(self) -> None:
        frames = np.stack(
            [self.frame(object_xy=(151, 138), with_box=False) for _ in range(8)]
        )
        evidence = analyze_wrist_frames(
            DecodedFrames(
                frames=frames,
                timestamps=np.arange(8, dtype=float),
                frame_indices=np.arange(8),
            ),
            grasp_timestamp=1.0,
            release_timestamp=6.0,
        )
        self.assertTrue(evidence.wrist_object_near_gripper)
        self.assertTrue(evidence.wrist_object_retained)
        self.assertFalse(evidence.wrist_drop_detected)

    def test_wrist_detects_object_leaving_before_release(self) -> None:
        frames = [
            self.frame(object_xy=(151, 138), with_box=False) for _ in range(4)
        ]
        frames.extend(
            self.frame(object_xy=(10, 20), with_box=False) for _ in range(4)
        )
        evidence = analyze_wrist_frames(
            DecodedFrames(
                frames=np.stack(frames),
                timestamps=np.arange(8, dtype=float),
                frame_indices=np.arange(8),
            ),
            grasp_timestamp=1.0,
            release_timestamp=7.0,
        )
        self.assertTrue(evidence.wrist_drop_detected)


if __name__ == "__main__":
    unittest.main()
