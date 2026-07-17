import unittest

import numpy as np

from vision import analyze_frames


class VisionEvidenceTests(unittest.TestCase):
    def frame(self, *, object_xy: tuple[int, int] | None, with_box: bool) -> np.ndarray:
        frame = np.full((180, 320, 3), 150, dtype=np.uint8)
        if with_box:
            frame[55:145, 185:295] = [150, 110, 5]
        if object_xy is not None:
            x, y = object_xy
            frame[y : y + 16, x : x + 18] = [120, 8, 12]
        return frame

    def test_detects_box_without_object(self) -> None:
        frames = np.stack([self.frame(object_xy=None, with_box=True) for _ in range(8)])
        evidence = analyze_frames(frames)
        self.assertEqual(evidence.scenario, "box_without_object")
        self.assertFalse(evidence.object_present)
        self.assertTrue(evidence.box_present)

    def test_tracks_object_into_box(self) -> None:
        frames = []
        for x in np.linspace(40, 220, 12).astype(int):
            frames.append(self.frame(object_xy=(int(x), 90), with_box=True))
        evidence = analyze_frames(np.stack(frames), grasp_fraction=0.2)
        self.assertEqual(evidence.scenario, "normal")
        self.assertTrue(evidence.moved_toward_box)
        self.assertEqual(evidence.final_object_relation, "inside")


if __name__ == "__main__":
    unittest.main()
