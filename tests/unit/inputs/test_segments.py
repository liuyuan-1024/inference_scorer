import json
import tempfile
import unittest
from pathlib import Path

from inputs.segments import RecordedTaskSegmentResolver, TaskSegmentAssignments


class RecordedTaskSegmentResolverTests(unittest.TestCase):
    def test_resolves_state_and_both_camera_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            machine_flow = [
                {
                    "event": "chunk_inference",
                    "frame_index": 10,
                    "task_segment_index": 1,
                },
                {
                    "event": "chunk_inference",
                    "frame_index": 11,
                    "task_segment_index": 2,
                },
            ]
            boundaries = [
                {
                    "type": "task_boundary",
                    "event": "start",
                    "task_segment_index": 1,
                    "video_frame_counts": {"cam_mid": 100, "cam_left": 200},
                },
                {
                    "type": "task_boundary",
                    "event": "end",
                    "task_segment_index": 1,
                    "video_frame_counts": {"cam_mid": 120, "cam_left": 225},
                },
                {
                    "type": "task_boundary",
                    "event": "start",
                    "task_segment_index": 2,
                    "video_frame_counts": {"cam_mid": 120, "cam_left": 225},
                },
                {
                    "type": "task_boundary",
                    "event": "end",
                    "task_segment_index": 2,
                    "video_frame_counts": {"cam_mid": 140, "cam_left": 250},
                },
            ]
            self._write_jsonl(data_dir / "machine_flow.jsonl", machine_flow)
            self._write_jsonl(data_dir / "task_segments.jsonl", boundaries)

            result = RecordedTaskSegmentResolver().resolve(data_dir)

        self.assertIsInstance(result, TaskSegmentAssignments)
        self.assertEqual(result.state_frame_to_segment, {10: 1, 11: 2})
        self.assertEqual(result.video_ranges["cam_mid"], {1: (100, 119), 2: (120, 139)})
        self.assertEqual(result.video_ranges["cam_left"], {1: (200, 224), 2: (225, 249)})
        self.assertEqual(result.segment_indices, (1, 2))
        self.assertEqual(result.source, "recorded_metadata")

    @staticmethod
    def _write_jsonl(path: Path, items: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in items),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
