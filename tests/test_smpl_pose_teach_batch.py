import json
import tempfile
import unittest
from pathlib import Path

from run_smpl_pose_dtw_teach_batch import (
    TargetFiles,
    discover_targets,
    result_is_current,
)
from visualize_smpl_pose_dtw_teach_boundaries import Layout


class SmplPoseTeachBatchTests(unittest.TestCase):
    def test_layout_width_tracks_two_or_three_columns(self):
        two = Layout(column_count=2)
        three = Layout(column_count=3)
        self.assertEqual(three.canvas_width - two.canvas_width, two.cell_width)
        self.assertEqual(
            two.canvas_height,
            two.outer_margin * 2 + two.header_height + 24 * two.row_height,
        )

    def test_discovery_excludes_reference_and_reports_missing_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            tracking = root / "tracking"
            videos.mkdir()
            for video_id in ("reference", "complete", "missing_video"):
                result_dir = tracking / video_id / "results"
                result_dir.mkdir(parents=True)
                (result_dir / f"demo_{video_id}.pkl").touch()
            (videos / "complete.mp4").touch()

            targets, problems = discover_targets(videos, tracking, "reference")

        self.assertEqual([item.video_id for item in targets], ["complete"])
        self.assertEqual(len(problems), 1)
        self.assertIn("missing_video", problems[0])

    def test_current_result_requires_matching_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "target.mp4"
            tracking = root / "target.pkl"
            reference_video = root / "reference.mp4"
            reference_tracking = root / "reference.pkl"
            gold = root / "gold.txt"
            summary = root / "summary.json"
            target = TargetFiles("target", video, tracking)
            payload = {
                "status": "ok",
                "video_id": "target",
                "input_video": str(video.absolute()),
                "input_tracking": str(tracking.absolute()),
                "reference_video": str(reference_video.absolute()),
                "reference_tracking": str(reference_tracking.absolute()),
                "gold_boundaries": str(gold.absolute()),
                "sample_fps": 5.0,
            }
            summary.write_text(json.dumps(payload), encoding="utf-8")

            self.assertTrue(
                result_is_current(
                    summary, target, reference_video, reference_tracking, gold, 0.0
                )
            )
            payload["reference_video"] = "different.mp4"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(
                result_is_current(
                    summary, target, reference_video, reference_tracking, gold, 0.0
                )
            )


if __name__ == "__main__":
    unittest.main()
