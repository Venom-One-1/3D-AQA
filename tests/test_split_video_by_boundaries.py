import csv
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from split_video_by_boundaries import VideoInfo, build_segments, load_boundaries


class SplitVideoByBoundariesTests(unittest.TestCase):
    def test_ranges_cover_every_source_frame_once_and_last_move_reaches_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary_path = Path(directory) / "boundaries.txt"
            with boundary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("URLID", "TagID", "Tag", "End")
                )
                writer.writeheader()
                for move_id in range(1, 25):
                    writer.writerow(
                        {
                            "URLID": "video_5FPS",
                            "TagID": move_id,
                            "Tag": f"move {move_id}",
                            "End": move_id * 2,
                        }
                    )
            boundaries = load_boundaries(boundary_path, 5.0)

        info = VideoInfo(
            fps_numerator=30,
            fps_denominator=1,
            frame_count=1500,
            width=640,
            height=480,
            has_audio=True,
        )
        segments = build_segments("video", boundaries, info, 5.0)
        self.assertEqual(segments[0].start_frame_0based, 0)
        self.assertEqual(segments[0].end_frame_0based, 60)
        self.assertEqual(segments[1].start_frame_0based, 61)
        self.assertEqual(segments[1].end_frame_0based, 120)
        self.assertEqual(segments[-1].start_frame_0based, 1381)
        self.assertEqual(segments[-1].end_frame_0based, 1499)
        self.assertEqual(segments[-1].end_policy, "video_last_frame")
        self.assertEqual(sum(item.frame_count for item in segments), 1500)

    def test_fractional_source_fps_uses_floor_sampling_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            boundary_path = Path(directory) / "boundaries.txt"
            with boundary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("TagID", "Tag", "End"))
                writer.writeheader()
                for move_id in range(1, 25):
                    writer.writerow(
                        {"TagID": move_id, "Tag": str(move_id), "End": move_id}
                    )
            boundaries = load_boundaries(boundary_path, 5.0)

        info = VideoInfo(30000, 1001, 900, 640, 480, False)
        segments = build_segments("video", boundaries, info, 5.0)
        self.assertEqual(segments[0].end_frame_0based, int(Fraction(30000, 1001)))
        self.assertEqual(segments[1].start_frame_0based, segments[0].end_frame_0based + 1)


if __name__ == "__main__":
    unittest.main()
