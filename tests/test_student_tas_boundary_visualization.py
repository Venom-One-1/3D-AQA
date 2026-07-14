import csv
import tempfile
import unittest
from pathlib import Path

from visualize_student_tas_boundary_frames import (
    Layout,
    discover_student_ids,
    load_student_boundaries,
)


class StudentTasBoundaryVisualizationTests(unittest.TestCase):
    def test_layout_contains_all_24_rows_in_one_canvas(self):
        layout = Layout()
        self.assertEqual(
            layout.canvas_height,
            layout.outer_margin * 2 + layout.header_height + 24 * layout.row_height,
        )
        self.assertLessEqual(layout.canvas_width, 900)

    def test_load_student_boundaries_uses_selected_dtw_source_frame(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_dir = root / "00"
            result_dir.mkdir()
            path = result_dir / "boundary_mapping.csv"
            fieldnames = [
                "move_id",
                "move_name",
                "target_source_frame_0based",
                "target_boundary_time_seconds",
                "selected_local_geodesic_degrees",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for move_id in range(1, 25):
                    writer.writerow(
                        {
                            "move_id": move_id,
                            "move_name": f"move_{move_id}",
                            "target_source_frame_0based": move_id * 10,
                            "target_boundary_time_seconds": move_id * 2.0,
                            "selected_local_geodesic_degrees": 5.0,
                        }
                    )

            boundaries = load_student_boundaries(root, "00")

            self.assertEqual(len(boundaries), 24)
            self.assertEqual(boundaries[0].source_frame_index, 10)
            self.assertEqual(boundaries[-1].source_frame_index, 240)

    def test_discovery_requires_video_and_segmentation_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            videos = root / "videos"
            results = root / "results"
            videos.mkdir()
            results.mkdir()
            (videos / "00.mp4").touch()
            (videos / "01.mp4").touch()
            (results / "00").mkdir()
            (results / "00" / "boundary_mapping.csv").touch()
            (results / "02").mkdir()
            (results / "02" / "boundary_mapping.csv").touch()

            self.assertEqual(discover_student_ids(results, videos), ["00"])


if __name__ == "__main__":
    unittest.main()
