import csv
import tempfile
import unittest
from pathlib import Path

from visualize_smpl_pose_dtw_boundaries import (
    Layout,
    load_ground_truth_boundaries,
    load_predicted_boundaries,
)


class SmplPoseBoundaryVisualizationTests(unittest.TestCase):
    def test_layout_contains_all_rows_in_one_image(self):
        layout = Layout()
        self.assertEqual(
            layout.canvas_height,
            layout.outer_margin * 2 + layout.header_height + 24 * layout.row_height,
        )
        self.assertLessEqual(layout.canvas_width, 900)

    def test_loads_ground_truth_boundary_time_and_predicted_source_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth_path = root / "ground_truth.csv"
            predicted_path = root / "boundaries.csv"
            with ground_truth_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "video_id",
                        "move_id",
                        "move_name",
                        "end_time",
                        "frame_end_boundary_time",
                    ),
                )
                writer.writeheader()
                for move_id in range(1, 25):
                    writer.writerow(
                        {
                            "video_id": "video",
                            "move_id": move_id,
                            "move_name": f"move_{move_id}",
                            "end_time": move_id + 0.8,
                            "frame_end_boundary_time": move_id,
                        }
                    )
            with predicted_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "move_id",
                        "move_name",
                        "target_end_time_seconds",
                        "target_source_frame_0based",
                        "local_geodesic_degrees",
                    ),
                )
                writer.writeheader()
                for move_id in range(1, 25):
                    writer.writerow(
                        {
                            "move_id": move_id,
                            "move_name": f"move_{move_id}",
                            "target_end_time_seconds": move_id + 0.2,
                            "target_source_frame_0based": move_id * 30,
                            "local_geodesic_degrees": 5.0,
                        }
                    )

            ground_truth = load_ground_truth_boundaries(ground_truth_path, "video")
            predictions = load_predicted_boundaries(predicted_path, "video")

        self.assertEqual(len(ground_truth), 24)
        self.assertEqual(ground_truth[0].end_time, 1.0)
        self.assertEqual(predictions[0].end_time, 1.2)
        self.assertEqual(predictions[-1].source_frame_index_0based, 720)


if __name__ == "__main__":
    unittest.main()
