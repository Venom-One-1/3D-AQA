import unittest

from export_tas_ground_truth_comparison import compare_segments
from export_tas_ground_truth_comparison import GroundTruthSegment


class TasGroundTruthComparisonTests(unittest.TestCase):
    def test_comparison_reports_signed_errors_and_temporal_iou(self):
        ground_truth = [
            GroundTruthSegment(
                video_id="video",
                move_id=1,
                move_name="qishi",
                source_annotated_first_active_time=10.0,
                source_last_matching_label_time=20.0,
                source_boundary_end_time=20.0,
                trim_start_time=10.0,
                annotated_first_active_time=0.0,
                ground_truth_start_time=0.0,
                ground_truth_end_time=10.0,
                ground_truth_duration=10.0,
                start_frame_5fps=1,
                end_frame_5fps=50,
                sample_fps=5.0,
                annotation_anomaly_count=0,
                time_policy="test",
            )
        ]
        comparison = compare_segments(ground_truth, "dtw_prediction", {1: (1.0, 12.0)})[0]
        self.assertAlmostEqual(comparison.start_error_seconds, 1.0)
        self.assertAlmostEqual(comparison.end_error_seconds, 2.0)
        self.assertAlmostEqual(comparison.duration_error_seconds, 1.0)
        self.assertAlmostEqual(comparison.temporal_iou, 9.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
