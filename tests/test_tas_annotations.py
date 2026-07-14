import unittest

from export_tas_reference_annotations import PointLabel, build_segments, compress_point_labels


class TasAnnotationExportTest(unittest.TestCase):
    def test_non_overlapping_5fps_ranges_follow_previous_end_second(self):
        labels = []
        for second in range(0, 16):
            labels.append(_label(tag_id=1, second=second))
        for second in range(16, 42):
            labels.append(_label(tag_id=2, second=second))
        for second in range(42, 49):
            labels.append(_label(tag_id=3, second=second))

        groups = compress_point_labels(labels)
        segments = build_segments(groups, sample_fps=5.0)

        self.assertEqual((segments[0].start_frame, segments[0].end_frame), (1, 75))
        self.assertEqual((segments[1].start_frame, segments[1].end_frame), (76, 205))
        self.assertEqual((segments[2].start_frame, segments[2].end_frame), (206, 240))
        self.assertEqual((segments[1].start_time, segments[1].end_time), (16.0, 41.0))
        self.assertEqual(
            (segments[1].frame_start_boundary_time, segments[1].frame_end_boundary_time),
            (15.0, 41.0),
        )

    def test_background_labels_are_excluded_by_default(self):
        labels = [_label(tag_id=0, second=0), _label(tag_id=1, second=1), _label(tag_id=1, second=2)]

        groups = compress_point_labels(labels)
        segments = build_segments(groups, sample_fps=5.0)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].move_id, 1)


def _label(tag_id: int, second: int) -> PointLabel:
    return PointLabel(
        video_id="QxVvRcRn2TA",
        url="QxVvRcRn2TA.mp4",
        tag_id=tag_id,
        start_second=float(second),
        end_second=float(second),
        state=1,
    )


if __name__ == "__main__":
    unittest.main()
