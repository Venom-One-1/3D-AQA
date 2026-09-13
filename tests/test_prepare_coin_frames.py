import unittest

from video_utils.prepare_coin_frames import build_frame_rows, merge_unique_url_rows


class PrepareCoinFramesTests(unittest.TestCase):
    def test_builds_five_fps_rows_with_decimal_times(self):
        rows = build_frame_rows("target", 4, 5.0)
        self.assertEqual(
            rows,
            [
                {"URLID": "target", "Frame": "00000.jpg", "Time": "0.0"},
                {"URLID": "target", "Frame": "00001.jpg", "Time": "0.2"},
                {"URLID": "target", "Frame": "00002.jpg", "Time": "0.4"},
                {"URLID": "target", "Frame": "00003.jpg", "Time": "0.6"},
            ],
        )

    def test_merge_preserves_other_rows_and_replaces_target(self):
        existing = [
            {"URLID": "other", "Frame": "00000.jpg", "Time": "0.0"},
            {"URLID": "target", "Frame": "old.jpg", "Time": "0.0"},
        ]
        new = [{"URLID": "target", "Frame": "00000.jpg", "Time": "0.0"}]
        self.assertEqual(
            merge_unique_url_rows(existing, new, "target"),
            [existing[0], new[0]],
        )


if __name__ == "__main__":
    unittest.main()
