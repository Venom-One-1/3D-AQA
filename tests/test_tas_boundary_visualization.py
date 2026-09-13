import unittest

from visualize_tas_boundary_frames import (
    VideoMetadata,
    boundary_time_to_source_frame,
    load_font,
)


class TasBoundaryVisualizationTests(unittest.TestCase):
    def test_default_font_contains_chinese_glyphs(self):
        font = load_font(18)
        chinese_glyph = bytes(font.getmask("起"))
        missing_glyph = bytes(font.getmask("\u0378"))
        self.assertNotEqual(chinese_glyph, missing_glyph)

    def test_boundary_time_maps_to_nearest_source_frame(self):
        metadata = VideoMetadata(fps=30.0, frame_count=900)
        self.assertEqual(boundary_time_to_source_frame(10.2, metadata), 306)

    def test_final_boundary_is_clamped_to_last_frame(self):
        metadata = VideoMetadata(fps=30.0, frame_count=330)
        self.assertEqual(boundary_time_to_source_frame(11.0, metadata), 329)


if __name__ == "__main__":
    unittest.main()
