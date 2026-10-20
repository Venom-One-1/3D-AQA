"""Focused source and export checks for the Fig. 2 artifact."""

import hashlib
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(sys.argv.pop(1))


class FigureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((ROOT / 'Fig2_Method_Overview.audit.json').read_text())
        cls.svg = ET.parse(ROOT / 'Fig2_Method_Overview.svg').getroot()

    def test_canvas_and_visible_regions(self):
        with Image.open(ROOT / 'Fig2_Method_Overview.png') as image:
            self.assertEqual(image.size, (3200, 1450))
            pixels = np.asarray(image.convert('RGB'))
        for x0, x1 in [(35, 540), (620, 1740), (1810, 2410), (2520, 3155)]:
            panel = pixels[175:1330, x0:x1]
            self.assertGreater(float(panel.std()), 15)
            self.assertGreater(float(np.mean(panel.min(axis=2) < 150)), .01)

    def test_layout_validation(self):
        self.assertEqual(self.audit['text_overlap_check'], 'passed')
        self.assertEqual(self.audit['text_width_check'], 'passed')

    def test_editable_and_concise_text(self):
        labels = [''.join(node.itertext()) for node in self.svg.findall('.//{http://www.w3.org/2000/svg}text')]
        self.assertEqual(len(labels), self.audit['text_items'] + 1)  # One two-line label.
        all_text = ' '.join(labels)
        self.assertLess(len(all_text.split()), 160)
        for label in ['Student Videos', 'Reference Demonstrations', 'Form segmentation',
                      'For each student', 'LLM', 'Relative Ranking', 'Illustrative feedback']:
            self.assertIn(label, all_text)

    def test_embedded_images(self):
        images = self.svg.findall('.//{http://www.w3.org/2000/svg}image')
        self.assertEqual(len(images), 12)
        for node in images:
            self.assertTrue(node.attrib['{http://www.w3.org/1999/xlink}href'].startswith('data:image/png;base64,'))

    def test_frozen_pose_hashes(self):
        self.assertEqual(len(self.audit['pose_assets']), 8)
        for source in self.audit['pose_assets']:
            self.assertEqual(hashlib.sha256(Path(source['path']).read_bytes()).hexdigest(), source['sha256'])

    def test_actual_measurements_and_honest_scope(self):
        knee, wrist, stance = self.audit['plotted_metrics']
        self.assertLessEqual(knee['teacher_range'][0], knee['student_value'])
        self.assertLessEqual(knee['student_value'], knee['teacher_range'][1])
        self.assertLessEqual(wrist['teacher_range'][0], wrist['student_value'])
        self.assertLessEqual(wrist['student_value'], wrist['teacher_range'][1])
        self.assertLess(stance['student_value'], stance['teacher_range'][0])
        self.assertEqual(self.audit['teacher_count'], 10)
        self.assertFalse(self.audit['reference_anchor_in_teacher_statistics'])
        self.assertTrue(self.audit['rank_visual_is_schematic'])
        self.assertFalse(self.audit['cohort_rank_computed'])
        self.assertEqual(self.audit['model_calls'], 0)

    def test_pdf_present(self):
        self.assertTrue((ROOT / 'Fig2_Method_Overview.pdf').read_bytes().startswith(b'%PDF'))


if __name__ == '__main__':
    unittest.main()
