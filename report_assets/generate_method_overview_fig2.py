#!/usr/bin/env python
"""Render the agreed high-level Fig. 2 with frozen, genuine SMPL assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT = Path('/home/sqw/Projects/3D-AQA')
sys.path.insert(0, str(PROJECT / 'report_assets'))
from generate_coaching_feedback_figure import Diagram


ASSETS = PROJECT / 'report_assets/first_bow_stance_figures_v2'
OUTPUT = PROJECT / 'report_assets/method_overview_fig2_v1'
BLUE_LIGHT, GREEN_LIGHT = '#E4EFFF', '#E5F5DF'
GRAY, PALE = '#667486', '#EFF4FA'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class Overview(Diagram):
    width, height = 3200, 1450

    def picture_fit(self, path, x, y, w, h):
        with Image.open(path) as source:
            pixels = np.asarray(source)
        scale = min(w / pixels.shape[1], h / pixels.shape[0])
        pw, ph = pixels.shape[1] * scale, pixels.shape[0] * scale
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
        self.ax.imshow(pixels, extent=(px, px + pw, py + ph, py),
                       aspect='auto', zorder=2)
        return px, py, pw, ph

    def circle(self, x, y, radius, face, edge='white', lw=1.2, z=5):
        from matplotlib.patches import Circle
        self.ax.add_patch(Circle((x, y), radius, facecolor=face,
                                 edgecolor=edge, linewidth=lw, zorder=z))

    def person(self, x, y, color, scale=1):
        from matplotlib.patches import Arc
        self.circle(x, y, 10 * scale, color, color)
        self.ax.add_patch(Arc((x, y + 30 * scale), 34 * scale, 30 * scale,
                             theta1=180, theta2=360, color=color,
                             linewidth=2.7, zorder=4))


def extract_frame(video, frame0, destination):
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open {video}')
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame0)
        ok, pixels = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot read frame {frame0}: {video}')
        position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if position != frame0:
            raise RuntimeError(f'Unexpected frame after seek: {position} != {frame0}')
        Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(destination)
        return {'video': str(video), 'frame0': frame0, 'image': record(destination)}
    finally:
        capture.release()


def load_sources(output):
    data = read_json(ASSETS / 'figure_data.json')
    assert data['pose_id'] == '2.part_mane_1_L'
    assert data['first_event_feedback_available'] is False
    poses, checks = {}, []
    for subject in data['subjects']:
        for item in subject['images']:
            path = ASSETS / item['image']
            actual = record(path)
            assert actual['sha256'] == item['sha256'], path
            with Image.open(path) as source:
                assert source.mode == 'RGBA' and source.size == (768, 1024)
            poses[(subject['role'], item['frame0'])] = path
            checks.append(actual)
    manifest = read_json(PROJECT / 'reference_data/BV1WE411W7JB/reference_manifest.json')
    student = read_json(PROJECT / 'temporal_event_metric_results/first3_five_students_v1/student/10/measurements.json')
    videos = [
        extract_frame(manifest['reference']['video_path'], 686, output / 'reference_frame686.png'),
        extract_frame(student['subject']['input_video']['path'], 623, output / 'student10_frame623.png'),
    ]
    with (ASSETS / 'first_event_metrics.csv').open(encoding='utf-8-sig', newline='') as handle:
        metrics = {row['metric_id']: row for row in csv.DictReader(handle)}
    assert all(row['teacher_count'] == '10' for row in metrics.values())
    return data, poses, metrics, checks, videos


def document(d, x, y):
    d.box(x, y, 74, 94, edge=d.blue, radius=8, lw=2.0)
    for dy, w in ((23, 45), (42, 45), (61, 45), (79, 28)):
        d.line([(x + 14, y + dy), (x + 14 + w, y + dy)], d.blue, 1.8)


def timeline(d, x, y, w, color, window):
    lo, center, hi = [x + f * w for f in window]
    d.box(lo, y - 15, hi - lo, 30, fill='#F9ECC0' if color == d.gold else '#FFE5CF',
          edge='none', radius=0)
    d.line([(x, y), (x + w, y)], color, 2.4)
    for position in (x, x + w):
        d.line([(position, y - 11), (position, y + 11)], color, 2)
    for position in (lo, hi):
        d.line([(position, y - 20), (position, y + 20)], color, 1.0)
    d.circle(center, y, 7.5, color, d.ink)
    return lo, center, hi


def input_group(d, y, title, video, pose, color, caption):
    d.box(35, y, 505, 365, edge=color, radius=32, lw=1.7)
    d.text(287, y + 20, title, 33, True, align='center', max_width=470)
    # Back sheets indicate a collection; only the foreground video is a real example.
    for dx, dy in ((20, -20), (10, -10)):
        d.box(63 + dx, y + 116 + dy, 245, 145, PALE, color, 8, 0.7)
    d.picture_fit(video, 63, y + 116, 245, 145)
    d.arrow([(317, y + 190), (358, y + 190)], color, 2.0)
    d.picture_fit(pose, 359, y + 87, 157, 210)
    d.text(437, y + 65, '3D Pose', 25, color=GRAY, align='center')
    d.text(287, y + 315, caption, 24, color=GRAY, align='center', max_width=465)


def draw_inputs(d, output, poses):
    d.box(35, 175, 505, 235, edge='#151515', radius=32)
    d.text(287, 194, 'Technique Instructions', 34, True, align='center', max_width=475)
    document(d, 75, 269)
    d.text(179, 274, 'Left bow stance;\npart the hands.', 32, max_width=325)
    input_group(d, 510, 'Reference Demonstrations', output / 'reference_frame686.png',
                poses['reference', 686], d.gold, 'Reference anchor shown')
    input_group(d, 965, 'Student Videos', output / 'student10_frame623.png',
                poses['student', 623], d.orange, 'Student 10 shown')


def draw_inspection(d, poses):
    d.box(620, 175, 1120, 1155, BLUE_LIGHT, '#7599CC', 44, 1.3)
    d.box(760, 175, 840, 66, '#3974D7', '#3974D7', 16)
    d.text(1180, 184, 'Technique-Guided Inspection', 39, True, 'white', 'center', 800)
    d.box(650, 280, 1060, 180, 'white', '#151515', 25)
    d.text(1180, 296, 'Offline specification', 32, True, align='center')
    d.text(1180, 338, 'LLM-assisted, reviewed', 25, color=GRAY, align='center')
    d.text(901, 393, 'When to inspect', 33, True, d.blue, 'center', 450)
    d.text(1458, 393, 'What to measure', 33, True, d.blue, 'center', 450)
    d.arrow([(540, 302), (650, 302)], d.blue)
    d.arrow([(901, 460), (901, 535)], d.blue)
    d.arrow([(1458, 460), (1458, 535)], d.blue)
    d.box(650, 535, 500, 750, 'white', '#151515', 28)
    d.box(1210, 535, 500, 750, 'white', '#151515', 28)
    d.text(900, 556, 'Event Localization', 35, True, align='center', max_width=460)
    d.text(1460, 556, 'Checkpoint Measurement', 35, True, align='center', max_width=465)
    d.text(900, 610, 'Form segmentation', 26, color=GRAY, align='center')
    for x, width, color in ((690, 88, '#E2E8F0'), (784, 226, '#B7D1F8'),
                            (1016, 94, '#E2E8F0')):
        d.box(x, 651, width, 11, color, 'none', 0)
    d.text(685, 668, 'Reference', 25, color=d.gold)
    d.text(685, 1002, 'Student', 25, color=d.orange)
    for i, frame in enumerate((510, 560, 686)):
        d.picture_fit(poses['reference', frame], 745 + i * 125, 685, 120, 160)
    reference_points = timeline(d, 690, 873, 420, d.gold, (.30, .47, .64))
    student_points = timeline(d, 690, 965, 420, d.orange, (.43, .66, .88))
    for ref, student in zip(reference_points, student_points):
        d.line([(ref, 894), (student, 941)], d.ink, 1.25, True)
        d.arrow([(student - 5, 934), (student, 945)], d.ink, 1.0)
    for i, frame in enumerate((455, 503, 623)):
        d.picture_fit(poses['student', frame], 745 + i * 125, 1025, 120, 160)
    d.text(900, 1228, 'Matched semantic events', 27, color=GRAY, align='center', max_width=460)
    d.arrow([(540, 704), (650, 704)], d.gold)
    d.arrow([(540, 1156), (650, 1156)], d.orange)
    d.arrow([(1150, 913), (1210, 913)], d.blue)
    for x, role, frame, label, color in (
        (1224, 'reference', 686, 'Reference', d.gold),
        (1461, 'student', 623, 'Student', d.orange),
    ):
        d.text(x + 112, 643, label, 27, True, color, 'center')
        d.picture_fit(poses[role, frame], x, 706, 235, 313.3333333333)
    # Symbolic callouts mark anatomical regions, not measured pixel coordinates.
    for x, knee, wrist in ((1224, (.60, .69), (.82, .27)),
                           (1461, (.565, .684), (.837, .225))):
        d.circle(x + 235 * knee[0], 706 + 313.333 * knee[1], 15, 'none', d.blue, 1.6)
        d.circle(x + 235 * wrist[0], 706 + 313.333 * wrist[1], 11, 'none', d.blue, 1.4)
    d.line([(1248, 1056), (1670, 1056)], '#D6E0ED', 1)
    for label, y in [('Knee angle', 1100), ('Wrist height', 1160), ('Stance length', 1220)]:
        d.circle(1261, y + 15, 5, d.blue, d.blue)
        d.text(1290, y, label, 29)
    d.text(1180, 1299, 'For each student', 27, True, d.blue, 'center')


def draw_evidence(d, metrics):
    d.box(1810, 175, 600, 1155, BLUE_LIGHT, '#7599CC', 44, 1.3)
    d.box(1850, 175, 520, 66, '#3974D7', '#3974D7', 16)
    d.text(2110, 184, 'Geometric Evidence', 39, True, 'white', 'center', 485)
    d.text(2110, 291, 'Teacher reference distributions', 30, True, align='center', max_width=560)
    for x in (1960, 2025, 2090, 2155, 2220):
        d.person(x, 365, d.gold, .9)
    d.text(2110, 432, 'Technique + measurements', 29, color=GRAY, align='center')
    d.box(1844, 506, 532, 639, 'white', '#151515', 28)
    selected = [('left_knee_angle', 'Knee angle'),
                ('left_wrist_head_up_torso_ratio', 'Wrist height'),
                ('stance_length_leg_ratio', 'Stance length')]
    evidence = []
    for i, (metric_id, label) in enumerate(selected):
        row = metrics[metric_id]
        lower = float(row['teacher_median_2mad_lower'])
        upper = float(row['teacher_median_2mad_upper'])
        value = float(row['student_window_median'])
        start, end = min(lower, value), max(upper, value)
        padding = (end - start) * .25
        minimum, maximum = start - padding, end + padding
        locate = lambda v: 1894 + (v - minimum) / (maximum - minimum) * 427
        y = 558 + i * 158
        d.text(1880, y, label, 31, True)
        d.line([(1894, y + 79), (2321, y + 79)], '#CBD4E0', 1.7)
        d.box(locate(lower), y + 62, locate(upper) - locate(lower), 34,
              '#F7E9B5', 'none', 3)
        d.line([(locate(lower), y + 79), (locate(upper), y + 79)], d.gold, 2.6)
        for number in (lower, upper):
            d.line([(locate(number), y + 57), (locate(number), y + 101)], d.gold, 1.3)
        d.circle(locate(value), y + 79, 10, d.orange, 'white', 1.5)
        evidence.append({'metric_id': metric_id, 'teacher_range': [lower, upper],
                         'student_value': value, 'axis_range': [minimum, maximum]})
    d.line([(1880, 1051), (1990, 1051)], d.gold, 4)
    d.text(2012, 1035, 'Teacher range', 27, color=GRAY)
    d.circle(1904, 1102, 9, d.orange, 'white')
    d.text(1930, 1086, 'Student value', 27, color=GRAY)
    for dx, dy in ((16, -16), (8, -8), (0, 0)):
        d.box(1880 + dx, 1210 + dy, 60, 64, 'white', '#9EACBD', 5, .8)
    d.text(1971, 1221, 'Per-student evidence', 29, True, max_width=410)
    d.arrow([(1740, 913), (1810, 913)], d.blue)
    return evidence


def speech(d, x, y, w, h):
    from matplotlib.patches import Polygon
    d.box(x, y, w, h, 'white', '#151515', 24, 1.2)
    d.ax.add_patch(Polygon([(x + 48, y + h - 2), (x + 48, y + h + 27),
                           (x + 87, y + h - 2)], closed=True,
                          facecolor='white', edgecolor='#151515', linewidth=1.2, zorder=2))
    d.line([(x + 50, y + h - 1), (x + 83, y + h - 1)], 'white', 2.5)


def draw_outputs(d):
    d.box(2520, 175, 635, 645, BLUE_LIGHT, '#7599CC', 44, 1.3)
    d.box(2563, 175, 550, 66, '#3974D7', '#3974D7', 16)
    d.text(2837, 184, 'Coaching Feedback', 39, True, 'white', 'center', 520)
    d.box(2692, 297, 290, 80, 'white', '#151515', 18)
    d.text(2837, 318, 'LLM', 36, True, d.blue, 'center')
    d.arrow([(2837, 377), (2837, 437)], d.blue)
    speech(d, 2563, 445, 550, 234)
    d.text(2603, 480, 'Lengthen your stance.', 34, max_width=470)
    d.text(2603, 554, 'Let the right arm settle.', 34, max_width=470)
    d.text(2837, 725, 'For each student', 29, True, d.blue, 'center')
    d.text(2837, 769, 'Illustrative feedback', 24, color=GRAY, align='center')
    d.box(2520, 900, 635, 430, GREEN_LIGHT, '#69A653', 44, 1.3)
    d.box(2563, 900, 550, 66, '#15931C', '#15931C', 16)
    d.text(2837, 910, 'Relative Ranking', 39, True, 'white', 'center', 520)
    d.text(2837, 1005, 'Aggregate & rank students', 31, True, align='center', max_width=575)
    for i, (length, color) in enumerate(((350, '#338F44'), (270, '#68B477'), (180, '#A6D1A5'))):
        y = 1094 + i * 68
        d.person(2637, y - 2, color, .65)
        d.box(2684, y - 12, length, 24, color, color, 4, .2)
    # Shared evidence branches before the LLM; ranking is not an LLM output.
    d.line([(2410, 913), (2467, 913)], d.blue, 2.8)
    d.arrow([(2467, 913), (2467, 337), (2692, 337)], d.blue)
    d.arrow([(2467, 913), (2467, 1055), (2520, 1055)], d.green)
    d.circle(2467, 913, 6, d.blue, d.blue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    data, poses, metrics, checks, videos = load_sources(output)
    d = Overview()
    d.text(1600, 36, 'Technique-Guided Tai Chi Assessment', 59, True,
           align='center', max_width=3000)
    draw_inputs(d, output, poses)
    draw_inspection(d, poses)
    evidence = draw_evidence(d, metrics)
    draw_outputs(d)
    d.text(1600, 1384, 'Shared geometric evidence for individual coaching and cohort-level ranking',
           29, color=GRAY, align='center', max_width=3000)
    audit = d.save(output, 'Fig2_Method_Overview')
    audit.update({
        'source_figure_data': record(ASSETS / 'figure_data.json'),
        'pose_assets': checks, 'video_frames': videos, 'plotted_metrics': evidence,
        'event_id': data['pose_id'], 'reference_id': 'BV1WE411W7JB', 'student_id': '10',
        'exact_smpl_assets_reused': True,
        'reference_anchor_in_teacher_statistics': False, 'teacher_count': 10,
        'rank_visual_is_schematic': True, 'cohort_rank_computed': False,
        'unified_checkpoint_ranking_is_proposed_not_validated': True,
        'feedback_status': 'Illustrative abridgment of borrowed endpoint F014 feedback, not first-event inference.',
        'range_note': 'Frozen median +/- 2 MAD; individual axes scaled for display; no normative thresholds implied.',
        'timeline_note': 'Schematic correspondence within a form, not to time scale; existing frozen SMPL frames.',
        'annotation_note': 'Small joint circles identify regions, not computed 2D joint projections.',
        'renderer': 'Existing project matplotlib vector diagram API; no generative repainting of SMPL.',
        'model_calls': 0,
    })
    (output / 'Fig2_Method_Overview.audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'validation': audit['text_overlap_check'],
                      'size': audit['png_size'], 'text_items': audit['text_items']}, indent=2))


if __name__ == '__main__':
    main()
