#!/usr/bin/env python
"""Build a source-grounded coaching diagram with exact tracked SMPL inserts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from aqa3d.endpoint_metrics import MOVE_METRICS, compute_endpoint_metrics, teacher_interval
from aqa3d.tracking import DEFAULT_SMPL_MODEL_PATH, load_stitched_root_track
from export_qishi_smpl_renders import (
    file_record, read_json, reconstruct, shared_camera, validate_rgba, write_json, yaw_rotation,
)
from run_f_group_experiment import validate_response


CASE_ROOT = PROJECT / 'f_group_experiment/first3_five_students_api'
METRIC_ROOT = PROJECT / 'endpoint_metric_results/first3_five_students'
METRIC_NAMES = {
    'left_knee_angle': 'L knee angle',
    'right_knee_angle': 'R knee angle',
    'left_elbow_angle': 'L elbow angle',
    'right_elbow_angle': 'R elbow angle',
    'stance_length_leg_ratio': 'Stance length / leg length',
    'left_wrist_head_up_torso_ratio': 'L wrist height vs. head',
    'left_wrist_head_forward_torso_ratio': 'L wrist reach vs. head',
    'right_wrist_hip_distance_torso_ratio': 'R wrist-to-hip / torso',
}


def load_case() -> dict:
    paths = {
        'input': CASE_ROOT / 'inputs/F014.json',
        'response': CASE_ROOT / 'responses/F014.json',
        'prompt': CASE_ROOT / 'F_prompt.md',
        'reference_manifest': METRIC_ROOT / 'reference_manifest_snapshot.json',
        'alignment': METRIC_ROOT / 'student/10/alignment.json',
        'metric_summary': METRIC_ROOT / 'summary.json',
    }
    case, response = read_json(paths['input']), read_json(paths['response'])
    errors, checks = validate_response(case, response)
    if errors:
        raise ValueError(errors)
    assert len(checks) == 16
    assert [m['metric_id'] for m in case['metrics']] == list(MOVE_METRICS[2])
    for metric in case['metrics']:
        teacher = metric['teacher']
        recomputed = teacher_interval([s['value'] for s in teacher['samples']])
        np.testing.assert_allclose(
            [teacher['median_2mad']['lower'], teacher['median_2mad']['upper'],
             teacher['p10_p90']['lower'], teacher['p10_p90']['upper']],
            [recomputed['median_minus_2mad'], recomputed['median_plus_2mad'],
             recomputed['p10'], recomputed['p90']], rtol=0, atol=1e-9)
    manifest, alignment = read_json(paths['reference_manifest']), read_json(paths['alignment'])
    reference = next(e for e in manifest['endpoint_keyposes'] if e['move_id'] == 2)
    student = next(e for e in alignment['endpoints'] if e['move_id'] == 2)
    assert case['technique'] == reference['final_technique_step'] == student['final_technique_step']
    assert manifest['reference']['video_id'] == 'BV1WE411W7JB'
    assert alignment['subject_id'] == '10'
    assert reference['source_frame_index_0based'] == student['reference_source_frame_0based']
    subjects = [
        dict(subject_id='BV1WE411W7JB', role='reference',
             frame0=reference['source_frame_index_0based'],
             source_time=reference['source_frame_time_seconds'],
             tracking=manifest['reference']['tracking_path']),
        dict(subject_id='10', role='student', frame0=student['center_source_frame_0based'],
             source_time=student['source_frame_time_seconds'], tracking=alignment['input_tracking']),
    ]
    return dict(case=case, response=response, subjects=subjects,
                teacher_ids=read_json(paths['metric_summary'])['teacher_ids'],
                sources={key: file_record(path) for key, path in paths.items()},
                numerical_comparisons_verified=len(checks))


def render_poses(output: Path, data: dict) -> None:
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import pyrender
    import smplx
    import torch
    import trimesh

    torch.set_num_threads(4)
    smpl = smplx.SMPLLayer(model_path=str(DEFAULT_SMPL_MODEL_PATH), gender='neutral', num_betas=10).eval()
    subjects = []
    for item in data['subjects']:
        track = load_stitched_root_track(item['tracking']).at_source_frames([item['frame0']])
        subject = dict(item, track=track)
        reconstruct(subject, smpl, SimpleNamespace(view_mode='canonical-front', canonical_height=1.7))
        if item['role'] == 'student':
            measured = compute_endpoint_metrics(track.to_smpl24_joints(), [item['frame0']])
            lookup = measured.set_index('metric_id')['value'].to_dict()
            for metric in data['case']['metrics']:
                np.testing.assert_allclose(lookup[metric['metric_id']], metric['student']['center_value'],
                                           rtol=0, atol=1e-3)
            data['student_center_metrics_verified'] = 8
        subjects.append(subject)
    width, height = 768, 1024
    camera, xmag, ymag = shared_camera([s['vertices'] for s in subjects], width, height, 0, 0)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=0.05, zfar=30), pose=camera)
    for angle, intensity in ((-35, 2.3), (50, 1.1), (160, 0.8)):
        pose = camera.copy()
        pose[:3, :3] = yaw_rotation(np.radians(angle)) @ camera[:3, :3]
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity), pose=pose)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.65098, 0.74118, 0.85882, 1.0], metallicFactor=0, roughnessFactor=0.65)
    renderer = pyrender.OffscreenRenderer(width * 2, height * 2)
    try:
        for subject, item in zip(subjects, data['subjects']):
            mesh = trimesh.Trimesh(vertices=subject['vertices'][0], faces=smpl.faces, process=False)
            node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
            try:
                pixels, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            finally:
                scene.remove_node(node)
            picture = Image.fromarray(pixels, 'RGBA').resize((width, height), Image.Resampling.LANCZOS)
            validation = validate_rgba(picture)
            name = f"{subject['subject_id']}_yemafenzong_f{subject['frame0']:06d}.png"
            picture.save(output / name)
            item.update(image=name, image_sha256=file_record(output / name)['sha256'],
                        source_tracking=file_record(Path(item['tracking'])),
                        phalp_frame_1based=subject['frame0'] + 1,
                        track_id=int(subject['track'].source_track_ids[0]),
                        presentation=subject['presentation'], image_validation=validation)
            print(f"Rendered {name}: exact tracking; canonical frontal view", flush=True)
    finally:
        renderer.delete()
    data['camera'] = dict(pose=camera.tolist(), azimuth_degrees=0, elevation_degrees=0,
                          projection='orthographic', xmag=xmag, ymag=ymag, shared=True)
    data['smpl_model'] = file_record(DEFAULT_SMPL_MODEL_PATH / 'SMPL_NEUTRAL.pkl')
    data['render_helper'] = file_record(PROJECT / 'export_qishi_smpl_renders.py')
    data['measurement_note'] = 'Frozen window medians; SMPL inserts show center frames. No measurements were replaced.'
    data['output_note'] = 'English abridgment of F014 model output, not an expert diagnosis or a new model call.'
    write_json(output / 'figure_data.json', data)


class Diagram:
    width, height, dpi = 3000, 1600, 150
    blue, green, gold, orange = '#1670E8', '#169521', '#CBAB29', '#EF741D'
    ink, muted = '#121820', '#546173'

    def __init__(self):
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams.update({'font.family': 'Liberation Sans', 'svg.fonttype': 'none',
                                    'pdf.fonttype': 42, 'image.composite_image': False})
        import matplotlib.pyplot as plt
        self.fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, self.width), ylim=(self.height, 0))
        self.ax.set_axis_off()
        self.text_checks = []

    def box(self, x, y, w, h, fill='white', edge='#141414', radius=30, lw=1.3, z=1):
        from matplotlib.patches import FancyBboxPatch
        self.ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=fill, edgecolor=edge,
            linewidth=lw, zorder=z))

    def text(self, x, y, text, size=28, bold=False, color=None, align='left', max_width=None):
        artist = self.ax.text(x, y, text, fontsize=size * 72 / self.dpi,
            fontweight='bold' if bold else 'normal', color=color or self.ink,
            ha=align, va='top', linespacing=1.23, zorder=5)
        self.text_checks.append((artist, max_width))
        return artist

    def line(self, points, color='#CCD5DF', width=1.0, dashed=False):
        x, y = zip(*points)
        self.ax.plot(x, y, color=color, linewidth=width, linestyle='--' if dashed else '-', zorder=3)

    def arrow(self, points, color=None, width=2.8):
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import FancyArrowPatch
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        self.ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=16,
            linewidth=width, color=color or self.blue, zorder=4))

    def picture(self, path, x, y, w, h):
        with Image.open(path) as picture:
            pixels = np.array(picture)
        assert abs(pixels.shape[1] / pixels.shape[0] - w / h) < 1e-8
        self.ax.imshow(pixels, extent=(x, x + w, y + h, y), aspect='auto', zorder=2)

    def save(self, output: Path, name: str) -> dict:
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        bounds = []
        for artist, max_width in self.text_checks:
            bb = artist.get_window_extent(renderer)
            if max_width is not None and bb.width > max_width + 1:
                raise ValueError(f'Text too wide ({bb.width:.1f}>{max_width}): {artist.get_text()}')
            if bb.x0 < 0 or bb.x1 > self.width or bb.y0 < 0 or bb.y1 > self.height:
                raise ValueError(f'Text outside canvas: {artist.get_text()}')
            bounds.append((artist.get_text(), bb))
        for i, (left, a) in enumerate(bounds):
            for right, b in bounds[i + 1:]:
                if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                    raise ValueError(f'Overlapping text: {left!r} / {right!r}')
        for extension in ('png', 'svg', 'pdf'):
            path = output / f'{name}.{extension}'
            if path.exists():
                raise FileExistsError(path)
        for extension in ('png', 'svg', 'pdf'):
            self.fig.savefig(output / f'{name}.{extension}', dpi=self.dpi, facecolor='white')
        return dict(text_items=len(bounds), text_overlap_check='passed', text_width_check='passed',
                    png_size=[self.width, self.height])


def metric_range(metric: dict, key: str) -> str:
    digits = 1 if metric['unit'] == 'degree' else 3
    interval = metric['teacher'][key]
    return f"[{interval['lower']:.{digits}f}, {interval['upper']:.{digits}f}]"


def model_icon(d: Diagram, x: float, y: float, size: float) -> None:
    d.box(x, y, size, size, edge=d.blue, radius=size * 0.14, lw=1.8)
    for fraction in (0.26, 0.5, 0.74):
        p, pin = size * fraction, size * 0.12
        for points in (
            [(x - pin, y + p), (x, y + p)],
            [(x + size, y + p), (x + size + pin, y + p)],
            [(x + p, y - pin), (x + p, y)],
            [(x + p, y + size), (x + p, y + size + pin)],
        ):
            d.line(points, d.blue, 1.3)
    nodes = [(x + size * a, y + size * b)
             for a, b in ((0.25, 0.32), (0.25, 0.68), (0.55, 0.5), (0.78, 0.28), (0.78, 0.72))]
    for a, b in ((0, 2), (1, 2), (2, 3), (2, 4)):
        d.line([nodes[a], nodes[b]], d.blue, 1.2)
    for px, py in nodes:
        d.ax.plot(px, py, 'o', color=d.blue, markersize=4, zorder=4)


def checkpoint_evidence(d: Diagram, metric: dict, label: str, color: str, y: float) -> dict:
    from matplotlib.patches import Rectangle

    teacher = metric['teacher']
    student = metric['student']['value']
    reference_range = teacher['p10_p90']
    limits = [student, reference_range['lower'], reference_range['upper']]
    lo, hi = min(limits), max(limits)
    margin = (hi - lo) * 0.15
    if margin <= 0:
        raise ValueError(f'Cannot plot a zero-width range for {metric["metric_id"]}')
    domain = (lo - margin, hi + margin)
    x0, x1 = 737, 1555
    px = lambda value: x0 + (value - domain[0]) / (domain[1] - domain[0]) * (x1 - x0)
    digits = 1 if metric['unit'] == 'degree' else 3
    value_text = f'{student:.{digits}f}' + ('\N{DEGREE SIGN}' if digits == 1 else '')
    d.text(737, y, label, 30, True, color, max_width=635)
    d.text(1555, y, value_text, 28, True, d.orange, 'right', 130)
    line_y = y + 94
    d.line([(x0, line_y), (x1, line_y)], '#CCD5DF', 1.5)
    d.ax.add_patch(Rectangle((px(reference_range['lower']), line_y - 11),
                            px(reference_range['upper']) - px(reference_range['lower']), 22,
                            facecolor=d.gold, edgecolor='none', alpha=0.25, zorder=2,
                            gid=f'teacher-range-{metric["metric_id"]}'))
    d.line([(px(reference_range['lower']), line_y),
            (px(reference_range['upper']), line_y)], d.gold, 2.6)
    for bound in (reference_range['lower'], reference_range['upper']):
        d.line([(px(bound), line_y - 14), (px(bound), line_y + 14)], d.gold, 1.1)
    d.ax.plot(px(student), line_y, 'o', markersize=9, markerfacecolor=d.orange,
              markeredgecolor='white', markeredgewidth=1.5, zorder=5,
              gid=f'student-value-{metric["metric_id"]}')
    return dict(metric_id=metric['metric_id'], label=label, color=color,
                student_value=student, displayed_student_value=value_text,
                displayed_range='p10_p90', p10_p90=reference_range,
                plot_domain=list(domain), student_plot_x=px(student),
                plot_x_bounds=[x0, x1],
                teacher_plot_x={'p10_p90': [px(reference_range['lower']), px(reference_range['upper'])]})


def compose(output: Path, name: str, *, data: dict | None = None,
            destination: Path | None = None) -> None:
    if data is None:
        data = read_json(output / 'figure_data.json')
        current = load_case()
        assert current['sources'] == data['sources'], 'Frozen sources changed after rendering.'
    else:
        for record in data['sources'].values():
            assert file_record(Path(record['path']))['sha256'] == record['sha256']
    figure_text = data.get('figure_text', {})
    for subject in data['subjects']:
        assert file_record(output / subject['image'])['sha256'] == subject['image_sha256']
    d = Diagram()
    destination = output if destination is None else destination
    knee_color, stance_color, wrist_color = '#786092', '#E4523F', d.blue
    checkpoints = (
        ('left_knee_angle', 'Left knee angle', knee_color),
        ('stance_length_leg_ratio', 'Normalized stance length', stance_color),
        ('left_wrist_head_up_torso_ratio', 'Left wrist height', wrist_color),
    )
    d.text(1500, 35, 'Technique-Guided Coaching Feedback Generation', 60, True, align='center', max_width=2850)
    d.box(25, 128, 2950, 424, '#E4EFFF', '#5889CE', 65, 1.3, z=0)
    d.box(810, 128, 1380, 72, '#3671DB', '#2F66BF', 16, 1.1)
    d.text(1500, 144, '(a) Offline Checkpoint Design', 43, True, 'white', 'center', 1330)

    d.box(60, 230, 970, 295)
    d.text(545, 248, 'Technique Description', 37, True, align='center')
    d.text(95, 303, figure_text.get('technique_title', "Parting the Wild Horse's Mane: ending pose"),
           28, True, max_width=900)
    for y, phrase, color in (
        (352, 'Bend the left knee into a bow stance.', knee_color),
        (407, 'Step forward into a wide stance.', stance_color),
        (462, 'Raise the left hand to eye level.', wrist_color),
    ):
        d.text(95, y, phrase, 31, color=color, max_width=900)

    d.box(1085, 230, 345, 295)
    d.text(1257, 270, 'GPT-6', 52, True, d.blue, 'center')
    model_icon(d, 1222, 355, 70)
    d.text(1257, 470, 'Select checkpoints', 28, align='center', max_width=315)
    d.arrow([(1030, 375), (1080, 375)])
    d.arrow([(1430, 375), (1480, 375)])

    d.box(1485, 230, 1455, 295)
    d.text(2212, 248, 'Selected Checkpoints', 37, True, align='center')
    for x, label, color in (
        (1720, 'Left knee\nangle', knee_color),
        (2212, 'Normalized\nstance length', stance_color),
        (2700, 'Left wrist height\nrelative to head', wrist_color),
    ):
        d.text(x, 320, label, 31, True, color, 'center', 430)
    from matplotlib.patches import Arc, Circle, FancyArrowPatch
    joint_points = [(1690, 410), (1720, 465), (1785, 465)]
    d.line(joint_points, knee_color, 2.8)
    for px, py in joint_points:
        d.ax.add_patch(Circle((px, py), 7, facecolor='white', edgecolor=knee_color, linewidth=1.7, zorder=4))
    d.ax.add_patch(Arc((1720, 465), 60, 60, theta1=241, theta2=360,
                       edgecolor=knee_color, linewidth=1.5, zorder=4))
    d.ax.add_patch(FancyArrowPatch((2112, 452), (2312, 452), arrowstyle='<->',
                                  mutation_scale=15, color=stance_color, linewidth=2, zorder=4))
    for x in (2112, 2312):
        d.line([(x, 428), (x, 476)], stance_color, 2)
    d.ax.add_patch(Circle((2668, 420), 12, facecolor='white', edgecolor=wrist_color, linewidth=1.7, zorder=4))
    d.ax.add_patch(Circle((2700, 476), 7, facecolor='white', edgecolor=wrist_color, linewidth=1.7, zorder=4))
    d.line([(2668, 438), (2668, 462), (2700, 476)], wrist_color, 1.7)
    d.ax.add_patch(FancyArrowPatch((2750, 420), (2750, 476), arrowstyle='<->',
                                  mutation_scale=13, color=wrist_color, linewidth=1.7, zorder=4))

    d.box(25, 620, 2950, 900, '#E5F6DF', '#469B39', 65, 1.3, z=0)
    d.box(1050, 620, 1060, 70, '#11991A', '#087E10', 16, 1.1)
    d.text(1580, 636, '(b) Evidence-Based Coaching', 43, True, 'white', 'center', 1010)
    d.arrow([(2200, 525), (2200, 580), (910, 580), (910, 720)], d.blue)

    d.box(60, 720, 580, 750)
    d.text(350, 747, 'Localized SMPL Poses', 36, True, align='center', max_width=540)
    pose_label = 'First left bow stance' if data.get('pose_id') == '2.part_mane_1_L' else 'Ending pose'
    d.text(350, 814, pose_label, 28, align='center', max_width=530)
    d.text(210, 908, 'Reference', 29, True, d.gold, 'center')
    d.text(490, 908, 'Student', 29, True, d.orange, 'center')
    for subject, x in zip(data['subjects'], (72, 352)):
        d.picture(output / subject['image'], x, 965, 276, 368)

    d.box(695, 720, 920, 750)
    d.text(1155, 747, 'Measurement & Structured Evidence', 36, True, align='center', max_width=870)
    d.line([(740, 835), (805, 835)], d.gold, 3)
    d.text(822, 818, 'Teacher Ranges', 27)
    d.ax.plot(1315, 835, 'o', color=d.orange, markersize=8, zorder=4)
    d.text(1340, 818, 'Student', 27)
    lookup = {metric['metric_id']: metric for metric in data['case']['metrics']}
    evidence_rows = []
    for i, (metric_id, label, color) in enumerate(checkpoints):
        evidence_rows.append(checkpoint_evidence(d, lookup[metric_id], label, color, 900 + i * 195))
        if i < 2:
            d.line([(737, 1058 + i * 195), (1555, 1058 + i * 195)], '#E3E8EE', 0.7)

    d.box(1670, 720, 555, 750)
    d.text(1947, 747, 'LLM Reasoning', 37, True, align='center')
    model_icon(d, 1902, 855, 90)
    for y, label in ((1040, 'Compare checkpoints'), (1160, 'Integrate evidence'),
                     (1280, 'Handle uncertainty')):
        d.text(1947, y, label, 32, True, align='center', max_width=495)
    for y in (1090, 1210):
        d.arrow([(1947, y), (1947, y + 44)], d.green, 2)

    d.box(2280, 720, 660, 750)
    d.text(2610, 747, 'Coaching Feedback', 37, True, align='center')
    excerpts = [
        ('P1', 'Bow stance', 'Lengthen your stance.\nDeepen the front-knee bend.'),
        ('P3', 'Left-hand height', 'Insufficient evidence;\nno height correction.'),
    ]
    d.text(2315, 878, 'Bow stance', 30, True)
    d.text(2315, 935, 'Lengthen your stance.', 31, color=stance_color, max_width=590)
    d.text(2315, 994, 'Deepen the front-knee bend.', 31, color=knee_color, max_width=590)
    d.text(2315, 1100, 'Left-hand height', 30, True, wrist_color)
    d.text(2315, 1157, 'Insufficient evidence;\nno height correction.', 30, color=wrist_color, max_width=590)
    d.line([(2315, 1294), (2905, 1294)], '#9FBCE7', 1.1)
    d.text(2315, 1315, 'Coach summary', 30, True)
    summary = 'Focus on a fuller bow stance.'
    d.text(2315, 1370, summary, 30, max_width=590)
    for start, end in ((640, 695), (1615, 1670), (2225, 2280)):
        d.arrow([(start, 1095), (end - 4, 1095)], d.green)
    checks = d.save(destination, name)
    write_json(destination / f'{name}.audit.json', dict(
        checks=checks, displayed_evidence=evidence_rows,
        displayed_checkpoint_ids=[row[0] for row in checkpoints],
        full_checkpoint_count=len(data['case']['metrics']),
        semantic_colors={label: color for _, label, color in checkpoints},
        reference_ranges={'Teacher Ranges': 'P10-P90 (formerly B)'},
        evidence_plot_note='Each metric uses its own linear scale; only the frozen P10-P90 range is displayed.',
        feedback_excerpts=[dict(finding_id=fid, heading=heading, english_abridgment=body)
                           for fid, heading, body in excerpts],
        coach_summary_abridgment=summary,
        source_case='F014', source_response_sha256=data['sources']['response']['sha256'],
        geometry_event_id=data.get('pose_id', 'move_2_endpoint'),
        feedback_status=data.get('feedback_status', 'same_case_model_output'),
        caption_required='Feedback is borrowed endpoint output, not inference on first-event measurements.'
                         if data.get('feedback_status') else None,
        note=figure_text.get('audit_note',
            'Abridged translations of original output, not newly inferred or expert-verified statements.'),
        generator=file_record(Path(__file__)),
        frozen_figure_data=file_record(output / 'figure_data.json'),
        smpl_images=[file_record(output / subject['image']) for subject in data['subjects']],
    ))
    print(f'Saved {name}.png / .svg / .pdf; all {checks["text_items"]} text items fit without overlap.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=PROJECT / 'report_assets/coaching_feedback_yemafenzong_v1')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--render-only', action='store_true')
    mode.add_argument('--compose-only', action='store_true')
    parser.add_argument('--figure-name', default='coaching_feedback_v1')
    parser.add_argument('--figure-output-root', type=Path,
                        help='New destination using frozen figure_data.json; requires --compose-only.')
    args = parser.parse_args()
    if args.figure_output_root is not None and not args.compose_only:
        parser.error('--figure-output-root requires --compose-only')
    if not args.compose_only:
        data = load_case()
        args.output_root.mkdir(parents=True, exist_ok=False)
        render_poses(args.output_root, data)
    if not args.render_only:
        if args.figure_output_root is not None:
            data = read_json(args.output_root / 'figure_data.json')
            args.figure_output_root.mkdir(parents=True, exist_ok=False)
            compose(args.output_root, args.figure_name, data=data, destination=args.figure_output_root)
        else:
            compose(args.output_root, args.figure_name)


if __name__ == '__main__':
    main()
