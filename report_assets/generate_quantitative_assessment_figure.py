#!/usr/bin/env python
"""Draw the approved two-branch schematic from frozen experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from aqa3d.tracking import DEFAULT_SMPL_MODEL_PATH, load_primary_smpl_track
from export_qishi_smpl_renders import (
    file_record, read_json, reconstruct, shared_camera, validate_rgba, write_json, yaw_rotation,
)
from generate_coaching_feedback_figure import Diagram


DEFAULT_OUTPUT = PROJECT / 'report_assets/quantitative_assessment_two_branch_v1'
POSE_ROOT = PROJECT / 'teacher_1fps_smpl_dtw_results/10_2_yemafenzong'
PELVIS_ROOT = PROJECT / 'pelvis_quality_results'
DISPLAY_ANCHORS = [0, 12, 25]
PELVIS_FILL, PELVIS_EDGE, PELVIS_ACCENT = '#F4ECF4', '#C9A1CA', '#8E6195'
RANKING_FILL, RANKING_EDGE, RANKING_TITLE = '#E5F5DF', '#69A653', '#15931C'
RANKING_ROWS = ('#338F44', '#68B477', '#A6D1A5')


def load_frozen_data() -> dict:
    paths = {
        'pose_summary': POSE_ROOT / 'summary.json',
        'pose_errors': POSE_ROOT / 'geodesic_errors.npz',
        'pose_matches': POSE_ROOT / 'teacher_anchored_keyframes.csv',
        'teacher_pelvis': PELVIS_ROOT / 'teacher_pelvis_model.npz',
        'student_pelvis': PELVIS_ROOT / 'clips/10_2_yemafenzong/pelvis_signals.npz',
        'pelvis_calibration': PROJECT / 'center_fusion_experiment_results/center_calibration.csv',
        'pose_implementation': PROJECT / 'run_teacher_1fps_smpl_dtw_batch.py',
        'pose_matching_implementation': PROJECT / 'run_teacher_keyframe_smpl_dtw_batch.py',
        'pelvis_implementation': PROJECT / 'aqa3d/pelvis_quality.py',
        'pelvis_aggregation': PROJECT / 'aqa3d/center_fusion.py',
        'rank_implementation': PROJECT / 'aqa3d/rank_fusion.py',
    }
    summary = read_json(paths['pose_summary'])
    assert summary['anchor_strategy'] == 'teacher_uniform_1fps_include_first_last'
    assert summary['global_orientation_included'] is False and summary['score'] is None
    with np.load(paths['pose_errors'], allow_pickle=False) as saved:
        errors = saved['errors_degrees'].copy()
        teacher_frames = saved['teacher_source_frame_indices'].astype(int)
        student_frames = saved['student_source_frame_indices'].astype(int)
        assert errors.shape == (26, 23) and np.isfinite(errors).all()
        np.testing.assert_allclose(errors.mean(), summary['mean_geodesic_distance_degrees'])
    with paths['pose_matches'].open(newline='', encoding='utf-8-sig') as handle:
        matches = list(csv.DictReader(handle))
    assert len(matches) == len(teacher_frames)
    np.testing.assert_array_equal([int(r['teacher_source_frame_0based']) for r in matches], teacher_frames)
    np.testing.assert_array_equal([int(r['student_source_frame_0based']) for r in matches], student_frames)
    with np.load(paths['teacher_pelvis'], allow_pickle=False) as saved:
        move_index = saved['move_names'].tolist().index('yemafenzong')
        signal_names = saved['signal_names'].tolist()
        teacher_ids = saved['teacher_ids'].tolist()
        teacher_profile = saved['median'][move_index].copy()
        progress = saved['progress_grid'].copy()
    with np.load(paths['student_pelvis'], allow_pickle=False) as saved:
        student_profile = saved['profile'].copy()
        np.testing.assert_allclose(saved['progress_grid'], progress)
    assert teacher_profile.shape == student_profile.shape == (7, 201)
    assert np.isfinite(teacher_profile[:3]).all() and np.isfinite(student_profile[:3]).all()
    subjects = [
        dict(subject_id='QxVvRcRn2TA', role='reference', track_id=summary['teacher_track_id'],
             tracking='/home/sqw/VisualSearch/aqa/Tracking/teach/QxVvRcRn2TA_2_yemafenzong/results/demo_QxVvRcRn2TA_2_yemafenzong.pkl',
             frame0=teacher_frames[DISPLAY_ANCHORS].tolist()),
        dict(subject_id='10', role='student', track_id=summary['student_track_id'],
             tracking=summary['student_tracking'], frame0=student_frames[DISPLAY_ANCHORS].tolist()),
    ]
    return dict(
        sources={key: file_record(path) for key, path in paths.items()}, subjects=subjects,
        teacher_frames=teacher_frames.tolist(), student_frames=student_frames.tolist(),
        anchor_orders_displayed=DISPLAY_ANCHORS, errors_degrees=errors.tolist(),
        progress_grid=progress.tolist(), signal_names=signal_names, teacher_ids=teacher_ids,
        teacher_profile=teacher_profile.tolist(), student_profile=student_profile.tolist(),
        experiment_recomputed=False, cohort_ranking_computed=False, pause_branch_included=False,
        pose_aggregation='Uniform mean across matched teacher anchors and 23 local SMPL joints',
        pose_score_calibration=False, fusion_weights={'pose': 0.5, 'pelvis': 0.5},
        teacher_reference_note='QxVvRcRn2TA is the pose anchor and pictured reference; pelvis template uses ten teachers.',
        pelvis_alignment_note='Frozen pelvis profiles use their own cached DTW alignment; not the dense pose-branch path.',
        fusion_note='Approved two-branch schematic; this combined configuration has not been rerun.',
    )


def render_meshes(output: Path, data: dict) -> None:
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import pyrender
    import smplx
    import torch
    import trimesh

    torch.set_num_threads(4)
    smpl = smplx.SMPLLayer(model_path=str(DEFAULT_SMPL_MODEL_PATH), gender='neutral', num_betas=10).eval()
    subjects = []
    for item in data['subjects']:
        track = load_primary_smpl_track(item['tracking'], track_id=item['track_id']).at_source_frames(item['frame0'])
        np.testing.assert_array_equal(track.frame_numbers, np.asarray(item['frame0']) + 1)
        subject = dict(item, track=track)
        reconstruct(subject, smpl, SimpleNamespace(view_mode='canonical-front', canonical_height=1.7))
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
        for item, subject in zip(data['subjects'], subjects):
            item['source_tracking'] = file_record(Path(item['tracking']))
            item['presentation'] = subject['presentation']
            item['images'] = []
            for index, frame0 in enumerate(item['frame0']):
                name = f"{item['subject_id']}_yemafenzong_clip_f{frame0:06d}.png"
                path = output / name
                if path.exists():
                    raise FileExistsError(path)
                mesh = trimesh.Trimesh(vertices=subject['vertices'][index], faces=smpl.faces, process=False)
                node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
                try:
                    pixels, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                finally:
                    scene.remove_node(node)
                picture = Image.fromarray(pixels, 'RGBA').resize((width, height), Image.Resampling.LANCZOS)
                validation = validate_rgba(picture)
                picture.save(path)
                item['images'].append(dict(file=name, source_frame_0based=frame0,
                    phalp_frame_1based=frame0 + 1, sha256=file_record(path)['sha256'], validation=validation))
                print(f'Rendered exact tracked pose: {name}', flush=True)
    finally:
        renderer.delete()
    data['camera'] = dict(pose=camera.tolist(), xmag=xmag, ymag=ymag, projection='orthographic',
                          shared=True, azimuth_degrees=0, elevation_degrees=0)
    data['smpl_model'] = file_record(DEFAULT_SMPL_MODEL_PATH / 'SMPL_NEUTRAL.pkl')
    write_json(output / 'figure_data.json', data)


def dot(d: Diagram, x: float, y: float, color: str, size: float = 5) -> None:
    d.ax.plot([x], [y], 'o', color=color, markersize=size, markeredgecolor='white',
              markeredgewidth=0.7, zorder=4)


def timeline(d: Diagram, x: float, y: float, w: float, color: str, positions=None) -> None:
    d.line([(x, y), (x + w, y)], color, 2.5)
    for end in (x, x + w):
        d.line([(end, y - 10), (end, y + 10)], color, 2)
    if positions is not None:
        for position in positions:
            dot(d, x + w * position, y, color, 4)


def draw_frozen_error_map(d: Diagram, data: dict) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    errors = np.asarray(data['errors_degrees'])
    cmap = LinearSegmentedColormap.from_list('pose_errors', ['#EFF5FC', '#7BABD9', '#E4974E'])
    d.ax.imshow(errors, extent=(1845, 2260, 555, 360), aspect='auto', cmap=cmap,
                vmin=0, vmax=float(errors.max()), interpolation='nearest', zorder=2)
    label = d.text(1820, 458, 'Anchors', 24, color=d.muted, align='center')
    label.set_rotation(90)
    label.set_verticalalignment('center')
    d.text(2048, 568, 'Joints', 24, color=d.muted, align='center')
    d.text(2048, 318, 'Joint Distances', 34, True, align='center', max_width=485)


def draw_frozen_pelvis_profile(d: Diagram, data: dict) -> None:
    index = data['signal_names'].index('support_lateral')
    curves = [np.asarray(data[key])[index] for key in ('teacher_profile', 'student_profile')]
    values = np.concatenate(curves)
    ymin, ymax = float(values.min()), float(values.max())
    span = ymax - ymin
    assert span > 0
    xmin, xmax, ytop, ybottom = 1240, 1690, 1090, 1280
    ymin, ymax = ymin - 0.08 * span, ymax + 0.08 * span
    for curve, color in zip(curves, (d.gold, d.orange)):
        points = np.column_stack((xmin + np.asarray(data['progress_grid']) * (xmax - xmin),
            ybottom - (curve - ymin) / (ymax - ymin) * (ybottom - ytop)))
        d.line(points.tolist(), color, 2.2)
    d.arrow([(xmin, ytop - 10), (xmin, ybottom), (xmax + 12, ybottom)], '#333C44', 1.0)
    d.text(1232, 1294, '0', 22, color=d.muted)
    d.text(1686, 1294, '1', 22, color=d.muted)
    d.text(1465, 1294, 'Aligned form progress', 23, color=d.muted, align='center')
    d.text(1455, 969, 'Trajectory Comparison', 34, True, align='center', max_width=530)
    d.text(1465, 1030, 'Lateral pelvis trajectory', 26, align='center', max_width=490)
    d.line([(1220, 1341), (1260, 1341)], d.gold, 2.4)
    d.text(1274, 1325, 'Teacher Median', 23, color=d.muted)
    d.line([(1530, 1341), (1570, 1341)], d.orange, 2.4)
    d.text(1584, 1325, 'Student', 23, color=d.muted)


def draw_body_axes(d: Diagram) -> None:
    origin = (800, 1235)
    for end in ((800, 1130), (1015, 1235), (985, 1150)):
        d.arrow([origin, end], PELVIS_ACCENT, 2.0)
    dot(d, *origin, PELVIS_ACCENT, 5)
    d.text(800, 1083, 'Height', 27, color=PELVIS_ACCENT, align='center')
    d.text(1000, 1110, 'Forward', 27, color=PELVIS_ACCENT, align='center')
    d.text(975, 1253, 'Lateral', 27, color=PELVIS_ACCENT, align='center')


def draw_rank_symbol(d: Diagram, x: float, y: float) -> None:
    from matplotlib.patches import Circle

    # Anonymous bars illustrate ordering only; they do not encode cohort results.
    for row, (length, color) in enumerate(zip((240, 180, 120), RANKING_ROWS)):
        cy = y + row * 62
        d.ax.add_patch(Circle((x, cy - 7), 8, facecolor=color, edgecolor='none', zorder=3))
        d.line([(x - 13, cy + 17), (x - 9, cy + 8), (x + 9, cy + 8), (x + 13, cy + 17)],
               color, 2.5)
        d.box(x + 43, cy - 10, length, 24, color, color, radius=4, lw=0)


def compose(output: Path, name: str, destination: Path | None = None) -> dict:
    destination = output if destination is None else destination
    destination.mkdir(parents=True, exist_ok=True)
    data = read_json(output / 'figure_data.json')
    for record in data['sources'].values():
        assert file_record(Path(record['path']))['sha256'] == record['sha256'], 'Frozen source changed.'
    for subject in data['subjects']:
        for item in subject['images']:
            assert file_record(output / item['file'])['sha256'] == item['sha256']
    d = Diagram()
    d.text(1500, 33, 'Quantitative Action Quality Assessment', 60, True, align='center', max_width=2900)
    d.text(1500, 110, "Parting the Wild Horse's Mane: form-level sequences", 31,
           color=d.muted, align='center')

    d.box(30, 200, 500, 1230, edge='#5889CE', radius=45)
    d.text(280, 236, 'Segmented SMPL\nSequences', 37, True, align='center', max_width=450)
    for subject, y, label, color in zip(data['subjects'], (430, 860),
            ('Reference', 'Students'), (d.gold, d.orange)):
        d.text(280, y, label, 29, True, color=color, align='center', max_width=450)
        for index, item in enumerate(subject['images']):
            x = 50 + index * 157
            d.picture(output / item['file'], x, y + 75, 145, 145 * 4 / 3)
        timeline(d, 78, y + 303, 405, color)
        d.text(280, y + 327, 'Start                 ...                 End', 25,
               color=d.muted, align='center')
    d.text(280, 1320, 'For each student', 28, True, color=d.muted, align='center', max_width=450)

    d.box(585, 200, 1765, 570, '#E4EFFF', '#5889CE', 48, z=0)
    d.box(840, 200, 1255, 65, '#3671DB', '#2F66BF', 14)
    d.text(1467, 214, '(a) Uniform-Anchor Pose Comparison', 38, True, 'white', 'center', 1205)
    d.box(585, 850, 1765, 580, PELVIS_FILL, PELVIS_EDGE, 48, z=0)
    d.box(910, 850, 1115, 65, PELVIS_ACCENT, PELVIS_ACCENT, 14)
    d.text(1467, 864, '(b) Pelvis Motion Comparison', 38, True, 'white', 'center', 1065)
    d.arrow([(530, 805), (560, 805)], '#7A8C9F', 2.0)
    d.line([(560, 515), (560, 1150)], '#7A8C9F', 2.0)
    dot(d, 560, 805, '#7A8C9F', 4)
    d.arrow([(560, 515), (615, 515)])
    d.arrow([(560, 1150), (615, 1150)], PELVIS_ACCENT)

    # Reference anchors are selected sparsely; DTW still consumes the dense sequences.
    for x, w in ((620, 490), (1170, 570), (1800, 515)):
        d.box(x, 295, w, 405)
        d.box(x, 945, w, 420)
    d.text(865, 318, 'Uniform Sampling', 34, True, align='center', max_width=450)
    d.text(865, 387, '1 FPS', 32, True, d.blue, 'center', 450)
    frames = np.asarray(data['teacher_frames'], dtype=float)
    timeline(d, 663, 488, 407, d.gold, frames / frames[-1])
    d.text(865, 630, 'Reference Anchors', 30, True, d.blue, 'center', 450)
    d.arrow([(1110, 505), (1170, 505)])

    d.text(1455, 318, 'SMPL-DTW Matching', 34, True, align='center', max_width=530)
    selected = [0, 6, 12, 19, 25]
    ref = np.asarray(data['teacher_frames'], dtype=float)
    stu = np.asarray(data['student_frames'], dtype=float)
    d.text(1200, 441, 'Ref.', 24, color=d.gold)
    d.text(1200, 553, 'Stu.', 24, color=d.orange)
    timeline(d, 1280, 458, 420, d.gold)
    timeline(d, 1280, 570, 420, d.orange)
    for i in selected:
        rx, sx = 1280 + 420 * ref[i] / ref[-1], 1280 + 420 * stu[i] / stu[-1]
        dot(d, rx, 458, d.gold)
        dot(d, sx, 570, d.orange)
        d.line([(rx, 466), (sx, 562)], '#727F90', 1.0, dashed=True)
    d.arrow([(1740, 505), (1800, 505)])

    # The heatmap is cached anchor-by-joint error, not a fabricated DTW cost matrix.
    draw_frozen_error_map(d, data)
    d.text(2057, 622, 'Mean Pose Distance', 34, True, d.blue, 'center', 480)

    d.text(865, 969, 'Body-Relative Signals', 32, True, align='center', max_width=455)
    d.text(865, 1030, 'Pelvis - ankle midpoint', 26, align='center', max_width=420)
    draw_body_axes(d)
    d.arrow([(1110, 1160), (1170, 1160)], PELVIS_ACCENT)
    draw_frozen_pelvis_profile(d, data)
    d.arrow([(1740, 1160), (1800, 1160)], PELVIS_ACCENT)

    d.text(2057, 969, 'Pelvis Deviation', 34, True, align='center', max_width=475)
    for y, label in ((1050, 'Horizontal trajectory error'), (1117, 'Endpoint deviation'),
                     (1184, 'Vertical trajectory error')):
        dot(d, 1845, y + 14, PELVIS_ACCENT, 4)
        d.text(1870, y, label, 27, max_width=420)
    d.arrow([(2057, 1230), (2057, 1265)], PELVIS_ACCENT, 2.0)
    d.text(2057, 1290, 'Weighted Sum', 31, True, PELVIS_ACCENT, 'center', 465)

    # The approved two-branch Borda scheme is explanatory only: no cohort ranks are computed.
    d.box(2420, 200, 550, 1230, edge='#637991', radius=45)
    d.text(2695, 236, 'Equal-Weight\nBorda Fusion', 37, True, align='center', max_width=505)
    d.box(2460, 442, 420, 140, '#EDF4FE', '#8FB4E5', 15)
    d.text(2670, 491, 'Pose Rank', 36, True, d.blue, 'center')
    d.arrow([(2315, 512), (2460, 512)])
    d.box(2460, 653, 420, 140, PELVIS_FILL, PELVIS_EDGE, 15)
    d.text(2670, 702, 'Pelvis Rank', 36, True, PELVIS_ACCENT, 'center')
    d.arrow([(2315, 1160), (2384, 1160), (2384, 723), (2460, 723)], PELVIS_ACCENT)
    d.arrow([(2880, 512), (2921, 512), (2921, 940), (2880, 940)])
    d.arrow([(2670, 793), (2670, 865)], PELVIS_ACCENT)
    d.box(2460, 865, 420, 175, '#F4F6F8', '#8996A7', 15)
    d.text(2565, 905, 'Pose', 29, True, d.blue, 'center')
    d.text(2775, 905, 'Pelvis', 29, True, PELVIS_ACCENT, 'center')
    d.text(2565, 951, '50%', 38, True, d.blue, 'center')
    d.text(2775, 951, '50%', 38, True, PELVIS_ACCENT, 'center')
    d.text(2670, 951, '+', 38, True, d.muted, 'center')
    d.arrow([(2670, 1040), (2670, 1100)], '#526B83')
    d.box(2450, 1105, 490, 285, RANKING_FILL, RANKING_EDGE, 30)
    d.box(2470, 1105, 450, 58, RANKING_TITLE, RANKING_TITLE, 12)
    d.text(2695, 1118, 'Relative Ranking', 37, True, 'white', 'center', 420)
    draw_rank_symbol(d, 2517, 1220)

    checks = d.save(destination, name)
    checks.update(exact_tracking_meshes=6, real_cached_error_cells=26 * 23,
                  real_cached_pelvis_profile_points=201, experiments_recomputed=False,
                  cohort_ranks_computed=False, frozen_source_hashes_checked=len(data['sources']),
                  figure_data=file_record(output / 'figure_data.json'),
                  generator=file_record(Path(__file__)),
                  teacher_pelvis_reference='Pointwise median of ten teachers',
                  rank_bars='Anonymous schematic only; no numeric ranking depicted',
                  fusion_weights=data['fusion_weights'],
                  palette={
                      'pose_accent': d.blue,
                      'pelvis': dict(fill=PELVIS_FILL, edge=PELVIS_EDGE, accent=PELVIS_ACCENT),
                      'ranking': dict(fill=RANKING_FILL, edge=RANKING_EDGE, title=RANKING_TITLE,
                                      rows=list(RANKING_ROWS)),
                      'reference': d.gold, 'student': d.orange,
                  })
    write_json(destination / f'{name}_validation.json', checks)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--compose-only', action='store_true')
    parser.add_argument('--figure-output-root', type=Path,
                        help='New figure directory; requires --compose-only and reuses frozen assets.')
    parser.add_argument('--name', default='Quantitative_Assessment_two_branch')
    args = parser.parse_args()
    if args.figure_output_root is not None and not args.compose_only:
        parser.error('--figure-output-root requires --compose-only')
    if not args.compose_only:
        args.output.mkdir(parents=True, exist_ok=False)
        render_meshes(args.output, load_frozen_data())
    checks = compose(args.output, args.name, args.figure_output_root)
    print(checks, flush=True)


if __name__ == '__main__':
    main()
