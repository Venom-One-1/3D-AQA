#!/usr/bin/env python
"""Source-audited first-event KEL and coaching figures; no new model inference."""

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

from aqa3d.endpoint_metrics import MOVE_METRICS, compute_endpoint_metrics, teacher_interval
from aqa3d.smpl_dtw import pairwise_geodesic_costs
from aqa3d.tracking import DEFAULT_SMPL_MODEL_PATH, load_stitched_root_track
from export_qishi_smpl_renders import (
    file_record, read_json, reconstruct, shared_camera, validate_rgba, write_json, yaw_rotation,
)
from generate_coaching_feedback_figure import Diagram, compose as compose_coaching
from run_f_group_experiment import validate_response


POSE_ID = '2.part_mane_1_L'
EVENT_ID = 'part_mane_1_L'
TRANSFER = PROJECT / 'temporal_keypose_transfer_results/first3_v2'
LOCALIZATION = PROJECT / 'temporal_keypose_results/first3_v2/BV1WE411W7JB/02_yemafenzong'
METRICS = PROJECT / 'temporal_event_metric_results/first3_five_students_v1'
F_GROUP = PROJECT / 'f_group_experiment/first3_five_students_api'


def only_event(events: list[dict]) -> dict:
    matches = [event for event in events if event.get('pose_id') == POSE_ID]
    if len(matches) != 1:
        raise ValueError(f'Expected one {POSE_ID}; found {len(matches)}')
    return matches[0]


def load_evidence() -> dict:
    paths = {
        'reference_events': TRANSFER / 'reference/02_yemafenzong/keypose_events.json',
        'student_events': TRANSFER / 'student/10/keypose_events.json',
        'dtw_path': TRANSFER / 'student/10/dtw_path.npz',
        'signals': LOCALIZATION / 'motion_signals.npz',
        'query': LOCALIZATION / 'temporal_queries.json',
        'teacher_reference': METRICS / 'teacher_reference.json',
        'student_comparison': METRICS / 'student_comparison.json',
        'metric_definitions': METRICS / 'metric_definitions.json',
        'metric_summary': METRICS / 'summary.json',
        'input_provenance': METRICS / 'input_provenance.json',
        'reference_measurements': METRICS / 'reference/BV1WE411W7JB/measurements.json',
        'student_measurements': METRICS / 'student/10/measurements.json',
        'input': F_GROUP / 'inputs/F014.json',
        'response': F_GROUP / 'responses/F014.json',
        'prompt': F_GROUP / 'F_prompt.md',
    }
    source = {key: file_record(path) for key, path in paths.items()}
    reference = only_event(read_json(paths['reference_events'])['events'])
    target = only_event(read_json(paths['student_events'])['events'])
    assert reference['representative_source_frame_0based'] == 686
    assert target['points']['representative']['target_source_frame_0based'] == 623
    assert reference['technique'] == target['technique']
    assert target['status'] == 'mapped' and not target['review_required']
    query = read_json(paths['query'])
    event_query = next(e for e in query['events'] if e['event_id'] == EVENT_ID)
    assert event_query['terminal_extension'] == 'none'
    assert event_query['after_event']['event_id'] == 'hold_ball_1'

    summaries = read_json(paths['metric_summary'])
    assert summaries['llm_called'] is False
    assert summaries['reference_anchor_in_teacher_statistics'] is False
    teacher = {item['metric_id']: item for item in read_json(paths['teacher_reference'])
               if item['pose_id'] == POSE_ID}
    student = {item['metric_id']: item for item in read_json(paths['student_comparison'])
               if item['pose_id'] == POSE_ID and item['subject_id'] == '10'}
    expected = list(MOVE_METRICS[2])
    assert set(student) == set(teacher) == set(expected)
    metrics = []
    for metric_id in expected:
        row, reference_range = student[metric_id], teacher[metric_id]['reference']
        assert row['representative_source_frame_0based'] == 623
        assert row['kind'] == 'pose' and row['status'] == 'valid'
        assert reference_range['valid_teacher_count'] == 10
        raw = reference_range['teacher_values']
        assert {v['subject_id'] for v in raw} == set(summaries['teacher_ids'])
        calculated = teacher_interval([v['value'] for v in raw])
        for key in ('median', 'mad', 'median_minus_2mad', 'median_plus_2mad', 'p10', 'p90'):
            np.testing.assert_allclose(calculated[key], reference_range[key], rtol=0, atol=1e-9)
        metrics.append(dict(
            metric_id=metric_id, unit=row['unit'], label=row['label'],
            student=row,
            teacher=dict(median_2mad=dict(lower=reference_range['median_minus_2mad'],
                                          upper=reference_range['median_plus_2mad']),
                         p10_p90=dict(lower=reference_range['p10'], upper=reference_range['p90']),
                         samples=raw, source=reference_range),
        ))

    borrowed_input, response = read_json(paths['input']), read_json(paths['response'])
    errors, checked = validate_response(borrowed_input, response)
    if errors:
        raise ValueError(errors)
    assert len(checked) == 16
    # Validate borrowed feedback ONLY against its original endpoint input, never first-event data.
    subjects = []
    for role, subject_id in (('reference', 'BV1WE411W7JB'), ('student', '10')):
        measurements = read_json(paths[f'{role}_measurements'])
        records = [r for r in measurements['measurements'] if r['pose_id'] == POSE_ID]
        frame = 686 if role == 'reference' else 623
        assert len(records) == 8 and all(r['representative_source_frame_0based'] == frame for r in records)
        metadata = measurements['subject']
        subjects.append(dict(role=role, subject_id=subject_id, frame0=frame,
                             source_time=frame / metadata['source_fps'], fps=metadata['source_fps'],
                             tracking=metadata['input_tracking']['path'], measurements=records))
    return dict(
        pose_id=POSE_ID, event_query=event_query, move=query['move'],
        reference_event=reference, target_event=target,
        case=dict(case_id='first_bow_stance_student_10', technique=reference['technique'], metrics=metrics),
        response=response, teacher_ids=summaries['teacher_ids'], subjects=subjects, sources=source,
        feedback_status='borrowed_endpoint_F014_user_authorized_illustration_only',
        first_event_llm_called=False, first_event_feedback_available=False,
        figure_text=dict(
            technique_title="Parting the Wild Horse's Mane: first left bow stance",
            window_note='Student value: median within +/-0.2 s of the representative frame.',
            feedback_caption='Illustration: end-pose feedback reused',
            footer='Draft example: first-event poses and measurements; feedback borrowed from the third bow stance. First-event Qwen run pending.',
            audit_note='User authorized reusing F014 endpoint feedback as a figure placeholder. '
                       'These excerpts are NOT an inference result for the first event. '
                       'Geometry, curves and teacher ranges all come from 2.part_mane_1_L.',
        ),
    )


def prepare_assets(output: Path, data: dict) -> None:
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import pyrender
    import smplx
    import torch
    import trimesh

    torch.set_num_threads(4)
    with np.load(Path(data['sources']['dtw_path']['path'])) as archive:
        path = {key: archive[key].copy() for key in archive.files}
    with np.load(Path(data['sources']['signals']['path'])) as archive:
        signals = {key: archive[key].copy() for key in archive.files}
    mask = signals[f'event__{EVENT_ID}__search_mask']
    data['effective_search_seconds'] = signals['time_seconds'][mask][[0, -1]].tolist()
    assert data['effective_search_seconds'][0] < data['reference_event']['start_time_seconds']

    reference_path = path['reference_source_frame_indices_0based']
    target_path = path['target_source_frame_indices_0based']
    valid_pair = ((reference_path >= data['reference_event']['reference_segment_start_source_frame_0based']) &
                  (target_path >= data['target_event']['target_segment_start_source_frame_0based']))
    first_reference_frame = int(reference_path[valid_pair][0])
    reference_frames = [first_reference_frame, 560, 624, 686]
    target_frames, sequence_pairs = [], []
    for frame in reference_frames:
        indices = path['reference_source_frame_indices_0based']
        nearest = int(indices[np.argmin(abs(indices - frame))])
        candidates = np.unique(path['target_source_frame_indices_0based'][indices == nearest])
        candidates = candidates[(candidates >= data['target_event']['target_segment_start_source_frame_0based']) &
                                 (candidates <= data['target_event']['target_segment_end_source_frame_0based'])]
        assert len(candidates), 'Sequence illustration must remain inside the target form.'
        mapped = int(candidates[(len(candidates) - 1) // 2])
        target_frames.append(mapped)
        sequence_pairs.append(dict(reference_frame0=frame, reference_grid_frame0=nearest,
                                   student_frame0=mapped, selection='lower_median_path_candidate'))
    assert target_frames[-1] == 623 and np.all(np.diff(target_frames) > 0)
    data['sequence_pairs'] = sequence_pairs

    smpl = smplx.SMPLLayer(model_path=str(DEFAULT_SMPL_MODEL_PATH), gender='neutral', num_betas=10).eval()
    tracks, rendered = {}, []
    for item, frames in zip(data['subjects'], (reference_frames, target_frames)):
        track = load_stitched_root_track(item['tracking'])
        tracks[item['role']] = track
        window_frames = item['measurements'][0]['window_source_frames_0based']
        selected = track.at_source_frames(window_frames)
        measured = compute_endpoint_metrics(selected.to_smpl24_joints(), window_frames)
        for row in item['measurements']:
            numbers = measured[measured.metric_id == row['metric_id']]['value'].to_numpy()
            center = numbers[window_frames.index(item['frame0'])]
            np.testing.assert_allclose([np.median(numbers), center],
                                       [row['value'], row['center_value']], atol=1e-3, rtol=0)
        item['metric_checks'] = '8 centers and 8 window medians independently recomputed'
        subject = dict(item, frames=frames, track=track.at_source_frames(frames))
        reconstruct(subject, smpl, SimpleNamespace(view_mode='canonical-front', canonical_height=1.7))
        item['presentation'] = subject['presentation']
        item['tracking_source'] = file_record(Path(item['tracking']))
        rendered.append(subject)

    # Recompute only the displayed local-cost block; retain the saved global DTW path.
    ref_bounds = (data['reference_event']['reference_segment_start_source_frame_0based'],
                  data['reference_event']['reference_segment_end_source_frame_0based'])
    tar_bounds = (data['target_event']['target_segment_start_source_frame_0based'],
                  data['target_event']['target_segment_end_source_frame_0based'])
    reference_all = path['reference_source_frame_indices_0based']
    target_all = path['target_source_frame_indices_0based']
    rframes = np.unique(reference_all[(reference_all >= ref_bounds[0]) & (reference_all <= ref_bounds[1])])
    tframes = np.unique(target_all[(target_all >= tar_bounds[0]) & (target_all <= tar_bounds[1])])
    costs = pairwise_geodesic_costs(tracks['student'].at_source_frames(tframes).body_poses,
                                    tracks['reference'].at_source_frames(rframes).body_poses)
    path_mask = ((reference_all >= rframes[0]) & (reference_all <= rframes[-1]) &
                 (target_all >= tframes[0]) & (target_all <= tframes[-1]))
    row = np.searchsorted(tframes, target_all[path_mask])
    col = np.searchsorted(rframes, reference_all[path_mask])
    np.testing.assert_allclose(costs[row, col], path['local_geodesic_radians'][path_mask], atol=2e-5, rtol=0)
    np.savez_compressed(output / 'plot_data.npz', costs_degrees=np.degrees(costs),
                        dtw_row=row, dtw_col=col, reference_frames=rframes, target_frames=tframes,
                        times=signals['time_seconds'],
                        wrist=signals['metric__L_wrist_up__smoothed'],
                        ankles=signals['metric__ankle_distance_value__smoothed'],
                        knee=signals['metric__L_knee_value__smoothed'], search_mask=mask)
    data['dtw_plot'] = dict(cost_shape=list(costs.shape), saved_path_points=len(row),
                            recomputed_path_cost_check='passed', dtw_recomputed=False,
                            joint_distance='mean SO(3) geodesic over 23 local body_pose rotations')

    width, height = 768, 1024
    camera, xmag, ymag = shared_camera([s['vertices'] for s in rendered], width, height, 0, 0)
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
        for subject, item in zip(rendered, data['subjects']):
            item['images'] = []
            for i, frame in enumerate(subject['frames']):
                mesh = trimesh.Trimesh(vertices=subject['vertices'][i], faces=smpl.faces, process=False)
                node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
                try:
                    pixels, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                finally:
                    scene.remove_node(node)
                picture = Image.fromarray(pixels, 'RGBA').resize((width, height), Image.Resampling.LANCZOS)
                check = validate_rgba(picture)
                name = f"{subject['subject_id']}_first_bow_f{frame:06d}.png"
                picture.save(output / name)
                record = dict(image=name, sha256=file_record(output / name)['sha256'], frame0=frame,
                              phalp_frame_1based=frame + 1, time_seconds=frame / item['fps'],
                              track_id=int(subject['track'].source_track_ids[i]), validation=check)
                item['images'].append(record)
                if frame == item['frame0']:
                    item.update(image=name, image_sha256=record['sha256'])
                print(f'Rendered {name}', flush=True)
    finally:
        renderer.delete()
    data['camera'] = dict(pose=camera.tolist(), xmag=xmag, ymag=ymag, azimuth=0, elevation=0,
                          projection='orthographic', shared_for_all_subjects_and_frames=True)
    data['smpl_model'] = file_record(DEFAULT_SMPL_MODEL_PATH / 'SMPL_NEUTRAL.pkl')
    data['plot_data'] = file_record(output / 'plot_data.npz')
    write_json(output / 'figure_data.json', data)


def verify_assets(output: Path, data: dict) -> None:
    assert data['pose_id'] == POSE_ID
    for record in [*data['sources'].values(), data['plot_data'], data['smpl_model']]:
        assert file_record(Path(record['path']))['sha256'] == record['sha256']
    for subject in data['subjects']:
        for record in subject['images']:
            assert file_record(output / record['image'])['sha256'] == record['sha256']


def span(d: Diagram, x: float, y: float, w: float, h: float, color: str, alpha=1.0) -> None:
    from matplotlib.patches import Rectangle
    d.ax.add_patch(Rectangle((x, y), w, h, color=color, linewidth=0, alpha=alpha, zorder=2))


def timeline(d: Diagram, start: float, end: float, y: float, marks: tuple, color: str) -> None:
    left, representative, right = marks
    span(d, left, y - 17, right - left, 34, color, 0.20)
    d.line([(start, y), (end, y)], color, 2.5)
    for x in (start, end):
        d.line([(x, y - 12), (x, y + 12)], color, 2.5)
    for x in (left, right):
        d.line([(x, y - 22), (x, y + 22)], color, 1.0)
    d.ax.plot(representative, y, 'o', markersize=7.5, markerfacecolor=color,
              markeredgecolor='#151515', markeredgewidth=1, zorder=5)


def compose_kel(output: Path, name: str, data: dict, *, destination: Path | None = None) -> None:
    d = Diagram()
    destination = output if destination is None else destination
    ref, student = data['subjects']
    event, target = data['reference_event'], data['target_event']
    cues = (
        ('wrist', 'L_wrist_up', 'Left wrist height', d.blue, 'High wrist (local maximum)'),
        ('ankles', 'ankle_distance_value', 'Ankle separation', '#E4523F', 'Wide step (local maximum)'),
        ('knee', 'L_knee_value', 'Left knee angle', '#786092', 'Bent knee (local minimum)'),
    )
    wrist_color, ankle_color, knee_color = (cue[3] for cue in cues)
    d.text(1500, 30, 'Technique-Guided Key Event Localization', 60, True, align='center', max_width=2850)
    d.text(1500, 105, "Parting the Wild Horse's Mane: first left bow stance and parting hands", 28,
           color=d.muted, align='center')
    d.box(25, 158, 2950, 752, '#E4EFFF', '#5889CE', 65, 1.3, z=0)
    d.box(900, 158, 1200, 70, '#3671DB', '#2F66BF', 16, 1.1)
    d.text(1500, 174, '(a) Reference Event Localization', 43, True, 'white', 'center')
    for x, w in ((60, 520), (625, 605), (1275, 1005), (2325, 615)):
        d.box(x, 265, w, 605)

    d.text(320, 292, 'Technique Description', 36, True, align='center', max_width=475)
    d.box(246, 385, 148, 172, 'white', d.blue, 12, 2.3)
    for y in (422, 456, 490, 524):
        d.line([(270, y), (367, y)], d.blue, 2.4)
    d.text(320, 608, 'First left bow stance', 32, True, align='center')
    for y, phrase, color in (
        (664, 'Step forward;', ankle_color),
        (700, 'bend the left knee.', knee_color),
        (736, 'Part the hands:', d.ink),
        (772, 'left hand up,', wrist_color),
        (808, 'right hand down beside the hip.', d.ink),
    ):
        d.text(320, y, phrase, 29, color=color, align='center', max_width=478)

    d.text(927, 292, 'Temporal Query', 37, True, align='center')
    d.text(927, 343, 'Authored offline; reviewed', 27, align='center')
    d.box(653, 401, 549, 200, '#F0F5FC', '#CCDAED', 16, 0.8)
    d.text(678, 419, 'Metrics', 29, True)
    d.box(653, 624, 549, 200, '#F0F5FC', '#CCDAED', 16, 0.8)
    d.text(678, 642, 'Conditions', 29, True)
    for i, (_, _, label, color, condition) in enumerate(cues):
        d.text(678, 463 + i * 44, label, 29, color=color, max_width=500)
        d.text(678, 686 + i * 44, condition, 28, color=color, max_width=500)

    d.text(1777, 292, 'Reference Localization', 37, True, align='center')
    d.text(1320, 350, 'Reference SMPL sequence', 27)
    for i, item in enumerate(ref['images'][1:]):
        d.picture(output / item['image'], 1330 + i * 122, 390, 90, 120)
    d.text(1995, 350, 'Known form boundaries', 27, align='center')
    d.line([(1745, 420), (2230, 420)], d.blue, 2)
    for x in (1745, 2230):
        d.line([(x, 405), (x, 435)], d.ink, 1.4)
    boundary_x = lambda t: 1745 + (t - 16.0) / (41.4 - 16.0) * 485
    search_a, search_b = map(boundary_x, data['effective_search_seconds'])
    span(d, search_a, 403, search_b - search_a, 34, '#DBEDFF')
    d.text(1745, 443, '16.0 s', 24)
    d.text(2230, 443, '41.4 s', 24, align='right')
    d.text(1320, 532, 'Metric time series', 27, True)
    plot_left, plot_right, plot_top, plot_bottom = 1325, 2230, 613, 769
    time_min, time_max = 16.0, 32.0
    px = lambda t: plot_left + (t - time_min) / (time_max - time_min) * (plot_right - plot_left)
    search_left, search_right = data['effective_search_seconds']
    span(d, px(search_left), plot_top, px(search_right) - px(search_left),
         plot_bottom - plot_top, '#DCEEFF')
    span(d, px(event['start_time_seconds']), plot_top,
         px(event['end_time_seconds']) - px(event['start_time_seconds']),
         plot_bottom - plot_top, '#FFF0B5')
    for t in (search_left, search_right):
        d.line([(px(t), plot_top), (px(t), plot_bottom)], d.blue, 0.9, True)
    for t in (event['start_time_seconds'], event['end_time_seconds']):
        d.line([(px(t), plot_top), (px(t), plot_bottom)], d.gold, 0.9, True)
    d.line([(px(event['representative_time_seconds']), plot_top),
            (px(event['representative_time_seconds']), plot_bottom)], '#B89C18', 0.8, True)
    d.text(px(search_left), 578, 'Search window', 24, color=d.blue)
    d.text(px(event['start_time_seconds']) + 55, 578, 'Event window', 24, color='#A07900')
    with np.load(output / 'plot_data.npz') as plot:
        times = plot['times']
        visible = (times >= time_min) & (times <= time_max)
        # Display-only min-max scaling lets unlike units share one compact axis.
        for key, _, _, color, _ in cues:
            values = plot[key][visible]
            scaled = (values - values.min()) / np.ptp(values)
            d.ax.plot(px(times[visible]), plot_bottom - 8 - scaled * (plot_bottom - plot_top - 16),
                      color=color, linewidth=1.3, zorder=4)
    d.line([(plot_left, plot_top), (plot_left, plot_bottom), (plot_right, plot_bottom)], d.ink, 1.0)
    for t in (16, 20, 24, 28, 32):
        d.text(px(t), 777, str(t), 20, align='center')
    for x, (_, _, label, color, _) in zip((1330, 1630, 1942), cues):
        d.line([(x, 824), (x + 34, 824)], color, 1.6)
        d.text(x + 43, 809, label, 24, color=color)
    d.text(1777, 845, 'Time (s)', 20, color=d.muted, align='center')

    d.text(2632, 292, 'Reference Events', 37, True, align='center')
    d.text(2632, 345, 'First left bow stance', 28, align='center')
    d.picture(output / ref['image'], 2488, 395, 288, 384)
    timeline(d, 2380, 2890, 805, (2475, 2597, 2740), d.gold)
    for a, b in ((580, 625), (1230, 1275), (2280, 2325)):
        d.arrow([(a, 566), (b - 3, 566)])

    d.box(25, 960, 2950, 550, '#E5F6DF', '#469B39', 65, 1.3, z=0)
    d.box(1040, 960, 940, 70, '#11991A', '#087E10', 16, 1.1)
    d.text(1510, 976, '(b) DTW Event Transfer', 43, True, 'white', 'center')
    for x, w in ((60, 710), (815, 525), (1385, 935), (2365, 575)):
        d.box(x, 1070, w, 405, edge=d.green)
    d.arrow([(2632, 870), (2632, 932), (2090, 932), (2090, 1066)], d.gold, 3.1)

    d.text(415, 1093, 'Pose Sequences', 36, True, align='center')
    for y, subject, label, color in ((1155, ref, 'Reference', d.gold),
                                    (1313, student, 'Student', d.orange)):
        d.text(83, y + 37, label, 25)
        for i, record in enumerate(subject['images']):
            d.picture(output / record['image'], 252 + i * 115, y, 78, 104)
        d.line([(245, y + 112), (730, y + 112)], color, 2.4)
        for x in (245, 730):
            d.line([(x, y + 101), (x, y + 123)], color, 2.4)

    d.text(1077, 1093, 'SMPL Pose DTW', 36, True, align='center')
    d.text(1077, 1141, 'Geodesic pose distance', 26, align='center')
    x0, y0, w, h = 908, 1190, 360, 213
    with np.load(output / 'plot_data.npz') as plot:
        costs, row, col = plot['costs_degrees'], plot['dtw_row'], plot['dtw_col']
        d.ax.imshow(costs, extent=(x0, x0 + w, y0 + h, y0), aspect='auto', cmap='Blues',
                    interpolation='nearest', zorder=2)
        d.ax.plot(x0 + (col + .5) / costs.shape[1] * w, y0 + (row + .5) / costs.shape[0] * h,
                  color='#ED332D', linewidth=1.7, zorder=4)
    d.text(x0 + w / 2, 1419, 'Reference', 24, align='center')
    d.text(848, 1280, 'Target', 21)

    d.text(1852, 1093, 'Event Transfer', 36, True, align='center')
    d.text(1852, 1141, 'Map interval boundaries and representative frames', 27, align='center')
    reference_marks = (1690, 1840, 1980)
    target_marks = (1780, 1995, 2230)
    timeline(d, 1620, 2270, 1223, reference_marks, d.gold)
    timeline(d, 1620, 2270, 1373, target_marks, d.orange)
    d.text(1415, 1205, 'Reference', 27)
    d.text(1415, 1355, 'Target', 27)
    from matplotlib.patches import FancyArrowPatch
    for a, b in zip(reference_marks, target_marks):
        d.ax.add_patch(FancyArrowPatch((a, 1246), (b, 1350), arrowstyle='-|>',
            mutation_scale=13, linewidth=1.25, linestyle=(0, (4, 3)), color=d.ink, zorder=4))

    d.text(2652, 1093, 'Target Events', 36, True, align='center')
    d.text(2652, 1141, 'Student | first left bow stance', 26, align='center')
    d.picture(output / student['image'], 2565, 1172, 174, 232)
    timeline(d, 2408, 2895, 1415, (2510, 2635, 2770), d.orange)
    for a, b in ((770, 815), (1340, 1385), (2320, 2365)):
        d.arrow([(a, 1273), (b - 3, 1273)], d.green)
    d.text(1500, 1542, 'Localize the first bow-stance event in the reference, then transfer its temporal correspondence to the student.',
           28, align='center', max_width=2850)
    checks = d.save(destination, name)
    write_json(destination / f'{name}.audit.json', dict(
        checks=checks, pose_id=POSE_ID, reference_frame0=686, student_frame0=623,
        actual_signal_columns=[cue[1] for cue in cues],
        semantic_colors={label: color for _, _, label, color, _ in cues},
        displayed_conditions=[cue[4] for cue in cues],
        displayed_query_scope='Three plotted cues only; full frozen query remains unchanged.',
        display_signal_scaling='per-signal min-max over visible 16-32 s; detection unchanged',
        search_fill='continuous pale blue across the whole effective search window',
        transfer_arrow_targets=['target_interval_start', 'target_representative', 'target_interval_end'],
        transfer_diagram_scale='schematic; independent timelines, not to scale; documented outside figure',
        frozen_asset_root=str(output.resolve()),
        figure_data=file_record(output / 'figure_data.json'),
        plot_data=file_record(output / 'plot_data.npz'),
        generator=file_record(Path(__file__)), sources=data['sources'],
    ))
    print(f'Saved {name}; {checks["text_items"]} text items validated.', flush=True)


def export_audit(output: Path, data: dict) -> None:
    rows = []
    for m in data['case']['metrics']:
        t, s = m['teacher'], m['student']
        rows.append(dict(event_id=POSE_ID, student_id='10', metric_id=m['metric_id'],
                         unit=m['unit'], student_window_median=s['value'], student_center=s['center_value'],
                         teacher_median_2mad_lower=t['median_2mad']['lower'],
                         teacher_median_2mad_upper=t['median_2mad']['upper'],
                         teacher_p10=t['p10_p90']['lower'], teacher_p90=t['p10_p90']['upper'],
                         teacher_count=len(t['samples']),
                         median_2mad_relation=s['median_2mad_relation'], p10_p90_relation=s['p10_p90_relation'],
                         measurement_category=s['category']))
    with (output / 'first_event_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'borrowed_F014_feedback.json', dict(
        status=data['feedback_status'], applies_to='third_left_bow_stance_form_2_endpoint',
        first_event_inference=False, source=data['sources']['response'],
        findings=data['response']['findings'], coach_summary=data['response']['coach_summary'],
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'report_assets/first_bow_stance_figures_v2')
    parser.add_argument('--compose-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--kel-only', action='store_true', help='Compose only the KEL figure.')
    parser.add_argument('--figure-output-root', type=Path,
                        help='New KEL destination; requires --compose-only --kel-only.')
    args = parser.parse_args()
    if args.figure_output_root is not None and not (args.compose_only and args.kel_only):
        parser.error('--figure-output-root requires --compose-only --kel-only')
    if args.render_only and (args.compose_only or args.kel_only):
        parser.error('--render-only cannot be combined with composition-only options')
    if not args.compose_only:
        data = load_evidence()
        args.output_root.mkdir(parents=True, exist_ok=False)
        prepare_assets(args.output_root, data)
        export_audit(args.output_root, data)
    if not args.render_only:
        data = read_json(args.output_root / 'figure_data.json')
        verify_assets(args.output_root, data)
        if args.figure_output_root is not None:
            args.figure_output_root.mkdir(parents=True, exist_ok=False)
        compose_kel(args.output_root, 'KEL_first_bow_stance' + args.suffix, data,
                    destination=args.figure_output_root)
        if not args.kel_only:
            compose_coaching(args.output_root, 'Coaching_Feedback_first_bow_stance' + args.suffix, data=data)


if __name__ == '__main__':
    main()
