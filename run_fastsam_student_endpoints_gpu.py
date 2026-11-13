#!/usr/bin/env python
"""Independent maximum-common-FPS GPU transfer for the twelve trimmed students."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from aqa3d.endpoint_transfer import manifest_boundaries
from aqa3d.fastsam_smpl import export_endpoint_smpl, load_fastsam_pose_sequence
from aqa3d.fastsam_student_high_fps import maximum_native_student_pair
from aqa3d.fastsam_trimmed import inspect_trimmed_video, load_trimmed_sequence
from aqa3d.smpl_dtw import backtrack_dtw_path, inspect_video_sampling, require_strictly_increasing_boundaries, select_reference_frame_matches
from aqa3d.smpl_dtw_gpu import cuda_device, dtw_from_cost_matrix_gpu, pairwise_geodesic_costs_gpu
from run_fastsam_student_endpoints import EXPECTED_STUDENTS
from run_fastsam_teacher_endpoints import ANCHOR, PROJECT, digest, render_review, write_json
from run_smpl_pose_dtw_segmentation import write_dict_csv
from visualize_tas_boundary_frames import read_video_frames


def bind_student_rows(manifest, matches, path, reference, student, ref_sampling, student_sampling,
                       ref_positions, student_positions, sid):
    require_strictly_increasing_boundaries(matches)
    if len(matches) != 24:
        raise ValueError('Expected 24 transferred endpoints.')
    rows = []
    for event, match in zip(manifest['endpoint_keyposes'], matches):
        row = {k: event[k] for k in ('move_id', 'move_name_pinyin', 'move_name_zh', 'pose_id',
                                    'keypose_role', 'final_technique_step', 'technique_binding_status')}
        row.update(video_id=sid, student_id=sid, reference_video_id=ANCHOR,
                   boundary_annotation_status='predicted', manual_review_status='pending', review_required=True,
                   boundary_policy='minimum_local_geodesic_among_dtw_path_candidates',
                   candidate_count=match.candidate_count, local_geodesic_radians=match.local_cost,
                   local_geodesic_degrees=float(np.degrees(match.local_cost)), dtw_sample_fps=student_sampling.sample_fps,
                   review_fps=5., automatic_endpoint_rounding_to_review_grid=False,
                   reference_annotation_time_seconds=event['boundary_time_seconds'])
        flags = ['multiple_dtw_candidates'] if match.candidate_count > 1 else []
        for role, sample, sequence, sampling, positions in (
                ('reference', match.reference_index, reference, ref_sampling, ref_positions),
                ('student', match.target_index, student, student_sampling, student_positions)):
            provenance = sequence.provenance(int(positions[sample]), sampling)
            if provenance['source_frame_0based'] != int(sampling.source_indices[sample]):
                raise ValueError('An endpoint does not address its exact source SMPL row.')
            row[role + '_sample_index_0based'] = int(sample)
            row.update({role + '_' + k: v for k, v in provenance.items()})
            if provenance['near_source_track_switch']:
                flags.append(role + '_source_track_switch')
        if row['reference_source_frame_0based'] != event['source_frame_index_0based']:
            raise ValueError('Reference annotation frame changed.')
        candidates = path[path[:, 1] == match.reference_index, 0]
        row.update(student_candidate_start_source_time_seconds=float(student.video_timestamps[student_sampling.source_indices[candidates.min()]]),
                   student_candidate_end_source_time_seconds=float(student.video_timestamps[student_sampling.source_indices[candidates.max()]]),
                   diagnostic_flags=';'.join(flags))
        rows.append(row)
    return rows


def export_segments(output, rows, sampling, pts, duration):
    previous_frame, previous_sample = -1, -1
    segments = []
    for row in rows:
        frame, sample = row['student_source_frame_0based'], row['student_sample_index_0based']
        start = previous_frame + 1
        segments.append({'video_id': row['video_id'], 'move_id': row['move_id'], 'move_name': row['move_name_zh'],
                         'start_frame_0based': start, 'end_frame_0based_inclusive': frame,
                         'start_sample_index_0based': previous_sample + 1, 'end_sample_index_0based_inclusive': sample,
                         'source_interval_start_time_seconds': float(pts[start]),
                         'source_interval_end_time_exclusive_seconds': float(pts[frame + 1]) if frame + 1 < len(pts) else duration,
                         'source_fps': sampling.source_fps, 'sample_fps': sampling.sample_fps,
                         'timestamp_basis': 'trimmed_video_PTS'})
        previous_frame, previous_sample = frame, sample
    write_dict_csv(output / 'segments.csv', segments)


def plot_student_path(output, costs, path, ref_sampling, student_sampling, pts, rows):
    rs, cs = max(int(np.ceil(len(costs) / 700)), 1), max(int(np.ceil(costs.shape[1] / 700)), 1)
    ref_times = ref_sampling.source_indices[path[:, 1]] / ref_sampling.source_fps
    student_times = pts[student_sampling.source_indices[path[:, 0]]]
    figure, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.imshow(np.degrees(costs[::rs, ::cs]), origin='lower', aspect='auto', cmap='viridis',
              extent=(0, ref_sampling.source_indices[-1] / ref_sampling.source_fps, 0,
                      float(pts[student_sampling.source_indices[-1]])))
    ax.plot(ref_times, student_times, color='white', linewidth=1.)
    ax.scatter([r['reference_source_time_seconds'] for r in rows], [r['student_source_time_seconds'] for r in rows],
                s=20, color='#e74c3c')
    ax.set(xlabel='Reference source time (s)', ylabel='Student actual video PTS (s)',
           title=f'Maximum common {student_sampling.sample_fps:.6g} FPS GPU Geodesic DTW')
    figure.savefig(output / 'dtw_diagnostics.png', dpi=130)
    plt.close(figure)


def load_student(args, entry):
    sid = entry['student_id']
    sampling, pts, metadata = inspect_trimmed_video(args.video_root / (sid + '.mp4'))
    trim = json.loads((args.smpl_root / 'provenance' / sid / 'trim_manifest.json').read_text())
    sequence = load_trimmed_sequence(args.smpl_root / entry['filename'], sampling, pts, trim)
    with np.load(sequence.source_path, allow_pickle=False) as data:
        native, native_fps = data['source_sample_indices'].copy(), float(data['fps'])
    return sampling, pts, metadata, sequence, native, native_fps


def align_student_gpu(args, entry, manifest, reference, ref_base, ref_native_fps, ref_frames):
    started = time.perf_counter()
    sid = entry['student_id']
    output = args.output_root / sid
    output.mkdir()
    if digest(args.smpl_root / entry['filename']) != entry['smpl']['sha256']:
        raise ValueError('Student source SMPL changed after extraction.')
    base, pts, metadata, student, native, native_fps = load_student(args, entry)
    required = np.asarray([e['source_frame_index_0based'] for e in manifest['endpoint_keyposes']])
    ref_sampling, sampling, positions, policy = maximum_native_student_pair(
        ref_base, base, ref_native_fps, native_fps, native, student.source_indices, required)
    ref_positions = reference.sample_positions(ref_sampling)
    np.testing.assert_array_equal(student.sample_positions(sampling), positions)
    indices = np.searchsorted(ref_sampling.source_indices, required)
    np.testing.assert_array_equal(ref_sampling.source_indices[indices], required)
    write_json(output / 'sampling_audit.json', dict(student.audit, **policy,
               native_sample_fps=native_fps, trimmed_video_duration_seconds=metadata['duration_seconds'],
               explicit_nearest_substituted_sample_count=0))
    np.savez_compressed(output / 'video_timeline.npz', video_pts_seconds=pts,
                        all_smpl_output_video_frame_indices=student.source_indices,
                        actual_sample_source_indices=sampling.source_indices, sample_npz_rows_0based=positions,
                        all_smpl_native_sample_indices=native, sample_native_indices_0based=native[positions],
                        native_sample_fps=np.asarray(native_fps), pair_sample_fps=np.asarray(sampling.sample_fps),
                        reference_source_frame_indices_0based=ref_sampling.source_indices,
                        reference_npz_rows_0based=ref_positions, reference_endpoint_sample_indices_0based=indices)
    load_seconds = time.perf_counter() - started
    print(f'[{sid}] {sampling.sample_fps:.9g} FPS; {len(positions)} x {len(ref_positions)} costs; exact native/video mapping', flush=True)
    cost_profile, dtw_profile = {}, {}
    phase = time.perf_counter()
    costs = pairwise_geodesic_costs_gpu(student.body_poses[positions], reference.body_poses[ref_positions],
                                       device=args.device, target_chunk_size=args.target_chunk_size,
                                       reference_chunk_size=args.reference_chunk_size, profile=cost_profile)
    cost_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    distance, _, accumulated = dtw_from_cost_matrix_gpu(costs, device=args.device, profile=dtw_profile)
    path = backtrack_dtw_path(accumulated)
    del accumulated
    dtw_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    matches = select_reference_frame_matches(costs, path, indices)
    rows = bind_student_rows(manifest, matches, path, reference, student, ref_sampling, sampling,
                             ref_positions, positions, sid)
    ti, ri = path[:, 0], path[:, 1]
    np.savez_compressed(output / 'dtw_path.npz', target_sample_indices_0based=ti, reference_sample_indices_0based=ri,
                        target_source_frame_indices_0based=sampling.source_indices[ti],
                        reference_source_frame_indices_0based=ref_sampling.source_indices[ri],
                        target_npz_rows_0based=positions[ti], reference_npz_rows_0based=ref_positions[ri],
                        target_original_source_frame_indices_0based=student.original_source_indices[positions[ti]],
                        target_source_times_seconds=pts[sampling.source_indices[ti]],
                        target_original_source_times_seconds=student.original_source_timestamps[positions[ti]],
                        reference_source_times_seconds=ref_sampling.source_indices[ri] / ref_sampling.source_fps,
                        local_geodesic_radians=costs[ti, ri], pair_sample_fps=np.asarray(sampling.sample_fps))
    write_json(output / 'selected_matches.json', [dict(reference_index=m.reference_index, target_index=m.target_index,
                candidate_count=m.candidate_count, local_cost=m.local_cost) for m in matches])
    selected = np.asarray([r['student_npz_row_0based'] for r in rows])
    export_endpoint_smpl(student.source_path, output / 'endpoint_smpl.npz', selected, list(range(1, 25)), entry['smpl']['sha256'])
    frames = read_video_frames(sampling.video_path, [r['student_source_frame_0based'] for r in rows])
    (output / 'keypose_frames').mkdir()
    for row in rows:
        image = f'keypose_frames/{row["move_id"]:02d}_student.jpg'
        frames[row['student_source_frame_0based']].save(output / image, quality=95)
        row.update(student_image=image, reference_image=f'../reference/keypose_frames/{row["move_id"]:02d}_reference.jpg')
    write_json(output / 'endpoint_keyposes.json', rows)
    write_dict_csv(output / 'endpoint_keyposes.csv', rows)
    export_segments(output, rows, sampling, pts, metadata['duration_seconds'])
    render_review(output, rows, ref_frames, frames, target_role='student')
    plot_student_path(output, costs, path, ref_sampling, sampling, pts, rows)
    result = {'status': 'ok', 'student_id': sid, 'video_id': sid, 'endpoint_count': 24,
              'sample_fps': sampling.sample_fps, 'sample_count': len(positions), 'reference_sample_count': len(ref_positions),
              'source_fps': sampling.source_fps, 'source_frame_count': len(pts), 'native_smpl_fps': native_fps,
              'timestamp_basis': 'actual trimmed video PTS', 'strictly_increasing': True, 'manual_review_status': 'pending',
              'review_fps': 5., 'automatic_endpoint_rounding_to_review_grid': False, 'explicit_substituted_sample_count': 0,
              'dtw_distance_radians': distance, 'dtw_path_length': len(path), 'dtype': 'float64',
              'gpu_cost_profile': cost_profile, 'gpu_dtw_profile': dtw_profile,
              'load_and_sampling_seconds': load_seconds, 'geodesic_cost_seconds': cost_seconds, 'dtw_seconds': dtw_seconds,
              'selection_and_export_seconds': time.perf_counter() - phase, 'elapsed_seconds': time.perf_counter() - started}
    write_json(output / 'summary.json', result)
    print(f'[{sid}] saved 24 endpoints; costs {cost_seconds:.3f}s; DTW {dtw_seconds:.3f}s; total {result["elapsed_seconds"]:.3f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smpl-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/FastSAM_SMPL/student_single_person_12'))
    parser.add_argument('--video-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/student_trimmed'))
    parser.add_argument('--reference-smpl', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/FastSAM_SMPL/teacher_single_person_11/BV1WE411W7JB.npz'))
    parser.add_argument('--reference-video', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4'))
    parser.add_argument('--reference-manifest', type=Path, default=PROJECT / 'reference_data/BV1WE411W7JB/reference_manifest.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-chunk-size', type=int, default=256)
    parser.add_argument('--reference-chunk-size', type=int, default=1024)
    parser.add_argument('--student-id', action='append', choices=EXPECTED_STUDENTS)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if min(args.target_chunk_size, args.reference_chunk_size) <= 0:
        parser.error('CUDA chunk sizes must be positive.')
    started = time.perf_counter()
    args.output_root = args.output_root.expanduser().absolute()
    args.output_root.mkdir(parents=True, exist_ok=False)
    extraction = json.loads((args.smpl_root / 'manifest.json').read_text())
    if tuple(e['student_id'] for e in extraction['students']) != EXPECTED_STUDENTS:
        raise ValueError('The source cohort is not the reviewed twelve-student cohort.')
    entries = [e for e in extraction['students'] if not args.student_id or e['student_id'] in args.student_id]
    manifest = json.loads(args.reference_manifest.read_text())
    ref_base = inspect_video_sampling(args.reference_video, 5.)
    manifest_boundaries(manifest, ref_base)
    reference = load_fastsam_pose_sequence(args.reference_smpl)
    with np.load(args.reference_smpl, allow_pickle=False) as data:
        ref_native_fps = float(data['fps'])
    hashes = {str(p): digest(p) for p in (args.reference_smpl, args.reference_video, args.reference_manifest,
                                          args.smpl_root / 'manifest.json')}
    audits = []
    required = np.asarray([e['source_frame_index_0based'] for e in manifest['endpoint_keyposes']])
    for entry in entries:
        sid = entry['student_id']
        for path, expected in ((args.smpl_root / entry['filename'], entry['smpl']['sha256']),
                               (args.video_root / (sid + '.mp4'), entry['video']['sha256'])):
            hashes[str(path)] = digest(path)
            if hashes[str(path)] != expected:
                raise ValueError('Source extraction checksum mismatch: ' + str(path))
        base, pts, metadata, sequence, native, native_fps = load_student(args, entry)
        _, sampling, positions, policy = maximum_native_student_pair(ref_base, base, ref_native_fps, native_fps,
                                                                    native, sequence.source_indices, required)
        np.testing.assert_array_equal(sequence.sample_positions(sampling), positions)
        audits.append(dict(sequence.audit, **policy, student_id=sid, native_smpl_fps=native_fps,
                           video_path=str(base.video_path), smpl_path=str(sequence.source_path)))
        print(f'Preflight {sid}: {native_fps:.9g} native FPS; {len(positions)} exact mapped samples', flush=True)
    write_json(args.output_root / 'preflight.json', {'status': 'passed', 'students': audits, 'source_files_sha256': hashes})
    provenance = args.output_root / 'provenance'
    provenance.mkdir()
    shutil.copy2(args.reference_manifest, provenance / 'reference_manifest_snapshot.json')
    shutil.copy2(args.smpl_root / 'manifest.json', provenance / 'extraction_manifest.json')
    code = ('run_fastsam_student_endpoints_gpu.py', 'aqa3d/fastsam_student_high_fps.py', 'aqa3d/fastsam_high_fps.py',
            'aqa3d/fastsam_trimmed.py', 'aqa3d/fastsam_smpl.py', 'aqa3d/smpl_dtw_gpu.py', 'aqa3d/smpl_dtw.py',
            'run_fastsam_student_endpoints.py', 'run_fastsam_teacher_endpoints.py',
            'run_smpl_pose_dtw_segmentation.py', 'visualize_tas_boundary_frames.py')
    code_hashes = {}
    for name in code:
        destination = provenance / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, destination)
        code_hashes[name] = digest(PROJECT / name)
    write_json(provenance / 'code_sha256.json', code_hashes)
    media = [{'student_id': e['student_id'], 'video_path': str(args.video_root / (e['student_id'] + '.mp4')),
              'smpl_path': str(args.smpl_root / e['filename']), 'video_sha256': e['video']['sha256'],
              'smpl_sha256': e['smpl']['sha256']} for e in entries]
    write_json(args.output_root / 'student_cohort.json', media)
    if args.preflight_only:
        print('Preflight only; no GPU alignment run.', flush=True)
        return
    args.device = str(cuda_device(args.device))
    phase = time.perf_counter()
    ref_frames = read_video_frames(args.reference_video, required)
    ref_output = args.output_root / 'reference'
    (ref_output / 'keypose_frames').mkdir(parents=True)
    for event in manifest['endpoint_keyposes']:
        ref_frames[event['source_frame_index_0based']].save(ref_output / 'keypose_frames' / f'{event["move_id"]:02d}_reference.jpg', quality=95)
    export_endpoint_smpl(args.reference_smpl, ref_output / 'endpoint_smpl.npz',
                         np.searchsorted(reference.source_indices, required), list(range(1, 25)), hashes[str(args.reference_smpl)])
    warm = pairwise_geodesic_costs_gpu(reference.body_poses[:4], reference.body_poses[:4], device=args.device)
    dtw_from_cost_matrix_gpu(warm, device=args.device)
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'reference_video_id': ANCHOR,
               'reference_video_path': str(args.reference_video), 'expected_student_count': len(entries),
               'algorithm': 'exact dense GPU Geodesic DTW; mean over 23 local SO(3) joints; global orientation excluded',
               'sampling_policy': 'maximum common nominal FPS on native student sample clock with exact cropped-video mapping',
               'dtype': 'float64', 'device': args.device, 'GPU': torch.cuda.get_device_name(args.device),
               'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
               'reference_setup_and_warmup_seconds': time.perf_counter() - phase,
               'rankings_reference_intervals_feedback_computed': False, 'results': []}
    for entry in entries:
        try:
            result = align_student_gpu(args, entry, manifest, reference, ref_base, ref_native_fps, ref_frames)
        except Exception as exc:
            result = {'status': 'failed', 'student_id': entry['student_id'], 'error_type': type(exc).__name__, 'error': str(exc)}
            folder = args.output_root / entry['student_id']
            folder.mkdir(exist_ok=True)
            write_json(folder / 'summary.json', result)
            print(f'[{entry["student_id"]}] FAILED: {exc}', flush=True)
        summary['results'].append(result)
        write_json(args.output_root / 'batch_summary.json', summary)
    changed = [name for name, sha in hashes.items() if digest(Path(name)) != sha]
    good = [r for r in summary['results'] if r['status'] == 'ok']
    summary.update(status='complete_pending_manual_review' if len(good) == len(entries) and not changed else 'incomplete',
                   successful_students=len(good), failed_students=len(entries) - len(good), transferred_endpoint_count=len(good) * 24,
                   source_files_unchanged=not changed, changed_source_files=changed,
                   summed_geodesic_cost_seconds=sum(r['geodesic_cost_seconds'] for r in good),
                   summed_dtw_seconds=sum(r['dtw_seconds'] for r in good),
                   summed_pair_elapsed_seconds=sum(r['elapsed_seconds'] for r in good),
                   batch_elapsed_seconds=time.perf_counter() - started)
    write_json(args.output_root / 'batch_summary.json', summary)
    if good:
        fields = ('student_id', 'sample_fps', 'sample_count', 'reference_sample_count', 'geodesic_cost_seconds', 'dtw_seconds', 'elapsed_seconds')
        write_dict_csv(args.output_root / 'timing_summary.csv', [{k: r[k] for k in fields} for r in good])
        combined = [row for r in good for row in json.loads((args.output_root / r['student_id'] / 'endpoint_keyposes.json').read_text())]
        write_json(args.output_root / 'all_student_endpoints.json', combined)
        write_dict_csv(args.output_root / 'all_student_endpoints.csv', combined)
    lines = ['# Maximum-FPS GPU Student Endpoint Transfer', '', 'Exact native SMPL rows and actual trimmed-video PTS; no pose interpolation or neighbor replacement.', '',
             '| Student | FPS | Geodesic (s) | DTW (s) | Total (s) |', '| --- | --- | --- | --- | --- |']
    for r in good:
        lines.append(f'| [{r["student_id"]}]({args.output_root}/{r["student_id"]}/review.md) | {r["sample_fps"]:.6f} | '
                     f'{r["geodesic_cost_seconds"]:.3f} | {r["dtw_seconds"]:.3f} | {r["elapsed_seconds"]:.3f} |')
    lines.extend(['', 'All automatic endpoints remain pending human review; old results and annotations are unchanged.',
                  f'Batch wall time: {summary["batch_elapsed_seconds"]:.3f} s.', ''])
    (args.output_root / 'index.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, ensure_ascii=False, indent=2), flush=True)
    if summary['status'] == 'incomplete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
