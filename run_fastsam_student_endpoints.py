"""Transfer the confirmed anchor endpoints to the reviewed twelve-student cohort."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from aqa3d.endpoint_transfer import manifest_boundaries
from aqa3d.fastsam_smpl import export_endpoint_smpl, load_fastsam_pose_sequence, transferred_fastsam_rows
from aqa3d.fastsam_trimmed import inspect_trimmed_video, load_trimmed_sequence, sampling_from_native_smpl
from aqa3d.smpl_dtw import backtrack_dtw_path, dtw_from_cost_matrix, inspect_video_sampling, pairwise_geodesic_costs, select_reference_frame_matches
from aqa3d.smpl_pose_segmentation import build_mapped_boundaries, build_mapped_segments
from run_fastsam_teacher_endpoints import ANCHOR, PROJECT, digest, render_review, save_path, write_json
from run_smpl_pose_dtw_segmentation import plot_diagnostics, write_dict_csv
from visualize_tas_boundary_frames import read_video_frames


EXPECTED_STUDENTS = ('01', '03', '04', '05', '06', '09', '10', '12', '13', '14', '17', '19')


def align_student(args: argparse.Namespace, entry: dict, manifest: dict, reference: object,
                  reference_sampling: object, reference_positions: np.ndarray,
                  reference_frames: dict) -> dict:
    started = time.perf_counter()
    sid = entry['student_id']
    output = args.output_root / sid
    output.mkdir()
    path = args.smpl_root / entry['filename']
    if digest(path) != entry['smpl']['sha256']:
        raise ValueError('Student NPZ differs from the extracted ZIP member.')
    sampling, pts, metadata = inspect_trimmed_video(args.video_root / (sid + '.mp4'))
    trim = json.loads((args.smpl_root / 'provenance' / sid / 'trim_manifest.json').read_text())
    sequence = load_trimmed_sequence(path, sampling, pts, trim)
    gap = 1 if sid in args.allow_nearest_case else 0
    native_arrays = {}
    if sid in args.native_sampling_case:
        with np.load(path, allow_pickle=False) as data:
            native_indices = data['source_sample_indices'].copy()
            native_fps = float(data['fps'])
        sampling = sampling_from_native_smpl(sampling, native_indices, sequence.source_indices, native_fps)
        stride = int(round(native_fps / sampling.sample_fps))
        metadata.update(sample_policy='native SMPL clock decimation; exact stored output-video mapping',
                        sample_count=sampling.sample_count, native_sample_fps=native_fps,
                        native_sample_stride=stride, native_first_sample_index_0based=int(native_indices[0]),
                        native_sampling_complete=True,
                        video_frames_without_smpl_interpretation='normal native reconstruction sampling; not lost poses')
        native_arrays = {'all_smpl_native_sample_indices': native_indices,
                         'sample_native_indices_0based': native_indices[::stride],
                         'native_sample_fps': np.asarray(native_fps)}
    positions = sequence.sample_positions(sampling, max_gap_frames=gap)
    actual = sequence.source_indices[positions]
    if np.any(np.diff(actual) <= 0):
        raise ValueError('Frame substitution duplicates a sampled source frame.')
    effective = replace(sampling, source_indices=actual)
    substitutions = [{'sample_index_0based': int(i), 'grid_time_seconds': i / 5.,
                      'requested_output_frame_0based': int(sampling.source_indices[i]),
                      'actual_output_frame_0based': int(actual[i]),
                      'frame_offset': int(actual[i] - sampling.source_indices[i])}
                     for i in np.flatnonzero(actual != sampling.source_indices)]
    write_json(output / 'sampling_audit.json', dict(sequence.audit, **metadata,
               explicit_nearest_max_gap_frames=gap, substitutions=substitutions))
    np.savez_compressed(output / 'video_timeline.npz', video_pts_seconds=pts,
                        requested_sample_source_indices=sampling.source_indices,
                        actual_sample_source_indices=actual,
                        sample_npz_rows_0based=positions,
                        all_smpl_output_video_frame_indices=sequence.source_indices, **native_arrays)
    gold = manifest_boundaries(manifest, reference_sampling)
    print(f'[{sid}] geodesic costs {sampling.sample_count} x {reference_sampling.sample_count}; substitutions={len(substitutions)}', flush=True)
    costs = pairwise_geodesic_costs(sequence.body_poses[positions], reference.body_poses[reference_positions], chunk_size=args.pairwise_chunk_size)
    distance, _, accumulated = dtw_from_cost_matrix(costs)
    path_indices = backtrack_dtw_path(accumulated)
    del accumulated
    save_path(output, path_indices, costs, reference_sampling, effective, reference_positions, positions, target_video_pts=pts)
    matches = select_reference_frame_matches(costs, path_indices, [b.sample_index_0based for b in gold])
    write_json(output / 'selected_matches.json', [asdict(match) for match in matches])
    boundaries = build_mapped_boundaries(sid, gold, matches, reference_sampling, effective)
    rows = transferred_fastsam_rows(manifest, boundaries, path_indices, reference_sampling, effective,
                                   reference, sequence, reference_positions, positions, target_role='student')
    for row in rows:
        sample = row['student_sample_index_0based']
        requested = int(sampling.source_indices[sample])
        offset = row['student_source_frame_0based'] - requested
        row.update(student_requested_source_frame_0based=requested,
                   student_requested_video_pts_seconds=float(pts[requested]), student_frame_offset=offset)
        if offset:
            row['student_tracking_match'] = 'explicit_nearest_substitution'
            row['diagnostic_flags'] += ';explicit_nearest_substitution'
    selected = np.asarray([r['student_npz_row_0based'] for r in rows])
    export_endpoint_smpl(path, output / 'endpoint_smpl.npz', selected, list(range(1, 25)), entry['smpl']['sha256'])
    frames = read_video_frames(sampling.video_path, [r['student_source_frame_0based'] for r in rows])
    (output / 'keypose_frames').mkdir()
    for row in rows:
        name = f'keypose_frames/{row["move_id"]:02d}_student.jpg'
        frames[row['student_source_frame_0based']].save(output / name, quality=95)
        row['student_image'] = name
        row['reference_image'] = f'../reference/keypose_frames/{row["move_id"]:02d}_reference.jpg'
    write_json(output / 'endpoint_keyposes.json', rows)
    write_dict_csv(output / 'endpoint_keyposes.csv', rows)
    write_dict_csv(output / 'boundaries.csv', [{k: v for k, v in asdict(b).items() if k != 'target_phalp_frame_1based'} for b in boundaries])
    segments = build_mapped_segments(sid, boundaries, effective)
    segment_rows = []
    for segment in segments:
        record = asdict(segment)
        start, end = segment.start_frame - 1, segment.end_frame - 1
        record['source_interval_start_time_seconds'] = float(pts[start])
        record['source_interval_end_time_exclusive_seconds'] = float(pts[end + 1]) if end + 1 < len(pts) else metadata['duration_seconds']
        segment_rows.append(record)
    write_dict_csv(output / 'segments.csv', segment_rows)
    plot_diagnostics(output / 'dtw_diagnostics.png', sid, costs, path_indices, gold, boundaries, reference_sampling, effective)
    render_review(output, rows, reference_frames, frames, target_role='student')
    result = {'status': 'ok', 'student_id': sid, 'video_id': sid, 'endpoint_count': 24,
              'sample_fps': 5., 'source_frame_count': sampling.source_frame_count,
              'source_fps': sampling.source_fps, 'duration_seconds': metadata['duration_seconds'],
              'timestamp_basis': 'actual trimmed video PTS', 'sample_count': sampling.sample_count,
              'sampling_policy': metadata['sample_policy'],
              'strictly_increasing': True, 'manual_review_status': 'pending',
              'explicit_substituted_sample_count': len(substitutions),
              'dtw_distance_radians': distance, 'dtw_path_length': len(path_indices),
              'elapsed_seconds': time.perf_counter() - started}
    write_json(output / 'summary.json', result)
    print(f'[{sid}] saved 24 ordered student endpoints ({result["elapsed_seconds"]:.1f}s)', flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smpl-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/FastSAM_SMPL/student_single_person_12'))
    parser.add_argument('--video-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/student_trimmed'))
    parser.add_argument('--reference-smpl', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/FastSAM_SMPL/teacher_single_person_11/BV1WE411W7JB.npz'))
    parser.add_argument('--reference-video', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4'))
    parser.add_argument('--reference-manifest', type=Path, default=PROJECT / 'reference_data/BV1WE411W7JB/reference_manifest.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--allow-nearest-case', action='append', default=[], choices=EXPECTED_STUDENTS,
                        help='Explicitly authorize a maximum one-frame gap for this named student only.')
    parser.add_argument('--student-id', action='append', choices=EXPECTED_STUDENTS,
                        help='Optional explicit subset for a separate diagnostic run.')
    parser.add_argument('--native-sampling-case', action='append', default=[], choices=EXPECTED_STUDENTS,
                        help='Decimate this student native SMPL sampling clock to 5 FPS using exact stored frame mappings.')
    parser.add_argument('--pairwise-chunk-size', type=int, default=32)
    args = parser.parse_args()
    if args.pairwise_chunk_size <= 0:
        parser.error('Chunk size must be positive.')
    if set(args.native_sampling_case) & set(args.allow_nearest_case):
        parser.error('Native sampling must not be combined with nearest-frame substitution for the same student.')
    args.output_root = args.output_root.expanduser().absolute()
    args.output_root.mkdir(parents=True, exist_ok=False)
    extraction = json.loads((args.smpl_root / 'manifest.json').read_text())
    entries = extraction['students']
    if tuple(e['student_id'] for e in entries) != EXPECTED_STUDENTS:
        raise ValueError('The student cohort differs from the reviewed twelve-case ZIP.')
    if args.student_id:
        entries = [e for e in entries if e['student_id'] in args.student_id]
    manifest = json.loads(args.reference_manifest.read_text())
    reference_sampling = inspect_video_sampling(args.reference_video, 5.)
    gold = manifest_boundaries(manifest, reference_sampling)
    reference = load_fastsam_pose_sequence(args.reference_smpl)
    reference_positions = reference.sample_positions(reference_sampling)
    provenance = args.output_root / 'provenance'
    provenance.mkdir()
    shutil.copy2(args.reference_manifest, provenance / 'reference_manifest_snapshot.json')
    shutil.copy2(args.smpl_root / 'manifest.json', provenance / 'extraction_manifest.json')
    code = ['run_fastsam_student_endpoints.py', 'run_fastsam_teacher_endpoints.py', 'aqa3d/fastsam_smpl.py',
            'aqa3d/fastsam_trimmed.py', 'aqa3d/smpl_dtw.py', 'aqa3d/smpl_pose_segmentation.py']
    hashes = {}
    for name in code:
        dest = provenance / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
        hashes[name] = digest(PROJECT / name)
    write_json(provenance / 'code_sha256.json', hashes)
    ref_sources = [e['source_frame_index_0based'] for e in manifest['endpoint_keyposes']]
    ref_frames = read_video_frames(args.reference_video, ref_sources)
    ref_output = args.output_root / 'reference'
    (ref_output / 'keypose_frames').mkdir(parents=True)
    for event in manifest['endpoint_keyposes']:
        ref_frames[event['source_frame_index_0based']].save(ref_output / 'keypose_frames' / f'{event["move_id"]:02d}_reference.jpg', quality=95)
    export_endpoint_smpl(args.reference_smpl, ref_output / 'endpoint_smpl.npz',
                         np.searchsorted(reference.source_indices, ref_sources), list(range(1, 25)), digest(args.reference_smpl))
    media = [{'student_id': e['student_id'], 'video_path': str(args.video_root / (e['student_id'] + '.mp4')),
              'smpl_path': str(args.smpl_root / e['filename']), 'video_sha256': e['video']['sha256'],
              'smpl_sha256': e['smpl']['sha256']} for e in entries]
    write_json(args.output_root / 'student_cohort.json', media)
    summary = {'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'status': 'running',
               'reference_video_id': ANCHOR, 'reference_video_path': str(args.reference_video),
               'reference_smpl_sha256': digest(args.reference_smpl), 'expected_student_count': len(entries),
               'allow_nearest_cases': args.allow_nearest_case,
               'native_sampling_cases': args.native_sampling_case,
               'sampling_policy': '5 FPS; named native cases use native SMPL clock, others actual video PTS; no interpolated poses',
               'algorithm': 'unchanged mean 23-local-joint SO(3) geodesic full-sequence DTW; global_orient excluded',
               'reference_intervals_rankings_feedback_computed': False, 'results': []}
    for entry in entries:
        try:
            result = align_student(args, entry, manifest, reference, reference_sampling, reference_positions, ref_frames)
        except Exception as exc:
            result = {'status': 'failed', 'student_id': entry['student_id'], 'error_type': type(exc).__name__, 'error': str(exc)}
            folder = args.output_root / entry['student_id']
            folder.mkdir(exist_ok=True)
            write_json(folder / 'summary.json', result)
            print(f'[{entry["student_id"]}] FAILED: {exc}', flush=True)
        summary['results'].append(result)
        write_json(args.output_root / 'batch_summary.json', summary)
    summary.update(successful_students=sum(r['status'] == 'ok' for r in summary['results']),
                   failed_students=sum(r['status'] != 'ok' for r in summary['results']),
                   transferred_endpoint_count=sum(r.get('endpoint_count', 0) for r in summary['results']))
    summary['status'] = 'complete_pending_manual_review' if summary['successful_students'] == len(entries) else 'incomplete'
    write_json(args.output_root / 'batch_summary.json', summary)
    combined = []
    lines = ['# FastSAM 学生结束定势迁移', '', '自动迁移结果均待人工复核。', '', '| 学生 | 状态 | 定势 |', '| --- | --- | --- |']
    for result in summary['results']:
        sid = result['student_id']
        if result['status'] == 'ok':
            combined.extend(json.loads((args.output_root / sid / 'endpoint_keyposes.json').read_text()))
            link = f'[配对复核图]({args.output_root}/{sid}/review.md)'
        else:
            link = f'[失败记录]({args.output_root}/{sid}/summary.json)'
        lines.append(f'| {sid} | {result["status"]} | {link} |')
    if combined:
        write_json(args.output_root / 'all_student_endpoints.json', combined)
        write_dict_csv(args.output_root / 'all_student_endpoints.csv', combined)
    (args.output_root / 'index.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary['failed_students']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
