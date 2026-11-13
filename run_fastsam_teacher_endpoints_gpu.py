#!/usr/bin/env python
"""Separate maximum-common-FPS CUDA teacher endpoint transfer and timing report."""

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

from aqa3d.fastsam_high_fps import maximum_fps_pair
from aqa3d.fastsam_smpl import export_endpoint_smpl, load_fastsam_pose_sequence
from aqa3d.smpl_dtw import backtrack_dtw_path, dtw_from_cost_matrix, inspect_video_sampling, require_strictly_increasing_boundaries, select_reference_frame_matches
from aqa3d.smpl_dtw_gpu import cuda_device, dtw_from_cost_matrix_gpu, pairwise_geodesic_costs_gpu
from run_fastsam_teacher_endpoints import ANCHOR, PROJECT, digest, preflight, render_review, write_json
from run_smpl_pose_dtw_segmentation import write_dict_csv
from visualize_tas_boundary_frames import read_video_frames


def bind_endpoint_rows(manifest, matches, path, reference, target, ref_sampling, target_sampling,
                       ref_positions, target_positions, video_id):
    require_strictly_increasing_boundaries(matches)
    if len(matches) != 24:
        raise ValueError('Expected exactly 24 endpoint matches.')
    rows = []
    for event, match in zip(manifest['endpoint_keyposes'], matches):
        row = {key: event[key] for key in ('move_id', 'move_name_pinyin', 'move_name_zh', 'pose_id',
               'keypose_role', 'final_technique_step', 'technique_binding_status')}
        row.update(video_id=video_id, reference_video_id=ANCHOR,
                   boundary_annotation_status='predicted', manual_review_status='pending', review_required=True,
                   boundary_policy='minimum_local_geodesic_among_dtw_path_candidates',
                   candidate_count=match.candidate_count, local_geodesic_radians=match.local_cost,
                   local_geodesic_degrees=float(np.degrees(match.local_cost)),
                   dtw_sample_fps=target_sampling.sample_fps, review_fps=5.,
                   automatic_endpoint_rounding_to_review_grid=False,
                   reference_annotation_time_seconds=event['boundary_time_seconds'])
        flags = ['multiple_dtw_candidates'] if match.candidate_count > 1 else []
        for role, sample, sequence, sampling, positions in (
                ('reference', match.reference_index, reference, ref_sampling, ref_positions),
                ('teacher', match.target_index, target, target_sampling, target_positions)):
            provenance = sequence.provenance(int(positions[sample]), sampling)
            if provenance['source_frame_0based'] != int(sampling.source_indices[sample]):
                raise ValueError('Selected endpoint is not an exact SMPL source frame.')
            row[role + '_sample_index_0based'] = int(sample)
            row.update({role + '_' + key: value for key, value in provenance.items()})
            if provenance['near_source_track_switch']:
                flags.append(role + '_source_track_switch')
        if row['reference_source_frame_0based'] != event['source_frame_index_0based']:
            raise ValueError('An immutable reference annotation was shifted.')
        candidates = path[path[:, 1] == match.reference_index, 0]
        row.update(teacher_candidate_start_source_time_seconds=float(target_sampling.source_indices[candidates.min()] / target_sampling.source_fps),
                   teacher_candidate_end_source_time_seconds=float(target_sampling.source_indices[candidates.max()] / target_sampling.source_fps),
                   diagnostic_flags=';'.join(flags))
        rows.append(row)
    return rows


def save_segments(output, rows, target_sampling):
    previous_source, previous_sample = -1, -1
    segments = []
    for row in rows:
        end = row['teacher_source_frame_0based']
        sample = row['teacher_sample_index_0based']
        segments.append({'video_id': row['video_id'], 'move_id': row['move_id'],
                         'move_name': row['move_name_zh'], 'start_frame_0based': previous_source + 1,
                         'end_frame_0based_inclusive': end, 'start_sample_index_0based': previous_sample + 1,
                         'end_sample_index_0based_inclusive': sample,
                         'source_interval_start_time_seconds': (previous_source + 1) / target_sampling.source_fps,
                         'source_interval_end_time_exclusive_seconds': (end + 1) / target_sampling.source_fps,
                         'source_fps': target_sampling.source_fps, 'sample_fps': target_sampling.sample_fps})
        previous_source, previous_sample = end, sample
    write_dict_csv(output / 'segments.csv', segments)


def plot_path(output, costs, path, ref_sampling, target_sampling, rows):
    # Only the display heatmap is decimated. Alignment always uses the full dense matrix.
    rs = max(int(np.ceil(costs.shape[0] / 700)), 1)
    cs = max(int(np.ceil(costs.shape[1] / 700)), 1)
    preview = np.degrees(costs[::rs, ::cs])
    ref_times = ref_sampling.source_indices[path[:, 1]] / ref_sampling.source_fps
    target_times = target_sampling.source_indices[path[:, 0]] / target_sampling.source_fps
    figure, axes = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)
    axes.imshow(preview, origin='lower', aspect='auto', cmap='viridis',
                extent=(0, ref_sampling.source_indices[-1] / ref_sampling.source_fps,
                        0, target_sampling.source_indices[-1] / target_sampling.source_fps))
    axes.plot(ref_times, target_times, color='white', linewidth=1.)
    axes.scatter([r['reference_source_time_seconds'] for r in rows],
                 [r['teacher_source_time_seconds'] for r in rows], s=20, color='#e74c3c')
    axes.set(xlabel='Reference source time (s)', ylabel='Teacher source time (s)',
             title=f'Maximum common {target_sampling.sample_fps:.6g} FPS Geodesic DTW')
    figure.savefig(output / 'dtw_diagnostics.png', dpi=130)
    plt.close(figure)


def align_teacher_gpu(args, source, manifest, reference, reference_base, native_ref_fps, ref_frames):
    started = time.perf_counter()
    video_id = source['video_id']
    output = args.output_root / video_id
    output.mkdir()
    source_npz = args.smpl_root / source['filename']
    if digest(source_npz) != source['sha256']:
        raise ValueError('Source SMPL checksum changed after preflight.')
    target = load_fastsam_pose_sequence(source_npz)
    target_base = inspect_video_sampling(args.video_root / (video_id + '.mp4'), 5.)
    with np.load(source_npz, allow_pickle=False) as data:
        native_target_fps = float(data['fps'])
    required = np.asarray([r['source_frame_index_0based'] for r in manifest['endpoint_keyposes']])
    ref_sampling, target_sampling, policy = maximum_fps_pair(reference_base, target_base, native_ref_fps,
                                                          native_target_fps, required)
    # This teacher cohort has complete source-frame SMPL timelines. Do not infer missing poses.
    for sequence, sampling in ((reference, ref_sampling), (target, target_sampling)):
        if not np.array_equal(sequence.source_indices, np.arange(sampling.source_frame_count)):
            raise ValueError('Maximum-rate teacher entry requires a complete exact source-frame SMPL timeline.')
    ref_positions, target_positions = reference.sample_positions(ref_sampling), target.sample_positions(target_sampling)
    annotation_indices = np.searchsorted(ref_sampling.source_indices, required)
    np.testing.assert_array_equal(ref_sampling.source_indices[annotation_indices], required)
    write_json(output / 'sampling_audit.json', policy)
    np.savez_compressed(output / 'sampling.npz', reference_source_frame_indices_0based=ref_sampling.source_indices,
                        target_source_frame_indices_0based=target_sampling.source_indices,
                        reference_npz_rows_0based=ref_positions, target_npz_rows_0based=target_positions,
                        reference_source_times_seconds=ref_sampling.source_indices / ref_sampling.source_fps,
                        target_source_times_seconds=target_sampling.source_indices / target_sampling.source_fps,
                        reference_endpoint_sample_indices_0based=annotation_indices,
                        pair_sample_fps=np.asarray(policy['pair_sample_fps']))
    load_seconds = time.perf_counter() - started
    print(f'[{video_id}] {policy["pair_sample_fps"]:.9g} FPS; {len(target_positions)} x {len(ref_positions)} costs', flush=True)
    cost_profile, dtw_profile = {}, {}
    phase = time.perf_counter()
    costs = pairwise_geodesic_costs_gpu(target.body_poses[target_positions], reference.body_poses[ref_positions],
                                       device=args.device, target_chunk_size=args.target_chunk_size,
                                       reference_chunk_size=args.reference_chunk_size, profile=cost_profile)
    cost_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    if args.dtw_backend == 'cuda':
        distance, _, accumulated = dtw_from_cost_matrix_gpu(costs, device=args.device, profile=dtw_profile)
    else:
        distance, _, accumulated = dtw_from_cost_matrix(costs)
    path = backtrack_dtw_path(accumulated)
    del accumulated
    dtw_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    matches = select_reference_frame_matches(costs, path, annotation_indices)
    write_json(output / 'selected_matches.json', [dict(reference_index=m.reference_index, target_index=m.target_index,
               candidate_count=m.candidate_count, local_cost=m.local_cost) for m in matches])
    rows = bind_endpoint_rows(manifest, matches, path, reference, target, ref_sampling, target_sampling,
                              ref_positions, target_positions, video_id)
    ti, ri = path[:, 0], path[:, 1]
    np.savez_compressed(output / 'dtw_path.npz', target_sample_indices_0based=ti, reference_sample_indices_0based=ri,
                        target_source_frame_indices_0based=target_sampling.source_indices[ti],
                        reference_source_frame_indices_0based=ref_sampling.source_indices[ri],
                        target_npz_rows_0based=target_positions[ti], reference_npz_rows_0based=ref_positions[ri],
                        target_source_times_seconds=target_sampling.source_indices[ti] / target_sampling.source_fps,
                        reference_source_times_seconds=ref_sampling.source_indices[ri] / ref_sampling.source_fps,
                        local_geodesic_radians=costs[ti, ri], pair_sample_fps=np.asarray(policy['pair_sample_fps']))
    if args.save_local_costs:
        np.savez_compressed(output / 'local_costs.npz', local_geodesic_radians=costs)
    selected = np.asarray([r['teacher_npz_row_0based'] for r in rows])
    export_endpoint_smpl(source_npz, output / 'endpoint_smpl.npz', selected, list(range(1, 25)), source['sha256'])
    frames = read_video_frames(target_sampling.video_path, [r['teacher_source_frame_0based'] for r in rows])
    (output / 'keypose_frames').mkdir()
    for row in rows:
        name = f'keypose_frames/{row["move_id"]:02d}_teacher.jpg'
        frames[row['teacher_source_frame_0based']].save(output / name, quality=95)
        row.update(teacher_image=name, reference_image=f'../reference/keypose_frames/{row["move_id"]:02d}_reference.jpg')
    write_json(output / 'endpoint_keyposes.json', rows)
    write_dict_csv(output / 'endpoint_keyposes.csv', rows)
    save_segments(output, rows, target_sampling)
    render_review(output, rows, ref_frames, frames)
    plot_path(output, costs, path, ref_sampling, target_sampling, rows)
    export_seconds = time.perf_counter() - phase
    result = {'status': 'ok', 'video_id': video_id, 'reference_video_id': ANCHOR, 'endpoint_count': 24,
              'sample_fps': policy['pair_sample_fps'], 'reference_sample_count': len(ref_positions),
              'sample_count': len(target_positions), 'source_fps': target_sampling.source_fps,
              'source_frame_count': target_sampling.source_frame_count, 'strictly_increasing': True,
              'manual_review_status': 'pending', 'review_fps': 5.,
              'automatic_endpoint_rounding_to_review_grid': False, 'interpolation_or_nearest_substitution': False,
              'reference_extra_annotation_frame_count': len(policy['reference_extra_annotation_frames']),
              'dtw_distance_radians': distance, 'dtw_path_length': len(path), 'dtw_backend': args.dtw_backend,
              'dense_cost_matrix_saved': args.save_local_costs, 'gpu_cost_profile': cost_profile,
              'gpu_dtw_profile': dtw_profile, 'load_and_sampling_seconds': load_seconds,
              'geodesic_cost_seconds': cost_seconds, 'dtw_seconds': dtw_seconds,
              'selection_and_export_seconds': export_seconds, 'elapsed_seconds': time.perf_counter() - started}
    write_json(output / 'summary.json', result)
    print(f'[{video_id}] 24 endpoints; costs {cost_seconds:.3f}s; DTW {dtw_seconds:.3f}s; total {result["elapsed_seconds"]:.3f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smpl-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/FastSAM_SMPL/teacher_single_person_11'))
    parser.add_argument('--video-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/teach_trimmed'))
    parser.add_argument('--reference-manifest', type=Path, default=PROJECT / 'reference_data/BV1WE411W7JB/reference_manifest.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-chunk-size', type=int, default=256)
    parser.add_argument('--reference-chunk-size', type=int, default=1024)
    parser.add_argument('--dtw-backend', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--teacher-id', action='append', help='Optional named subset; output must still use a new directory.')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--save-local-costs', action='store_true', help='Optional large dense matrices; path and endpoint costs are always saved.')
    args = parser.parse_args()
    if min(args.target_chunk_size, args.reference_chunk_size) <= 0:
        parser.error('GPU tile sizes must be positive.')
    started = time.perf_counter()
    args.output_root = args.output_root.expanduser().absolute()
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest, sources, audit = preflight(args)
    targets = [r for r in sources if r['video_id'] != ANCHOR]
    if args.teacher_id:
        if set(args.teacher_id) - {r['video_id'] for r in targets}:
            raise ValueError('Unknown teacher or anchor requested as a target.')
        targets = [r for r in targets if r['video_id'] in args.teacher_id]
    audit.update(legacy_5fps_preflight_only_validates_source_identity=True,
                 production_sampling_policy='pair-specific maximum common video/SMPL FPS plus exact reference annotations',
                 review_fps=5., automatic_endpoint_rounding_to_review_grid=False)
    write_json(args.output_root / 'preflight.json', audit)
    provenance = args.output_root / 'provenance'
    provenance.mkdir()
    shutil.copy2(args.reference_manifest, provenance / 'reference_manifest_snapshot.json')
    code_files = ('run_fastsam_teacher_endpoints_gpu.py', 'aqa3d/smpl_dtw_gpu.py', 'aqa3d/fastsam_high_fps.py',
                  'run_fastsam_teacher_endpoints.py', 'aqa3d/fastsam_smpl.py', 'aqa3d/smpl_dtw.py',
                  'aqa3d/endpoint_transfer.py', 'aqa3d/smpl_pose_segmentation.py',
                  'run_smpl_pose_dtw_segmentation.py', 'visualize_tas_boundary_frames.py')
    hashes = {}
    for name in code_files:
        dest = provenance / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
        hashes[name] = digest(PROJECT / name)
    write_json(provenance / 'code_sha256.json', hashes)
    if args.preflight_only:
        print('Preflight only; no CUDA alignment run.', flush=True)
        return
    args.device = str(cuda_device(args.device))
    phase = time.perf_counter()
    reference_npz = args.smpl_root / (ANCHOR + '.npz')
    reference = load_fastsam_pose_sequence(reference_npz)
    reference_base = inspect_video_sampling(args.video_root / (ANCHOR + '.mp4'), 5.)
    with np.load(reference_npz, allow_pickle=False) as data:
        native_ref_fps = float(data['fps'])
    reference_source_frames = [r['source_frame_index_0based'] for r in manifest['endpoint_keyposes']]
    ref_frames = read_video_frames(reference_base.video_path, reference_source_frames)
    ref_output = args.output_root / 'reference'
    (ref_output / 'keypose_frames').mkdir(parents=True)
    for row in manifest['endpoint_keyposes']:
        ref_frames[row['source_frame_index_0based']].save(ref_output / 'keypose_frames' / f'{row["move_id"]:02d}_reference.jpg', quality=95)
    export_endpoint_smpl(reference_npz, ref_output / 'endpoint_smpl.npz',
                         np.searchsorted(reference.source_indices, reference_source_frames), list(range(1, 25)), digest(reference_npz))
    warm_costs = pairwise_geodesic_costs_gpu(reference.body_poses[:4], reference.body_poses[:4], device=args.device)
    if args.dtw_backend == 'cuda':
        dtw_from_cost_matrix_gpu(warm_costs, device=args.device)
    setup_seconds = time.perf_counter() - phase
    cohort = [{'teacher_code': f'T{i:02d}', 'video_id': r['video_id'], 'is_anchor': False,
               'smpl_file': str(args.smpl_root / r['filename']), 'replaces': r.get('replaces') or ''}
              for i, r in enumerate([r for r in sources if r['video_id'] != ANCHOR], 1) if r in targets]
    write_dict_csv(args.output_root / 'teacher_cohort.csv', cohort)
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
               'reference_video_id': ANCHOR, 'reference_video_path': str(reference_base.video_path),
               'algorithm': 'exact dense Geodesic DTW, mean of 23 local SO(3) rotations, global orientation excluded',
               'sampling_policy': 'pair-specific maximum common FPS plus immutable reference annotation frames',
               'device': args.device, 'GPU': torch.cuda.get_device_name(args.device),
               'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
               'dtype': 'float64', 'dtw_backend': args.dtw_backend, 'teacher_count_expected': len(targets),
               'reference_in_numeric_cohort': False, 'ranking_or_reference_intervals_computed': False,
               'reference_setup_and_cuda_warmup_seconds': setup_seconds, 'results': []}
    for source in targets:
        try:
            result = align_teacher_gpu(args, source, manifest, reference, reference_base, native_ref_fps, ref_frames)
        except Exception as exc:
            result = {'status': 'failed', 'video_id': source['video_id'], 'error_type': type(exc).__name__,
                      'error': str(exc), 'manual_review_status': 'pending'}
            output = args.output_root / source['video_id']
            output.mkdir(exist_ok=True)
            write_json(output / 'summary.json', result)
            print(f'[{source["video_id"]}] FAILED: {exc}', flush=True)
        summary['results'].append(result)
        write_json(args.output_root / 'batch_summary.json', summary)
    changed = [name for name, value in audit['source_files_sha256'].items() if digest(Path(name)) != value]
    good = [r for r in summary['results'] if r['status'] == 'ok']
    summary.update(status='complete_pending_manual_review' if len(good) == len(targets) and not changed else 'incomplete',
                   successful_teachers=len(good), failed_teachers=len(targets) - len(good),
                   transferred_endpoint_count=len(good) * 24, source_files_unchanged=not changed,
                   changed_source_files=changed, summed_geodesic_cost_seconds=sum(r['geodesic_cost_seconds'] for r in good),
                   summed_dtw_seconds=sum(r['dtw_seconds'] for r in good),
                   summed_pair_elapsed_seconds=sum(r['elapsed_seconds'] for r in good),
                   batch_elapsed_seconds=time.perf_counter() - started)
    write_json(args.output_root / 'batch_summary.json', summary)
    timing_fields = ('video_id', 'sample_fps', 'sample_count', 'reference_sample_count',
                     'geodesic_cost_seconds', 'dtw_seconds', 'selection_and_export_seconds', 'elapsed_seconds')
    if good:
        write_dict_csv(args.output_root / 'timing_summary.csv', [{k: r[k] for k in timing_fields} for r in good])
        combined = []
        for r in good:
            combined.extend(json.loads((args.output_root / r['video_id'] / 'endpoint_keyposes.json').read_text()))
        write_json(args.output_root / 'all_teacher_endpoints.json', combined)
        write_dict_csv(args.output_root / 'all_teacher_endpoints.csv', combined)
    lines = ['# Maximum-FPS GPU Teacher Endpoint Transfer', '',
             'The original 24 anchor source frames are unchanged. Automatic endpoints are not rounded to 5 FPS.', '',
             '| Teacher | FPS | Geodesic (s) | DTW (s) | Pair Total (s) |', '| --- | --- | --- | --- | --- |']
    for r in good:
        lines.append(f'| [{r["video_id"]}]({args.output_root}/{r["video_id"]}/review.md) | {r["sample_fps"]:.6f} | '
                     f'{r["geodesic_cost_seconds"]:.3f} | {r["dtw_seconds"]:.3f} | {r["elapsed_seconds"]:.3f} |')
    lines.extend(['', 'All transferred endpoints remain pending human review. Existing review annotations were not modified.',
                  'Review presentation is 5 FPS; this is separate from alignment sampling. The current review service was not replaced.',
                  f'Batch wall time: {summary["batch_elapsed_seconds"]:.3f} s. Dense local matrices saved: {args.save_local_costs}.', ''])
    (args.output_root / 'index.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary['status'] == 'incomplete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
