#!/usr/bin/env python
"""Transfer the existing 24 endpoint annotations to ten FastSAM SMPL teachers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image, ImageDraw

from aqa3d.endpoint_transfer import manifest_boundaries
from aqa3d.fastsam_smpl import (
    FastSamPoseSequence, export_endpoint_smpl, load_fastsam_pose_sequence, transferred_fastsam_rows,
)
from aqa3d.smpl_dtw import (
    VideoSampling, backtrack_dtw_path, dtw_from_cost_matrix, inspect_video_sampling,
    pairwise_geodesic_costs, select_reference_frame_matches,
)
from aqa3d.smpl_pose_segmentation import build_mapped_boundaries, build_mapped_segments
from run_smpl_pose_dtw_segmentation import plot_diagnostics, write_dict_csv
from visualize_tas_boundary_frames import fit_frame, load_font, read_video_frames


PROJECT = Path(__file__).resolve().parent
ANCHOR = 'BV1WE411W7JB'
SAMPLE_FPS = 5.0


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, data: object) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def preflight(args: argparse.Namespace) -> tuple[dict, list[dict], dict]:
    manifest = json.loads(args.reference_manifest.read_text(encoding='utf-8-sig'))
    if manifest['reference']['video_id'] != ANCHOR or manifest['reference']['sample_fps'] != SAMPLE_FPS:
        raise ValueError('Use the confirmed BV1WE411W7JB 5 FPS endpoint manifest.')
    with (args.smpl_root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        sources = list(csv.DictReader(handle))
    if (len(sources) != 11 or len({r['video_id'] for r in sources}) != 11
            or ANCHOR not in {r['video_id'] for r in sources}
            or {p.stem for p in args.smpl_root.glob('*.npz')} != {r['video_id'] for r in sources}):
        raise ValueError('Expected exactly the new eleven-video SMPL cohort including the anchor.')
    audits, files = [], {}
    for source in sources:
        video_id = source['video_id']
        if source['filename'] != video_id + '.npz' or Path(video_id).name != video_id:
            raise ValueError('Invalid source video identity.')
        npz = args.smpl_root / source['filename']
        video = args.video_root / (video_id + '.mp4')
        npz_sha = digest(npz)
        if npz_sha != source['sha256']:
            raise ValueError('SMPL source checksum mismatch: ' + video_id)
        sampling = inspect_video_sampling(video, SAMPLE_FPS)
        sequence = load_fastsam_pose_sequence(npz)
        positions = sequence.sample_positions(sampling)
        if (len(sequence.source_indices) != int(source['sampled_frames'])
                or sequence.audit['usable_frame_count'] != int(source['valid_frames'])):
            raise ValueError('SMPL row count differs from its source manifest.')
        if video_id == ANCHOR:
            manifest_boundaries(manifest, sampling)
        files[str(npz)] = npz_sha
        files[str(video)] = digest(video)
        audits.append(dict(sequence.audit, video_id=video_id, video_path=str(video),
                           smpl_path=str(npz), smpl_sha256=npz_sha,
                           source_frame_count=sampling.source_frame_count,
                           sample_count_5fps=sampling.sample_count,
                           exact_5fps_rows=len(positions),
                           complete_source_timeline=bool(np.array_equal(
                               sequence.source_indices, np.arange(sampling.source_frame_count))),
                           source_selection_method=source['selection_method'],
                           replaces=source.get('replaces') or None))
        print(f'Preflight {video_id}: {len(sequence.source_indices)} source rows, '
              f'{len(positions)} exact 5 FPS samples', flush=True)
    files[str(args.reference_manifest)] = digest(args.reference_manifest)
    for name in ('manifest.csv', 'manifest.json', 'README.md', 'SHA256SUMS.txt'):
        files[str(args.smpl_root / name)] = digest(args.smpl_root / name)
    report = {'status': 'passed', 'cohort_count': 11, 'reference_video_id': ANCHOR,
              'target_teacher_count': 10, 'endpoint_count_per_video': 24,
              'total_source_rows': sum(r['source_row_count'] for r in audits),
              'videos': audits, 'source_files_sha256': files,
              'reference_annotations_reused_without_time_changes': True,
              'reference_manifest_used_for': 'temporal annotations and event text only; old PHALP path is not loaded',
              'sample_fps': SAMPLE_FPS, 'interpolation_or_nearest_frame_substitution': False,
              'rotation_repair_or_global_orientation_canonicalization': False}
    return manifest, sources, report


def render_review(output: Path, rows: list[dict], reference_frames: dict,
                  teacher_frames: dict, *, target_role: str = 'teacher') -> None:
    title_font, font = load_font(22), load_font(16)
    review = output / 'review'
    review.mkdir()

    def page(items: list[dict], destination: Path) -> None:
        canvas = Image.new('RGB', (1160, 64 + 332 * len(items)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 14), f'{ANCHOR} Gold / {output.name} DTW (pending review)', font=title_font, fill='black')
        for i, row in enumerate(items):
            top = 64 + 332 * i
            draw.line((0, top, 1160, top), fill='#bbbbbb')
            draw.text((16, top + 6), f'{row["move_id"]:02d} {row["move_name_zh"]} | {row["pose_id"]}',
                      font=title_font, fill='black')
            for column, prefix, frames in ((0, 'reference', reference_frames), (1, target_role, teacher_frames)):
                source = row[prefix + '_source_frame_0based']
                label = f'{prefix}: {row[prefix + "_source_time_seconds"]:.3f}s | source frame0 {source}'
                draw.text((12 + 580 * column, top + 40), label, font=font, fill='#333333')
                canvas.paste(fit_frame(frames[source], 556, 225), (12 + 580 * column, top + 66))
            label = f'Local geodesic {row["local_geodesic_degrees"]:.2f} deg; path candidates {row["candidate_count"]}; manual review pending'
            draw.text((16, top + 302), label, font=font, fill='#333333')
        canvas.save(destination, quality=92)

    page(rows, output / 'boundary_frames.jpg')
    for index, first in enumerate(range(0, len(rows), 6), 1):
        page(rows[first:first + 6], review / f'page_{index:02d}.jpg')
    lines = [f'# {output.name} · 24 式结束定势', '',
             '左：既有人工确认的锚点边界。右：新 SMPL 的自动 DTW 迁移结果，尚未人工确认。', '',
             '[结束定势表](' + str(output / 'endpoint_keyposes.csv') + ') · '
             '[结束定势 SMPL](' + str(output / 'endpoint_smpl.npz') + ')', '']
    for index in range(1, 5):
        lines.extend([f'![第 {index} 页]({review}/page_{index:02d}.jpg)', ''])
    (output / 'review.md').write_text('\n'.join(lines), encoding='utf-8')


def save_path(output: Path, path: np.ndarray, costs: np.ndarray,
              reference_sampling: VideoSampling, teacher_sampling: VideoSampling,
              reference_positions: np.ndarray, teacher_positions: np.ndarray,
              *, target_video_pts: np.ndarray | None = None) -> None:
    target, reference = path[:, 0], path[:, 1]
    np.savez_compressed(
        output / 'dtw_path.npz', target_sample_indices_0based=target,
        reference_sample_indices_0based=reference,
        target_source_frame_indices_0based=teacher_sampling.source_indices[target],
        reference_source_frame_indices_0based=reference_sampling.source_indices[reference],
        target_npz_rows_0based=teacher_positions[target],
        reference_npz_rows_0based=reference_positions[reference],
        target_grid_times_seconds=target / SAMPLE_FPS,
        reference_grid_times_seconds=reference / SAMPLE_FPS,
        target_source_times_seconds=(teacher_sampling.source_indices[target] / teacher_sampling.source_fps
                                     if target_video_pts is None else target_video_pts[teacher_sampling.source_indices[target]]),
        reference_source_times_seconds=reference_sampling.source_indices[reference] / reference_sampling.source_fps,
        local_geodesic_radians=costs[target, reference],
    )
    np.savez_compressed(output / 'local_costs.npz', local_geodesic_radians=costs)


def align_teacher(args: argparse.Namespace, source: dict, manifest: dict, reference: FastSamPoseSequence,
                  reference_sampling: VideoSampling, reference_positions: np.ndarray,
                  reference_frames: dict) -> dict:
    started = time.perf_counter()
    video_id = source['video_id']
    output = args.output_root / video_id
    output.mkdir()
    npz = args.smpl_root / source['filename']
    if digest(npz) != source['sha256']:
        raise ValueError('SMPL file changed after preflight.')
    sampling = inspect_video_sampling(args.video_root / (video_id + '.mp4'), SAMPLE_FPS)
    teacher = load_fastsam_pose_sequence(npz)
    positions = teacher.sample_positions(sampling)
    gold = manifest_boundaries(manifest, reference_sampling)
    print(f'[{video_id}] geodesic costs {sampling.sample_count} x {reference_sampling.sample_count}', flush=True)
    costs = pairwise_geodesic_costs(teacher.body_poses[positions], reference.body_poses[reference_positions],
                                   chunk_size=args.pairwise_chunk_size)
    distance, _, accumulated = dtw_from_cost_matrix(costs)
    path = backtrack_dtw_path(accumulated)
    del accumulated
    save_path(output, path, costs, reference_sampling, sampling, reference_positions, positions)
    matches = select_reference_frame_matches(costs, path, [b.sample_index_0based for b in gold])
    write_json(output / 'selected_matches.json', [asdict(m) for m in matches])
    boundaries = build_mapped_boundaries(video_id, gold, matches, reference_sampling, sampling)
    segments = build_mapped_segments(video_id, boundaries, sampling)
    rows = transferred_fastsam_rows(manifest, boundaries, path, reference_sampling, sampling,
                                   reference, teacher, reference_positions, positions)
    selected = np.asarray([r['teacher_npz_row_0based'] for r in rows])
    export_endpoint_smpl(npz, output / 'endpoint_smpl.npz', selected,
                         [r['move_id'] for r in rows], source['sha256'])
    frames = read_video_frames(sampling.video_path, [r['teacher_source_frame_0based'] for r in rows])
    (output / 'keypose_frames').mkdir()
    for row in rows:
        image = f'keypose_frames/{row["move_id"]:02d}_teacher.jpg'
        frames[row['teacher_source_frame_0based']].save(output / image, quality=95)
        row['teacher_image'] = image
        row['reference_image'] = f'../reference/keypose_frames/{row["move_id"]:02d}_reference.jpg'
    write_dict_csv(output / 'endpoint_keyposes.csv', rows)
    write_json(output / 'endpoint_keyposes.json', rows)
    # The shared legacy dataclass has a PHALP alias; do not mislabel NPZ data with it.
    boundary_rows = [{k: v for k, v in asdict(b).items() if k != 'target_phalp_frame_1based'} for b in boundaries]
    write_dict_csv(output / 'boundaries.csv', boundary_rows)
    write_dict_csv(output / 'segments.csv', [asdict(s) for s in segments])
    plot_diagnostics(output / 'dtw_diagnostics.png', video_id, costs, path,
                     gold, boundaries, reference_sampling, sampling)
    render_review(output, rows, reference_frames, frames)
    result = {'status': 'ok', 'video_id': video_id, 'reconstruction': 'FastSAM SMPL',
              'reference_video_id': ANCHOR, 'endpoint_count': len(rows),
              'sample_fps': SAMPLE_FPS, 'source_fps': sampling.source_fps,
              'source_frame_count': sampling.source_frame_count, 'sample_count': sampling.sample_count,
              'smpl_npz_sha256': source['sha256'], 'strictly_increasing': True,
              'manual_review_status': 'pending', 'manual_review_required_count': 24,
              'multiple_candidate_endpoint_count': sum(r['candidate_count'] > 1 for r in rows),
              'maximum_endpoint_local_geodesic_degrees': max(r['local_geodesic_degrees'] for r in rows),
              'dtw_path_length': len(path), 'dtw_distance_radians': distance,
              'unmapped_tail_source_frame_count': sampling.source_frame_count - rows[-1]['teacher_source_frame_1based'],
              'elapsed_seconds': time.perf_counter() - started}
    write_json(output / 'summary.json', result)
    print(f'[{video_id}] saved 24 ordered endpoints ({result["elapsed_seconds"]:.1f}s)', flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smpl-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/FastSAM_SMPL/teacher_single_person_11'))
    parser.add_argument('--video-root', type=Path, default=Path('/data1/shenqingwei/VisualSearch/aqa/teach_trimmed'))
    parser.add_argument('--reference-manifest', type=Path, default=PROJECT / 'reference_data/BV1WE411W7JB/reference_manifest.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pairwise-chunk-size', type=int, default=32)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.pairwise_chunk_size <= 0:
        parser.error('Pairwise chunk size must be positive.')
    args.output_root = args.output_root.expanduser().absolute()
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest, sources, audit = preflight(args)
    write_json(args.output_root / 'preflight.json', audit)
    provenance = args.output_root / 'provenance'
    provenance.mkdir()
    shutil.copy2(args.reference_manifest, provenance / 'reference_manifest_snapshot.json')
    for name in ('manifest.csv', 'manifest.json', 'README.md', 'SHA256SUMS.txt'):
        shutil.copy2(args.smpl_root / name, provenance / ('source_' + name))
    code = ['run_fastsam_teacher_endpoints.py', 'aqa3d/fastsam_smpl.py', 'aqa3d/smpl_dtw.py',
            'aqa3d/endpoint_transfer.py', 'aqa3d/smpl_pose_segmentation.py',
            'run_smpl_pose_dtw_segmentation.py', 'visualize_tas_boundary_frames.py']
    hashes = {}
    for name in code:
        dest = provenance / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
        hashes[name] = digest(PROJECT / name)
    write_json(provenance / 'code_sha256.json', hashes)
    if args.preflight_only:
        print('Preflight only; no DTW alignments run.', flush=True)
        return
    reference_npz = args.smpl_root / (ANCHOR + '.npz')
    reference_sampling = inspect_video_sampling(args.video_root / (ANCHOR + '.mp4'), SAMPLE_FPS)
    reference = load_fastsam_pose_sequence(reference_npz)
    reference_positions = reference.sample_positions(reference_sampling)
    reference_sources = [r['source_frame_index_0based'] for r in manifest['endpoint_keyposes']]
    reference_frames = read_video_frames(reference_sampling.video_path, reference_sources)
    ref_output = args.output_root / 'reference'
    (ref_output / 'keypose_frames').mkdir(parents=True)
    for row in manifest['endpoint_keyposes']:
        reference_frames[row['source_frame_index_0based']].save(
            ref_output / 'keypose_frames' / f'{row["move_id"]:02d}_reference.jpg', quality=95)
    ref_rows = np.searchsorted(reference.source_indices, reference_sources)
    export_endpoint_smpl(reference_npz, ref_output / 'endpoint_smpl.npz', ref_rows,
                         list(range(1, 25)), digest(reference_npz))
    write_json(ref_output / 'endpoint_annotations.json', manifest['endpoint_keyposes'])
    targets = [r for r in sources if r['video_id'] != ANCHOR]
    cohort = [{'teacher_code': f'T{i:02d}', 'video_id': r['video_id'], 'is_anchor': False,
               'smpl_file': str(args.smpl_root / r['filename']), 'replaces': r.get('replaces') or ''}
              for i, r in enumerate(targets, 1)]
    write_dict_csv(args.output_root / 'teacher_cohort.csv', cohort)
    results = []
    summary = {'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
               'status': 'running', 'reference_video_id': ANCHOR,
               'algorithm': 'unchanged full-sequence 5 FPS SMPL geodesic DTW; mean over 23 local rotations; global_orient excluded',
               'teacher_count_expected': 10, 'reference_in_numeric_cohort': False,
               'ranking_or_reference_intervals_computed': False, 'results': results}
    for source in targets:
        try:
            result = align_teacher(args, source, manifest, reference, reference_sampling,
                                   reference_positions, reference_frames)
        except Exception as exc:
            result = {'status': 'failed', 'video_id': source['video_id'],
                      'error_type': type(exc).__name__, 'error': str(exc), 'manual_review_status': 'pending'}
            dest = args.output_root / source['video_id']
            dest.mkdir(exist_ok=True)
            write_json(dest / 'summary.json', result)
            print(f'[{source["video_id"]}] FAILED: {exc}', flush=True)
        results.append(result)
        write_json(args.output_root / 'batch_summary.json', summary)
    changed = [name for name, value in audit['source_files_sha256'].items() if digest(Path(name)) != value]
    summary.update(status='complete_pending_manual_review' if all(r['status'] == 'ok' for r in results) and not changed else 'incomplete',
                   successful_teachers=sum(r['status'] == 'ok' for r in results),
                   failed_teachers=sum(r['status'] != 'ok' for r in results),
                   transferred_endpoint_count=sum(r.get('endpoint_count', 0) for r in results),
                   source_files_unchanged=not changed, changed_source_files=changed)
    write_json(args.output_root / 'batch_summary.json', summary)
    combined = []
    lines = ['# FastSAM 教师结束定势迁移', '',
             '锚点 BV1WE411W7JB 沿用原有 24 式人工标注；其余 10 位教师使用新 FastSAM SMPL 对齐。', '',
             '所有自动迁移定势均待人工复核；对齐成功与时间有序不代表动作定势已被人工确认。', '',
             '| 教师 | 视频 | 状态 | 定势与截图 |', '| --- | --- | --- | --- |']
    for member, result in zip(cohort, results):
        video_id = result['video_id']
        if result['status'] == 'ok':
            rows = json.loads((args.output_root / video_id / 'endpoint_keyposes.json').read_text())
            combined.extend([dict(teacher_code=member['teacher_code'], **r) for r in rows])
            link = f'[24 式配对图]({args.output_root}/{video_id}/review.md)'
        else:
            link = f'[失败记录]({args.output_root}/{video_id}/summary.json)'
        lines.append(f'| {member["teacher_code"]} | {video_id} | {result["status"]} | {link} |')
    if combined:
        write_dict_csv(args.output_root / 'all_teacher_endpoints.csv', combined)
        write_json(args.output_root / 'all_teacher_endpoints.json', combined)
    lines.extend(['', '[教师映射表](' + str(args.output_root / 'teacher_cohort.csv') + ') · '
                  '[批处理摘要](' + str(args.output_root / 'batch_summary.json') + ')', '',
                  '网格时间与实际源帧时间分别保存；未平滑、插值、修正姿态或以最近帧替代缺失数据。',
                  '转换残差仅是模型转换诊断，不是真实人体重建误差或动作质量分数。', ''])
    (args.output_root / 'index.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary['status'] == 'incomplete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
