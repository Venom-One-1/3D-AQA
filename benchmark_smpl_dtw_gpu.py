"""Compare the unchanged CPU code and separate CUDA implementation on real poses."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from aqa3d.fastsam_smpl import load_fastsam_pose_sequence
from aqa3d.smpl_dtw import backtrack_dtw_path, dtw_from_cost_matrix, pairwise_geodesic_costs, select_reference_frame_matches
from aqa3d.smpl_dtw_gpu import cuda_device, dtw_from_cost_matrix_gpu, pairwise_geodesic_costs_gpu
from run_fastsam_teacher_endpoints import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('/data1/shenqingwei/VisualSearch/aqa/FastSAM_SMPL/teacher_single_person_11')
    parser.add_argument('--target-smpl', type=Path, default=root / 'QxVvRcRn2TA.npz')
    parser.add_argument('--reference-smpl', type=Path, default=root / 'BV1WE411W7JB.npz')
    parser.add_argument('--frames', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 2 or args.repeats < 1:
        parser.error('Use at least two frames and one repeat.')
    selected = cuda_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=False)
    target = load_fastsam_pose_sequence(args.target_smpl)
    reference = load_fastsam_pose_sequence(args.reference_smpl)
    target_rows = np.linspace(0, len(target.body_poses) - 1, min(args.frames, len(target.body_poses)), dtype=int)
    reference_rows = np.linspace(0, len(reference.body_poses) - 1, min(args.frames, len(reference.body_poses)), dtype=int)
    if not target.usable_mask[target_rows].all() or not reference.usable_mask[reference_rows].all():
        raise ValueError('Benchmark sample contains invalid poses.')
    a, b = target.body_poses[target_rows], reference.body_poses[reference_rows]
    warm_started = time.perf_counter()
    warm_costs = pairwise_geodesic_costs_gpu(a[:4], b[:4], device=selected)
    dtw_from_cost_matrix_gpu(warm_costs, device=selected)
    warm_seconds = time.perf_counter() - warm_started
    trials = []
    for repeat in range(args.repeats):
        started = time.perf_counter()
        cpu_costs = pairwise_geodesic_costs(a, b)
        cpu_cost_seconds = time.perf_counter() - started
        cost_profile, dtw_profile = {}, {}
        gpu_costs = pairwise_geodesic_costs_gpu(a, b, device=selected, profile=cost_profile)
        np.testing.assert_allclose(gpu_costs, cpu_costs, rtol=0, atol=1e-7)
        started = time.perf_counter()
        cpu_distance, _, cpu_accumulated = dtw_from_cost_matrix(cpu_costs)
        cpu_dtw_seconds = time.perf_counter() - started
        gpu_distance, _, gpu_accumulated = dtw_from_cost_matrix_gpu(gpu_costs, device=selected, profile=dtw_profile)
        _, _, cpu_same_cost_accumulated = dtw_from_cost_matrix(gpu_costs)
        np.testing.assert_array_equal(gpu_accumulated, cpu_same_cost_accumulated)
        cpu_path, gpu_path = backtrack_dtw_path(cpu_accumulated), backtrack_dtw_path(gpu_accumulated)
        np.testing.assert_array_equal(gpu_path, cpu_path)
        endpoints = np.unique(np.linspace(0, len(b) - 1, min(24, len(b)), dtype=int))
        cpu_matches = select_reference_frame_matches(cpu_costs, cpu_path, endpoints)
        gpu_matches = select_reference_frame_matches(gpu_costs, gpu_path, endpoints)
        assert [m.target_index for m in gpu_matches] == [m.target_index for m in cpu_matches]
        trial = {'repeat': repeat + 1, 'cpu_cost_seconds': cpu_cost_seconds,
                 'gpu_cost_seconds': cost_profile['wall_seconds'], 'cpu_dtw_seconds': cpu_dtw_seconds,
                 'gpu_dtw_seconds': dtw_profile['wall_seconds'],
                 'max_distance_matrix_error_radians': float(abs(gpu_costs - cpu_costs).max()),
                 'normalized_DTW_distance_error_radians': abs(gpu_distance - cpu_distance),
                 'path_exactly_equal': True, 'selected_frames_exactly_equal': True,
                 'same_cost_accumulated_matrix_bitwise_equal': True,
                 'gpu_cost_profile': cost_profile, 'gpu_dtw_profile': dtw_profile}
        trials.append(trial)
        print(f'Repeat {repeat + 1}: costs CPU {cpu_cost_seconds:.3f}s / GPU {cost_profile["wall_seconds"]:.3f}s; '
              f'DTW CPU {cpu_dtw_seconds:.3f}s / GPU {dtw_profile["wall_seconds"]:.3f}s', flush=True)
    medians = {key: float(np.median([t[key] for t in trials])) for key in
               ('cpu_cost_seconds', 'gpu_cost_seconds', 'cpu_dtw_seconds', 'gpu_dtw_seconds')}
    report = {'status': 'passed', 'matrix_shape': [len(a), len(b)], 'repeats': args.repeats,
              'device': str(selected), 'GPU': torch.cuda.get_device_name(selected),
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'dtype': 'float64',
              'warmup_seconds_excluded_from_trials': warm_seconds,
              'source_files_sha256': {str(p): digest(p) for p in (args.target_smpl, args.reference_smpl)},
              'sample_policy': 'uniform rows across full native sequences for validation only; not a production FPS setting',
              'median_seconds': medians,
              'distance_matrix_speedup': medians['cpu_cost_seconds'] / medians['gpu_cost_seconds'],
              'DTW_speedup': medians['cpu_dtw_seconds'] / medians['gpu_dtw_seconds'], 'trials': trials}
    np.savez_compressed(args.output_root / 'validation_samples.npz', target_npz_rows=target_rows,
                        reference_npz_rows=reference_rows, cpu_costs=cpu_costs, gpu_costs=gpu_costs,
                        cpu_path=cpu_path, gpu_path=gpu_path)
    for name in ('benchmark_smpl_dtw_gpu.py', 'aqa3d/smpl_dtw_gpu.py', 'aqa3d/smpl_dtw.py'):
        dest = args.output_root / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent / name, dest)
    write_json(args.output_root / 'benchmark.json', report)
    print('Benchmark passed: ' + str(args.output_root / 'benchmark.json'), flush=True)


if __name__ == '__main__':
    main()
