"""Opt-in float64 CUDA SMPL distances and exact-recurrence DTW.

The legacy CPU module is deliberately unchanged. No approximate DTW, rotation
repair, reduced precision, or silent CPU fallback is used here.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .smpl_dtw import backtrack_dtw_path


def cuda_device(device: str | torch.device = 'cuda:0') -> torch.device:
    selected = torch.device(device)
    if selected.type != 'cuda':
        raise ValueError('This implementation requires an explicit CUDA device.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; the GPU implementation does not silently fall back to CPU.')
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError('CUDA device index is outside the available devices.')
    return torch.device('cuda', index)


def _poses(values: np.ndarray, role: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4 or array.shape[1:] != (23, 3, 3) or not len(array):
        raise ValueError(f'Expected nonempty {role} poses shaped (N,23,3,3).')
    if not np.isfinite(array).all():
        raise ValueError(f'Non-finite {role} poses.')
    return np.ascontiguousarray(array)


def pairwise_geodesic_costs_gpu(target_poses: np.ndarray, reference_poses: np.ndarray, *,
                                device: str | torch.device = 'cuda:0',
                                target_chunk_size: int = 256, reference_chunk_size: int = 1024,
                                profile: dict | None = None) -> np.ndarray:
    """Return float64 radians, averaging SO(3) distances over 23 local joints."""
    if target_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError('Both GPU chunk sizes must be positive.')
    target, reference = _poses(target_poses, 'target'), _poses(reference_poses, 'reference')
    selected = cuda_device(device)
    torch.cuda.synchronize(selected)
    started = time.perf_counter()
    costs = np.empty((len(target), len(reference)), dtype=np.float64)
    with torch.inference_mode(), torch.cuda.device(selected):
        event_start, event_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        event_start.record()
        # trace(A @ B.T) == sum(A * B); joint-batched dot products avoid 3x3 temporaries.
        target_gpu = torch.from_numpy(target).to(selected).permute(1, 0, 2, 3).reshape(23, len(target), 9).contiguous()
        reference_gpu = torch.from_numpy(reference).to(selected).permute(1, 2, 3, 0).reshape(23, 9, len(reference)).contiguous()
        for first in range(0, len(target), target_chunk_size):
            end = min(first + target_chunk_size, len(target))
            for left in range(0, len(reference), reference_chunk_size):
                right = min(left + reference_chunk_size, len(reference))
                trace = torch.bmm(target_gpu[:, first:end], reference_gpu[:, :, left:right])
                trace.sub_(1.).mul_(.5).clamp_(-1., 1.).acos_()
                costs[first:end, left:right] = trace.mean(dim=0).cpu().numpy()
        event_end.record()
        torch.cuda.synchronize(selected)
        stream_seconds = event_start.elapsed_time(event_end) / 1000.
    if profile is not None:
        profile.update(wall_seconds=time.perf_counter() - started, cuda_stream_elapsed_seconds=stream_seconds,
                       device=str(selected), device_name=torch.cuda.get_device_name(selected), dtype='float64',
                       matrix_shape=list(costs.shape), target_chunk_size=target_chunk_size,
                       reference_chunk_size=reference_chunk_size, includes_host_device_transfers=True)
    return costs


def dtw_from_cost_matrix_gpu(local_costs: np.ndarray, coefficient: float = 1., *,
                             device: str | torch.device = 'cuda:0',
                             profile: dict | None = None) -> tuple[float, dict[int, list[int]], np.ndarray]:
    """Parallelize independent anti-diagonals; preserve the legacy DTW recurrence."""
    local = np.asarray(local_costs, dtype=np.float64)
    if (local.ndim != 2 or not local.size or not np.isfinite(local).all()
            or np.any(local < 0)):
        raise ValueError('DTW costs must be a nonempty finite nonnegative matrix.')
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('DTW coefficient must be finite and positive.')
    selected = cuda_device(device)
    torch.cuda.synchronize(selected)
    started = time.perf_counter()
    rows, cols = local.shape
    with torch.inference_mode(), torch.cuda.device(selected):
        event_start, event_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        event_start.record()
        accumulated = torch.from_numpy(np.ascontiguousarray(local)).to(selected)
        # Match even the legacy edge initialization and its summation order exactly.
        accumulated[0, 1:] = torch.from_numpy(np.cumsum(local[0, 1:]) + local[0, 0]).to(selected)
        accumulated[1:, 0] = torch.from_numpy(np.cumsum(local[1:, 0]) + local[0, 0]).to(selected)
        accumulated[0, 0] = float(local[0, 0] * coefficient)
        offsets = torch.arange(min(rows, cols), device=selected)
        for diagonal in range(2, rows + cols - 1):
            first, last = max(1, diagonal - cols + 1), min(rows - 1, diagonal - 1)
            if first > last:
                continue
            r = offsets[:last - first + 1] + first
            c = diagonal - r
            value = accumulated[r, c]
            up = value + accumulated[r - 1, c]
            left = value + accumulated[r, c - 1]
            corner = coefficient * value + accumulated[r - 1, c - 1]
            accumulated[r, c] = torch.minimum(torch.minimum(up, left), corner)
        host = accumulated.cpu().numpy()
        event_end.record()
        torch.cuda.synchronize(selected)
        stream_seconds = event_start.elapsed_time(event_end) / 1000.
    recurrence_seconds = time.perf_counter() - started
    backtrack_started = time.perf_counter()
    # Reuse the original diagonal/up/left tie priority, without a GPU-specific heuristic.
    path = backtrack_dtw_path(host)
    matching: dict[int, list[int]] = {}
    for target_index, reference_index in path:
        matching.setdefault(int(target_index), []).append(int(reference_index))
    if profile is not None:
        profile.update(wall_seconds=time.perf_counter() - started,
                       recurrence_and_transfers_wall_seconds=recurrence_seconds,
                       backtrack_wall_seconds=time.perf_counter() - backtrack_started,
                       cuda_stream_elapsed_seconds=stream_seconds, device=str(selected), dtype='float64',
                       matrix_shape=list(local.shape), backtrack_tie_priority=['diagonal', 'up', 'left'])
    return float(host[-1, -1] / len(path)), matching, host
