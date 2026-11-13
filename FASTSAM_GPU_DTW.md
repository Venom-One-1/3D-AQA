# Separate GPU Geodesic DTW

This implementation adds new files only. The existing CPU entry points, source
videos, source SMPL parameters, 5 FPS results, and human annotations are unchanged.

## Entry Points

- `aqa3d/smpl_dtw_gpu.py`: float64 CUDA distances and anti-diagonal DTW.
- `aqa3d/fastsam_high_fps.py`: pair-specific maximum common rate sampling.
- `benchmark_smpl_dtw_gpu.py`: unchanged CPU / CUDA comparison on real pose subsets.
- `run_fastsam_teacher_endpoints_gpu.py`: maximum-rate teacher transfer with timings.

```bash
conda activate 4d-humans
python -B benchmark_smpl_dtw_gpu.py \
  --frames 512 --repeats 3 --device cuda:0 \
  --output-root /home/sqw/tmp/fastsam_gpu_validation_new

python -B run_fastsam_teacher_endpoints_gpu.py \
  --device cuda:0 \
  --output-root /home/sqw/tmp/fastsam_teacher_endpoints_gpu_maxfps_new
```

Output directories must not already exist. `--teacher-id <video_id>` selects a
subset in a separate directory. `--preflight-only` checks the existing cohort
without running CUDA. `--dtw-backend cpu` retains the original CPU recurrence
while still using CUDA distances. CUDA is required; there is no silent fallback.

## Numerical Contract

Distances are the mean of 23 local SO(3) geodesic distances in radians, excluding
global orientation. `trace(A @ B.T)` is computed as the equivalent Frobenius dot
product, followed by the original cosine clamp and arccos. No float32/TF32 path,
approximate DTW band, pose interpolation, smoothing, or rotation repair is used.
Floating point distance values can differ slightly because summation order and
CUDA arccos differ from NumPy. Equivalence is checked with explicit tolerances,
and paths and selected endpoints are checked separately.

The GPU recurrence processes independent anti-diagonals. Initialization and
coefficient semantics match the legacy implementation, including its edge
initialization order. Backtracking reuses the original diagonal/up/left tie
priority; endpoint candidates still use the minimum local geodesic distance.

## Sampling and Timestamps

The common rate is the minimum of both videos' rates and both SMPL native rates.
This teacher entry requires the verified complete exact-frame SMPL timelines;
it rejects missing poses rather than substituting neighbors. Fractional rates
such as 24000/1001 are retained, not rounded to 24 or 30 FPS.

The reference manifest remains the original 5 FPS annotation record. Its source
frames and timestamps are not changed. For a lower/fractional common rate, any
annotated reference frame absent from the regular grid is additionally included
in the reference sequence. Thus the reference sequence can be slightly
nonuniform. All exported path and endpoint times use actual source frames and
verified source video FPS, never sequence index divided by nominal sampling FPS.

Automatic endpoints keep their exact high-rate source frames and are not rounded
to the 5 FPS review grid. `review_fps=5` is separate presentation metadata; this
entry does not modify or restart the existing review app or its annotations.

## Timing and Storage

Each teacher has separate load/sampling, distance-matrix, DTW, export, and total
wall times. CUDA stages synchronize before timings finish, include host/device
transfers, and report stream elapsed time separately. Stream elapsed time includes
host launch gaps and is not a pure-kernel duration. Batch wall time also includes
preflight hashing, shared reference setup, CUDA warmup, and final source checks.
Small-subset speedup is not claimed as full-video or end-to-end speedup.

`timing_summary.csv` and `batch_summary.json` summarize all pairs. Full-resolution
DTW paths, sampling frame maps, exact endpoint SMPL subsets, paired screenshots,
and source/code provenance are always saved. Dense local cost matrices are only
saved with `--save-local-costs`; ten high-rate matrices can exceed available disk
space. The diagnostic heatmap is downsampled for display only.

## Tests

```bash
python -B -m unittest discover -s tests -p test_smpl_dtw_gpu.py -v
python -B -m unittest discover -s tests -p test_fastsam_high_fps.py -v
```

CUDA equivalence tests are explicitly skipped when CUDA is unavailable; a skipped
GPU test is not counted as GPU verification. No GPU dependencies are installed
or changes made to the existing environment by these scripts.
