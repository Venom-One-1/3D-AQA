# Student Maximum-FPS GPU Transfer

This entry adds student-specific native-clock/PTS handling without changing the
verified teacher GPU implementation or existing 5 FPS results.

```bash
conda activate 4d-humans
python -B run_fastsam_student_endpoints_gpu.py \
  --device cuda:0 \
  --output-root /home/sqw/tmp/fastsam_student_endpoints_gpu_maxfps_new
```

The output directory must be new. `--student-id <ID>` selects a subset and
`--preflight-only` verifies source checksums and exact sampling maps without DTW.

## Sampling

The common nominal FPS is bounded by both videos and both native SMPL rates,
just as in the teacher GPU entry. Reference annotations are retained as exact
source frames even if outside a fractional-rate reference grid. Student samples
are selected on the continuous `source_sample_indices` clock and mapped through
`output_video_frame_indices`; they are not inferred by multiplying video FPS.

For student 05 the native rate is 30 FPS, with all 11,004 native poses retained.
Its six skipped video frames are expected reconstruction sampling, not missing
poses. Other videos can have nonuniform PTS and slightly different original,
trimmed-average, and nominal native rates. The common-rate cap can omit a few
native rows through ordinary sampling, but never changes or repairs the source
array. Native index gaps or unusable selected poses fail explicitly.

Every student endpoint and path point records the actual cropped-video PTS,
cropped frame, source NPZ row, and original-video frame/timestamp provenance.
Video timelines and all native/output frame maps are saved separately. Automatic
endpoints are not rounded to the 5 FPS review grid. Exact original SMPL parameters
are exported without interpolation, smoothing, orientation changes, or nearest
pose substitution. Dense cost matrices are not saved to avoid exhausting disk;
full paths, candidate costs, sampling maps, source/code hashes, and timings are.

## Four-Version Review

`run_fastsam_endpoint_review_app_v3.py` inherits the v2 UI and review behavior,
adding only the `student_gpu` result dataset. The three storage groups are:

- `review_progress.json`: existing student/teacher 5 FPS annotations.
- `review_progress_gpu_maxfps.json`: existing maximum-FPS teacher annotations.
- `review_progress_student_gpu_maxfps.json`: new maximum-FPS student annotations.

The files must be distinct. Existing state files and fingerprints are read as-is,
without migrations or copying old annotations into the new version. Batch review,
0.2-second navigation, exact automatic frames, mm:ss, CSV exports, and revision/
file-lock protections are inherited from v2. The old v1/v2 entries remain usable.

```bash
bash /home/sqw/tmp/fastsam_endpoint_review_20261111_v1/run_review_v3.sh
```

Use the original port-forwarding connection and `http://127.0.0.1:8509`. Do not
start a second server while that port is occupied. Refresh after deployment and
select the new student maximum-FPS GPU option. All new automatic endpoints start
pending human review independently of the old version.
