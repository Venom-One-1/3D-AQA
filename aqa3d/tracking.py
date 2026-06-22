"""Read the PHALP/4D-Humans tracking output used by this project."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np


@dataclass(frozen=True)
class TrackPoseSequence:
    """SMPL local rotations indexed by PHALP's one-based image frame number."""

    frame_numbers: np.ndarray
    body_poses: np.ndarray
    track_id: int

    def at_source_frames(self, source_frame_indices: Iterable[int]) -> np.ndarray:
        """Return poses for zero-based source-video frame indices.

        PHALP image keys start at ``000001.jpg`` while OpenCV/AQA frame indices
        start at zero, hence the required one-frame offset.
        """
        lookup = {int(number): pose for number, pose in zip(self.frame_numbers, self.body_poses)}
        requested = np.asarray(list(source_frame_indices), dtype=np.int64)
        tracking_frames = requested + 1
        missing = [int(frame) for frame in tracking_frames if int(frame) not in lookup]
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            suffix = "..." if len(missing) > 10 else ""
            raise KeyError(
                "No valid tracked SMPL pose for PHALP frame(s) "
                f"{preview}{suffix}. The video and tracking result may not match."
            )
        return np.stack([lookup[int(frame)] for frame in tracking_frames], axis=0)


def _frame_number(key: str) -> int:
    try:
        return int(Path(key).stem)
    except ValueError as error:
        raise ValueError(f"Cannot obtain a numeric frame number from tracking key: {key}") from error


def _body_pose_matrix(smpl: dict) -> np.ndarray:
    pose = np.asarray(smpl["body_pose"], dtype=np.float64)
    if pose.shape == (1, 23, 3, 3):
        pose = pose[0]
    if pose.shape != (23, 3, 3):
        raise ValueError(f"Expected PHALP body_pose shape (23, 3, 3), got {pose.shape}.")
    return pose


def load_primary_track(tracking_path: str | Path, track_id: int | None = None) -> TrackPoseSequence:
    """Load one complete PHALP track, selecting the longest track by default."""
    path = Path(tracking_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking result does not exist: {path}")

    data = joblib.load(path)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for key, record in data.items():
        tids = record.get("tid", [])
        smpls = record.get("smpl", [])
        for tid, smpl in zip(tids, smpls):
            try:
                candidates.append((int(tid), _frame_number(key), _body_pose_matrix(smpl)))
            except (KeyError, TypeError, ValueError):
                continue

    if not candidates:
        raise ValueError(f"No valid SMPL body_pose entries found in {path}.")
    available = Counter(item[0] for item in candidates)
    selected_id = track_id if track_id is not None else min(available, key=lambda tid: (-available[tid], tid))
    selected = [(frame, pose) for tid, frame, pose in candidates if tid == selected_id]
    if not selected:
        raise ValueError(f"Track id {selected_id} is not available in {path}; available: {sorted(available)}")

    selected.sort(key=lambda item: item[0])
    frames = np.asarray([frame for frame, _ in selected], dtype=np.int64)
    poses = np.stack([pose for _, pose in selected], axis=0)
    return TrackPoseSequence(frame_numbers=frames, body_poses=poses, track_id=int(selected_id))
