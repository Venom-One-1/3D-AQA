"""Read the PHALP/4D-Humans tracking output used by this project."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np


DEFAULT_SMPL_MODEL_PATH = Path.home() / ".cache" / "4DHumans" / "data" / "smpl"


@dataclass(frozen=True)
class TrackPoseSequence:
    """SMPL local rotations indexed by PHALP's one-based image frame number."""

    frame_numbers: np.ndarray
    body_poses: np.ndarray
    track_id: int
    source_track_ids: np.ndarray | None = None

    @property
    def used_track_ids(self) -> tuple[int, ...]:
        if self.source_track_ids is None:
            return (self.track_id,)
        return tuple(sorted(int(value) for value in np.unique(self.source_track_ids)))

    def at_source_frames(
        self,
        source_frame_indices: Iterable[int],
        *,
        nearest_max_distance_frames: int = 0,
    ) -> np.ndarray:
        """Return poses for zero-based source-video frame indices.

        PHALP image keys start at ``000001.jpg`` while OpenCV/AQA frame indices
        start at zero, hence the required one-frame offset.
        """
        lookup = {int(number): pose for number, pose in zip(self.frame_numbers, self.body_poses)}
        requested = np.asarray(list(source_frame_indices), dtype=np.int64)
        tracking_frames = requested + 1
        selected_frames: list[int] = []
        available = self.frame_numbers.astype(np.int64)
        for frame in tracking_frames:
            requested_frame = int(frame)
            if requested_frame in lookup:
                selected_frames.append(requested_frame)
                continue
            if nearest_max_distance_frames <= 0:
                selected_frames.append(requested_frame)
                continue
            insertion = int(np.searchsorted(available, requested_frame))
            candidates = available[max(insertion - 1, 0) : min(insertion + 1, len(available))]
            nearest = int(candidates[np.argmin(np.abs(candidates - requested_frame))])
            selected_frames.append(
                nearest
                if abs(nearest - requested_frame) <= nearest_max_distance_frames
                else requested_frame
            )
        missing = [frame for frame in selected_frames if frame not in lookup]
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            suffix = "..." if len(missing) > 10 else ""
            raise KeyError(
                "No valid tracked SMPL pose for PHALP frame(s) "
                f"{preview}{suffix}. The video and tracking result may not match."
            )
        return np.stack([lookup[frame] for frame in selected_frames], axis=0)


@dataclass(frozen=True)
class TrackSmplSequence:
    """SMPL parameters indexed by PHALP's one-based image frame number."""

    frame_numbers: np.ndarray
    global_orients: np.ndarray
    body_poses: np.ndarray
    betas: np.ndarray
    track_id: int

    def at_source_frames(self, source_frame_indices: Iterable[int]) -> "TrackSmplSequence":
        """Return SMPL parameters for zero-based source-video frame indices."""
        lookup = {int(number): index for index, number in enumerate(self.frame_numbers)}
        requested = np.asarray(list(source_frame_indices), dtype=np.int64)
        tracking_frames = requested + 1
        missing = [int(frame) for frame in tracking_frames if int(frame) not in lookup]
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            suffix = "..." if len(missing) > 10 else ""
            raise KeyError(
                "No valid tracked SMPL parameters for PHALP frame(s) "
                f"{preview}{suffix}. The video and tracking result may not match."
            )
        indices = np.asarray([lookup[int(frame)] for frame in tracking_frames], dtype=np.int64)
        return TrackSmplSequence(
            frame_numbers=tracking_frames,
            global_orients=self.global_orients[indices],
            body_poses=self.body_poses[indices],
            betas=self.betas[indices],
            track_id=self.track_id,
        )

    def to_smpl24_joints(
        self,
        *,
        model_path: str | Path = DEFAULT_SMPL_MODEL_PATH,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> np.ndarray:
        """Regress native SMPL 24 joints from PHALP SMPL parameters.

        ``smplx.SMPLLayer`` returns 45 joints for this model file: the first 24
        are the native SMPL joints, followed by extra joints. This method keeps
        only the first 24 so the result matches ``aqa3d.angle_metrics``.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        model_dir = Path(model_path).expanduser()
        if not model_dir.exists():
            raise FileNotFoundError(f"SMPL model path does not exist: {model_dir}")

        return _regress_smpl24_joints(
            self.global_orients,
            self.body_poses,
            self.betas,
            model_dir,
            batch_size,
            device,
        )


@dataclass(frozen=True)
class TrackRootSequence:
    """PHALP parameters, model-specific joints and camera translation."""

    frame_numbers: np.ndarray
    global_orients: np.ndarray
    body_poses: np.ndarray
    betas: np.ndarray
    joints: np.ndarray
    camera_translations: np.ndarray
    track_id: int
    source_track_ids: np.ndarray

    @property
    def used_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in np.unique(self.source_track_ids)))

    def at_source_frames(self, source_frame_indices: Iterable[int]) -> "TrackRootSequence":
        """Return root-track values for zero-based source-video frame indices."""
        lookup = {int(number): index for index, number in enumerate(self.frame_numbers)}
        requested = np.asarray(list(source_frame_indices), dtype=np.int64)
        tracking_frames = requested + 1
        missing = [int(frame) for frame in tracking_frames if int(frame) not in lookup]
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            suffix = "..." if len(missing) > 10 else ""
            raise KeyError(
                "No valid tracked root data for PHALP frame(s) "
                f"{preview}{suffix}. The video and tracking result may not match."
            )
        indices = np.asarray([lookup[int(frame)] for frame in tracking_frames], dtype=np.int64)
        return TrackRootSequence(
            frame_numbers=tracking_frames,
            global_orients=self.global_orients[indices],
            body_poses=self.body_poses[indices],
            betas=self.betas[indices],
            joints=self.joints[indices],
            camera_translations=self.camera_translations[indices],
            track_id=self.track_id,
            source_track_ids=self.source_track_ids[indices],
        )

    @property
    def camera_space_joints(self) -> np.ndarray:
        """Return PHALP's model-specific joints translated into camera space."""
        return self.joints + self.camera_translations[:, None, :]

    def to_smpl24_joints(
        self,
        *,
        model_path: str | Path = DEFAULT_SMPL_MODEL_PATH,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> np.ndarray:
        """Regress native SMPL-24 joints instead of using PHALP's reordered joints."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        model_dir = Path(model_path).expanduser()
        if not model_dir.exists():
            raise FileNotFoundError(f"SMPL model path does not exist: {model_dir}")
        return _regress_smpl24_joints(
            self.global_orients,
            self.body_poses,
            self.betas,
            model_dir,
            batch_size,
            device,
        )


@lru_cache(maxsize=4)
def _cached_smpl_layer(model_path: str, device: str):
    import smplx

    layer = smplx.SMPLLayer(model_path=model_path, gender="neutral", num_betas=10).to(device)
    layer.eval()
    return layer


def _regress_smpl24_joints(
    global_orients: np.ndarray,
    body_poses: np.ndarray,
    betas: np.ndarray,
    model_path: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    smpl = _cached_smpl_layer(str(model_path), device)
    joints: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(global_orients), batch_size):
            end = min(start + batch_size, len(global_orients))
            output = smpl(
                global_orient=torch.as_tensor(
                    global_orients[start:end],
                    dtype=torch.float32,
                    device=device,
                ),
                body_pose=torch.as_tensor(
                    body_poses[start:end],
                    dtype=torch.float32,
                    device=device,
                ),
                betas=torch.as_tensor(
                    betas[start:end],
                    dtype=torch.float32,
                    device=device,
                ),
                pose2rot=False,
            )
            if output.joints.shape[1] < 24:
                raise RuntimeError(
                    f"SMPLLayer returned only {output.joints.shape[1]} joints; expected at least 24."
                )
            joints.append(
                output.joints[:, :24, :].detach().cpu().numpy().astype(np.float64)
            )
    return np.concatenate(joints, axis=0)


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


def _global_orient_matrix(smpl: dict) -> np.ndarray:
    orient = np.asarray(smpl["global_orient"], dtype=np.float64)
    if orient.shape == (1, 1, 3, 3):
        orient = orient[0, 0]
    elif orient.shape == (1, 3, 3):
        orient = orient[0]
    if orient.shape != (3, 3):
        raise ValueError(f"Expected PHALP global_orient shape (3, 3), got {orient.shape}.")
    return orient


def _betas(smpl: dict) -> np.ndarray:
    betas = np.asarray(smpl["betas"], dtype=np.float64)
    if betas.shape == (1, 10):
        betas = betas[0]
    if betas.shape != (10,):
        raise ValueError(f"Expected PHALP betas shape (10,), got {betas.shape}.")
    return betas


def _camera_translation(value: object) -> np.ndarray:
    camera = np.asarray(value, dtype=np.float64)
    if camera.shape == (1, 3):
        camera = camera[0]
    if camera.shape != (3,):
        raise ValueError(f"Expected PHALP camera shape (3,), got {camera.shape}.")
    return camera


def _phalp_joints(value: object) -> np.ndarray:
    joints = np.asarray(value, dtype=np.float64)
    if joints.shape == (1, 45, 3):
        joints = joints[0]
    if joints.ndim != 2 or joints.shape[0] < 24 or joints.shape[1] != 3:
        raise ValueError(f"Expected PHALP joints shape (J>=24, 3), got {joints.shape}.")
    return joints


def _extract_smpl_candidates(data: dict) -> list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    candidates: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for key, record in data.items():
        tids = record.get("tid", [])
        smpls = record.get("smpl", [])
        for tid, smpl in zip(tids, smpls):
            try:
                candidates.append(
                    (
                        int(tid),
                        _frame_number(key),
                        _global_orient_matrix(smpl),
                        _body_pose_matrix(smpl),
                        _betas(smpl),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return candidates


def _extract_root_candidates(
    data: dict,
) -> list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    candidates: list[
        tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for key, record in data.items():
        tids = record.get("tid", [])
        smpls = record.get("smpl", [])
        cameras = record.get("camera", [])
        joints = record.get("3d_joints", [])
        for tid, smpl, camera, frame_joints in zip(tids, smpls, cameras, joints):
            try:
                candidates.append(
                    (
                        int(tid),
                        _frame_number(key),
                        _global_orient_matrix(smpl),
                        _body_pose_matrix(smpl),
                        _betas(smpl),
                        _phalp_joints(frame_joints),
                        _camera_translation(camera),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return candidates


def _select_track_id(candidates: Iterable[tuple], track_id: int | None) -> int:
    available = Counter(item[0] for item in candidates)
    if not available:
        raise ValueError("No valid SMPL entries found.")
    return int(track_id if track_id is not None else min(available, key=lambda tid: (-available[tid], tid)))


def load_primary_track(tracking_path: str | Path, track_id: int | None = None) -> TrackPoseSequence:
    """Load one complete PHALP track, selecting the longest track by default."""
    path = Path(tracking_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking result does not exist: {path}")

    data = joblib.load(path)
    smpl_candidates = _extract_smpl_candidates(data)
    candidates = [(tid, frame, body_pose) for tid, frame, _, body_pose, _ in smpl_candidates]

    if not candidates:
        raise ValueError(f"No valid SMPL body_pose entries found in {path}.")
    selected_id = _select_track_id(candidates, track_id)
    selected = [(frame, pose) for tid, frame, pose in candidates if tid == selected_id]
    if not selected:
        available = Counter(item[0] for item in candidates)
        raise ValueError(f"Track id {selected_id} is not available in {path}; available: {sorted(available)}")

    selected.sort(key=lambda item: item[0])
    frames = np.asarray([frame for frame, _ in selected], dtype=np.int64)
    poses = np.stack([pose for _, pose in selected], axis=0)
    return TrackPoseSequence(
        frame_numbers=frames,
        body_poses=poses,
        track_id=int(selected_id),
        source_track_ids=np.full(len(frames), selected_id, dtype=np.int64),
    )


def _pose_continuity_cost(previous: np.ndarray, current: np.ndarray) -> float:
    relative = current @ np.swapaxes(previous, -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)).mean())


def load_stitched_primary_track(tracking_path: str | Path, track_id: int | None = None) -> TrackPoseSequence:
    """Load the main person while stitching contiguous PHALP track-ID changes.

    The longest track is used initially. It remains selected while present; if
    it disappears, the candidate with the smallest local-pose geodesic change
    from the preceding frame becomes the new active track. The per-frame IDs
    remain available through ``TrackPoseSequence.source_track_ids``.
    """
    path = Path(tracking_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking result does not exist: {path}")

    smpl_candidates = _extract_smpl_candidates(joblib.load(path))
    candidates = [(tid, frame, body_pose) for tid, frame, _, body_pose, _ in smpl_candidates]
    if not candidates:
        raise ValueError(f"No valid SMPL body_pose entries found in {path}.")
    primary_id = _select_track_id(candidates, track_id)
    counts = Counter(item[0] for item in candidates)
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for candidate_id, frame, pose in candidates:
        by_frame.setdefault(int(frame), []).append((int(candidate_id), pose))

    current_id = primary_id
    previous_pose: np.ndarray | None = None
    selected: list[tuple[int, int, np.ndarray]] = []
    for frame in sorted(by_frame):
        options = by_frame[frame]
        active = next((item for item in options if item[0] == current_id), None)
        if active is None:
            if previous_pose is None:
                active = min(options, key=lambda item: (-counts[item[0]], item[0]))
            else:
                active = min(
                    options,
                    key=lambda item: (_pose_continuity_cost(previous_pose, item[1]), -counts[item[0]], item[0]),
                )
            current_id = int(active[0])
        selected.append((frame, int(active[0]), active[1]))
        previous_pose = active[1]

    return TrackPoseSequence(
        frame_numbers=np.asarray([frame for frame, _, _ in selected], dtype=np.int64),
        body_poses=np.stack([pose for _, _, pose in selected], axis=0),
        track_id=int(primary_id),
        source_track_ids=np.asarray([candidate_id for _, candidate_id, _ in selected], dtype=np.int64),
    )


def load_primary_smpl_track(tracking_path: str | Path, track_id: int | None = None) -> TrackSmplSequence:
    """Load PHALP SMPL parameters for one track, selecting the longest track by default."""
    path = Path(tracking_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking result does not exist: {path}")

    candidates = _extract_smpl_candidates(joblib.load(path))
    if not candidates:
        raise ValueError(f"No valid SMPL parameter entries found in {path}.")
    selected_id = _select_track_id(candidates, track_id)
    selected = [
        (frame, global_orient, body_pose, betas)
        for tid, frame, global_orient, body_pose, betas in candidates
        if tid == selected_id
    ]
    if not selected:
        available = Counter(item[0] for item in candidates)
        raise ValueError(f"Track id {selected_id} is not available in {path}; available: {sorted(available)}")

    selected.sort(key=lambda item: item[0])
    return TrackSmplSequence(
        frame_numbers=np.asarray([frame for frame, _, _, _ in selected], dtype=np.int64),
        global_orients=np.stack([global_orient for _, global_orient, _, _ in selected], axis=0),
        body_poses=np.stack([body_pose for _, _, body_pose, _ in selected], axis=0),
        betas=np.stack([betas for _, _, _, betas in selected], axis=0),
        track_id=int(selected_id),
    )


def load_stitched_root_track(
    tracking_path: str | Path,
    track_id: int | None = None,
) -> TrackRootSequence:
    """Load joints and camera translation while stitching main-person ID changes."""
    path = Path(tracking_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking result does not exist: {path}")

    candidates = _extract_root_candidates(joblib.load(path))
    if not candidates:
        raise ValueError(f"No valid SMPL joint/camera entries found in {path}.")
    primary_id = _select_track_id(candidates, track_id)
    counts = Counter(item[0] for item in candidates)
    by_frame: dict[int, list[tuple]] = {}
    for candidate in candidates:
        by_frame.setdefault(int(candidate[1]), []).append(candidate)

    current_id = primary_id
    previous_pose: np.ndarray | None = None
    selected: list[tuple] = []
    for frame in sorted(by_frame):
        options = by_frame[frame]
        active = next((item for item in options if item[0] == current_id), None)
        if active is None:
            if previous_pose is None:
                active = min(options, key=lambda item: (-counts[item[0]], item[0]))
            else:
                active = min(
                    options,
                    key=lambda item: (
                        _pose_continuity_cost(previous_pose, item[3]),
                        -counts[item[0]],
                        item[0],
                    ),
                )
            current_id = int(active[0])
        selected.append(active)
        previous_pose = active[3]

    return TrackRootSequence(
        frame_numbers=np.asarray([item[1] for item in selected], dtype=np.int64),
        global_orients=np.stack([item[2] for item in selected], axis=0),
        body_poses=np.stack([item[3] for item in selected], axis=0),
        betas=np.stack([item[4] for item in selected], axis=0),
        joints=np.stack([item[5] for item in selected], axis=0),
        camera_translations=np.stack([item[6] for item in selected], axis=0),
        track_id=int(primary_id),
        source_track_ids=np.asarray([item[0] for item in selected], dtype=np.int64),
    )
