"""Numerically stable SO(3) geodesic-distance utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert arrays ending in a 3D axis-angle vector to rotation matrices."""
    vectors = np.asarray(axis_angle, dtype=np.float64)
    if vectors.shape[-1] != 3:
        raise ValueError("Axis-angle input must have a final dimension of 3.")

    theta_squared = np.sum(vectors * vectors, axis=-1, keepdims=True)
    theta = np.sqrt(theta_squared)
    # These Taylor expansions keep the zero-rotation case well defined.
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    nonzero = theta_squared > 1e-16
    np.divide(np.sin(theta), theta, out=a, where=nonzero)
    np.divide(1.0 - np.cos(theta), theta_squared, out=b, where=nonzero)
    a[~nonzero] = (1.0 - theta_squared / 6.0)[~nonzero]
    b[~nonzero] = (0.5 - theta_squared / 24.0)[~nonzero]

    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack(
        (
            np.stack((zero, -z, y), axis=-1),
            np.stack((z, zero, -x), axis=-1),
            np.stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )
    identity = np.eye(3, dtype=np.float64)
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


def as_rotation_matrices(rotations: np.ndarray | Sequence[float]) -> np.ndarray:
    """Accept rotation matrices ``(..., 3, 3)`` or axis-angles ``(..., 3)``."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("Rotation input cannot be scalar.")
    if values.ndim >= 2 and values.shape[-2:] == (3, 3):
        return values
    return _axis_angle_to_matrix(values)


def geodesic_distance(
    student_rotations: np.ndarray | Sequence[float],
    teacher_rotations: np.ndarray | Sequence[float],
    *,
    frame_weights: np.ndarray | Sequence[float] | None = None,
    joint_weights: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-joint SO(3) errors and their weighted mean in radians.

    The two inputs have shape ``(..., T, J, 3, 3)`` (or the equivalent
    axis-angle shape ``(..., T, J, 3)``).  Leading dimensions are treated as
    batch dimensions.  ``frame_weights`` and ``joint_weights`` are shared by
    every batch item.
    """
    student = as_rotation_matrices(student_rotations)
    teacher = as_rotation_matrices(teacher_rotations)
    if student.shape != teacher.shape:
        raise ValueError(
            "Student and teacher rotations must have identical shapes; "
            f"got {student.shape} and {teacher.shape}."
        )
    if student.ndim < 4 or student.shape[-2:] != (3, 3):
        raise ValueError("Expected rotations shaped (..., T, J, 3, 3).")

    relative = student @ np.swapaxes(teacher, -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    errors = np.arccos(np.clip(cosine, -1.0, 1.0))

    frame_count, joint_count = errors.shape[-2:]
    if frame_weights is None:
        frame_weights_array = np.ones(frame_count, dtype=np.float64)
    else:
        frame_weights_array = np.asarray(frame_weights, dtype=np.float64)
        if frame_weights_array.shape != (frame_count,):
            raise ValueError(f"frame_weights must have shape ({frame_count},).")

    if joint_weights is None:
        joint_weights_array = np.ones(joint_count, dtype=np.float64)
    else:
        joint_weights_array = np.asarray(joint_weights, dtype=np.float64)
        if joint_weights_array.shape != (joint_count,):
            raise ValueError(f"joint_weights must have shape ({joint_count},).")

    if np.any(frame_weights_array < 0) or np.any(joint_weights_array < 0):
        raise ValueError("Frame and joint weights must be non-negative.")
    weights = frame_weights_array[..., :, None] * joint_weights_array
    normalizer = weights.sum()
    if normalizer <= 0:
        raise ValueError("At least one frame and one joint must have positive weight.")

    # Add leading singleton dimensions so weights broadcast over a batch.
    batch_shape = (1,) * (errors.ndim - 2)
    weighted_mean = np.sum(errors * weights.reshape(batch_shape + weights.shape), axis=(-2, -1))
    return errors, weighted_mean / normalizer
