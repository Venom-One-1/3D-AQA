#!/usr/bin/env python
"""Render tracked reference/student Qishi meshes on a shared transparent canvas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aqa3d.smpl_dtw import inspect_video_sampling
from aqa3d.tracking import DEFAULT_SMPL_MODEL_PATH, load_stitched_root_track


PROJECT = Path(__file__).resolve().parent
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size}


def segment_samples(sampling, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= start <= end < sampling.source_frame_count:
        raise ValueError("Segment is outside the source video.")
    indices = np.flatnonzero((sampling.source_indices >= start)
                            & (sampling.source_indices <= end))
    if not len(indices):
        raise ValueError("No samples in this segment.")
    return indices, sampling.source_indices[indices]


def yaw_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def fixed_heading_rotation(global_orients: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    # One constant yaw correction per clip; retain pitch, roll and subsequent turns.
    forward = np.einsum("ij,tjk,k->ti", CV_TO_GL, global_orients[:count], [0, 0, 1])
    heading = np.median(forward[:, [0, 2]], axis=0)
    if not np.isfinite(heading).all() or np.linalg.norm(heading) < 0.1:
        raise ValueError("Cannot establish the initial SMPL heading.")
    yaw = float(np.arctan2(heading[0], heading[1]))
    return yaw_rotation(-yaw) @ CV_TO_GL, yaw


def mesh_global_orients(global_orients: np.ndarray, view_mode: str) -> np.ndarray:
    if global_orients.ndim != 3 or global_orients.shape[1:] != (3, 3):
        raise ValueError("Expected global rotation matrices (T,3,3).")
    if view_mode == "canonical-front":
        return np.broadcast_to(np.eye(3), global_orients.shape).copy()
    if view_mode == "initial-heading":
        return global_orients.copy()
    raise ValueError(f"Unknown view mode: {view_mode}")


def configure_view(args) -> None:
    front = args.view_mode == "canonical-front"
    if args.azimuth is None:
        args.azimuth = 0.0 if front else 35.0
    if args.elevation is None:
        args.elevation = 0.0 if front else 8.0
    if front and (args.azimuth != 0.0 or args.elevation != 0.0):
        raise ValueError("canonical-front requires zero camera azimuth and elevation.")
    if args.output_root is None:
        name = "smpl_qishi_5fps_front" if front else "smpl_qishi_5fps"
        args.output_root = PROJECT / "report_assets" / name


def presentation_vertices(vertices: np.ndarray, pelvis: np.ndarray,
                          rotation: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid presentation scale.")
    centered = vertices - pelvis[:, None, :]
    result = (centered @ rotation.T) * scale
    floor = result[:, :, 1].min(axis=1)
    result[:, :, 1] -= floor[:, None]
    return result.astype(np.float32), floor


def shared_camera(vertices: list[np.ndarray], width: int, height: int,
                  azimuth: float, elevation: float) -> tuple[np.ndarray, float, float]:
    az, el = np.radians([azimuth, elevation])
    back = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    right = np.cross([0, 1, 0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    rotation = np.column_stack([right, up, back])
    bounds = [(v.reshape(-1, 3) @ rotation) for v in vertices]
    lo = np.min([v.min(axis=0) for v in bounds], axis=0)
    hi = np.max([v.max(axis=0) for v in bounds], axis=0)
    center = rotation @ ((lo + hi) / 2)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = center + 5.0 * back
    ymag = max((hi[1] - lo[1]) / 2, (hi[0] - lo[0]) * height / (2 * width)) * 1.14
    return pose, float(ymag * width / height), float(ymag)


def validate_rgba(image: Image.Image) -> dict:
    if image.mode != "RGBA":
        raise ValueError("Expected a transparent RGBA image.")
    alpha = np.asarray(image)[:, :, 3]
    ys, xs = np.nonzero(alpha > 8)
    if not len(xs) or not np.any(alpha == 0):
        raise ValueError("Blank image or missing transparent background.")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    width, height = image.size
    if min(bbox[0], bbox[1], width - bbox[2], height - bbox[3]) < 4:
        raise ValueError(f"Mesh touches the canvas border: {bbox}")
    return {"alpha_bbox_xyxy": bbox, "foreground_fraction": float(np.mean(alpha > 8)),
            "transparent_fraction": float(np.mean(alpha == 0))}


def load_subjects(args) -> list[dict]:
    manifest = read_json(args.reference_manifest)
    alignment = read_json(args.student_alignment)
    ref = manifest["reference"]
    endpoint = next(e for e in manifest["endpoint_keyposes"] if e["move_id"] == 1)
    student_end = next(e for e in alignment["endpoints"] if e["move_id"] == 1)
    if ref["video_id"] != "BV1WE411W7JB" or alignment["subject_id"] != "10":
        raise ValueError("This exporter expects reference BV1WE411W7JB and student 10.")
    subjects = [
        dict(subject_id=ref["video_id"], role="reference", video=ref["video_path"],
             tracking=ref["tracking_path"], source_fps=ref["source_fps"],
             start=endpoint["segment_source_start_index_0based"],
             end=endpoint["segment_source_end_index_0based"],
             endpoint_grid_time=endpoint["boundary_time_seconds"],
             boundary_provenance=file_record(args.reference_manifest)),
        dict(subject_id=alignment["subject_id"], role="student", video=alignment["input_video"],
             tracking=alignment["input_tracking"], source_fps=alignment["source_fps"],
             start=student_end["segment_start_source_frame_0based"],
             end=student_end["segment_end_source_frame_0based"],
             endpoint_grid_time=student_end["boundary_time_seconds"],
             boundary_provenance=file_record(args.student_alignment)),
    ]
    for subject in subjects:
        sampling = inspect_video_sampling(subject["video"], args.sample_fps)
        if not np.isclose(sampling.source_fps, subject["source_fps"], atol=1e-6, rtol=0):
            raise ValueError("Current source FPS differs from the boundary metadata.")
        sample_ids, frames = segment_samples(sampling, subject["start"], subject["end"])
        if frames[-1] != subject["end"]:
            raise ValueError("Requested sampling grid misses the frozen endpoint.")
        track = load_stitched_root_track(subject["tracking"])
        selected = track.at_source_frames(frames)
        for array in (selected.global_orients, selected.body_poses, selected.betas):
            if not np.isfinite(array).all():
                raise ValueError("Non-finite SMPL parameters.")
        subject.update(sampling=sampling, sample_ids=sample_ids, frames=frames, track=selected,
                       tracking_provenance=file_record(Path(subject["tracking"])))
        print(f"{subject['subject_id']}: {len(frames)} exact tracked samples, "
              f"frame0 {frames[0]}..{frames[-1]}", flush=True)
    return subjects


def reconstruct(subject: dict, smpl, args) -> None:
    import torch

    track = subject["track"]
    global_orients = mesh_global_orients(track.global_orients, args.view_mode)
    vertices, pelvis = [], []
    with torch.no_grad():
        for start in range(0, len(track.frame_numbers), 32):
            end = start + 32
            result = smpl(global_orient=torch.as_tensor(global_orients[start:end], dtype=torch.float32),
                          body_pose=torch.as_tensor(track.body_poses[start:end], dtype=torch.float32),
                          betas=torch.as_tensor(track.betas[start:end], dtype=torch.float32),
                          pose2rot=False)
            vertices.append(result.vertices.cpu().numpy())
            pelvis.append(result.joints[:, 0].cpu().numpy())
        rest = smpl(global_orient=torch.eye(3)[None],
                    body_pose=torch.eye(3).repeat(1, 23, 1, 1),
                    betas=torch.as_tensor(np.median(track.betas, axis=0)[None], dtype=torch.float32),
                    pose2rot=False).vertices[0].cpu().numpy()
    rest_height = float(np.ptp(rest[:, 1]))
    scale = args.canonical_height / rest_height
    if args.view_mode == "canonical-front":
        # Identity-root meshes already use SMPL model coordinates: Y up, Z forward.
        # Applying the PHALP-camera flip or the old yaw correction would undo this view.
        rotation, yaw = np.eye(3), None
        orientation_policy = "identity global_orient for every frame; no heading correction; original body_pose retained"
    else:
        rotation, yaw = fixed_heading_rotation(track.global_orients, max(1, int(args.sample_fps)))
        orientation_policy = "one initial yaw correction; preserve all framewise rotations and relative turns"
    rendered, floor = presentation_vertices(np.concatenate(vertices), np.concatenate(pelvis), rotation, scale)
    if not np.isfinite(rendered).all():
        raise ValueError("Non-finite mesh vertices.")
    subject.update(vertices=rendered, floor=floor, presentation={
        "view_mode": args.view_mode,
        "global_orient_policy": "identity_3x3_per_frame" if args.view_mode == "canonical-front" else "original_tracking",
        "mesh_to_presentation_rotation": rotation.tolist(),
        "initial_yaw_degrees": float(np.degrees(yaw)) if yaw is not None else None,
        "scale_fixed_for_clip": scale, "median_shape_rest_height": rest_height,
        "canonical_rest_height": args.canonical_height,
        "shape_policy": "original per-frame betas; median betas used ONLY to compute one fixed display scale",
        "orientation_policy": orientation_policy,
        "local_body_pose_policy": "original tracking matrices; no modification or smoothing",
        "translation_policy": "pelvis-centered X/Z, each mesh lowest vertex placed at Y=0; display only, not measured ground",
    })


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def contact_sheet(directory: Path, rows: list[dict], subject_id: str) -> None:
    columns, tw, th = 8, 160, 214
    cell_h, header = th + 42, 48
    sheet = Image.new("RGB", (columns * tw, header + ((len(rows) + columns - 1) // columns) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"{subject_id} | Qishi | 5 FPS | source timestamps", fill="#17232b", font=font(19))
    for i, row in enumerate(rows):
        with Image.open(directory / row["image"]) as original:
            thumb = original.resize((tw, th), Image.Resampling.LANCZOS)
        x, y = i % columns * tw, header + i // columns * cell_h
        sheet.paste(thumb, (x, y), thumb)
        draw.text((x + 7, y + th), f"{row['source_time_seconds']:.3f}s", fill="#17232b", font=font(13))
        draw.text((x + 7, y + th + 18), f"frame0 {row['source_frame_0based']}", fill="#536371", font=font(12))
    sheet.save(directory / "contact_sheet.jpg", quality=94)


def render(subjects: list[dict], faces: np.ndarray, args) -> dict:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    pose, xmag, ymag = shared_camera([s["vertices"] for s in subjects], args.width, args.height,
                                    args.azimuth, args.elevation)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.65098, 0.74118, 0.85882, 1.0], metallicFactor=0.0,
        roughnessFactor=0.65, alphaMode="OPAQUE")
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=0.05, zfar=30), pose=pose)
    for angle, intensity in ((-35, 2.3), (50, 1.1), (160, 0.8)):
        light_pose = pose.copy()
        light_pose[:3, :3] = yaw_rotation(np.radians(angle)) @ pose[:3, :3]
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity), pose=light_pose)
    renderer = pyrender.OffscreenRenderer(args.width * args.supersample, args.height * args.supersample)
    results = []
    try:
        for subject in subjects:
            directory = args.output_root / subject["subject_id"]
            directory.mkdir()
            rows = []
            count = len(subject["frames"])
            selected = sorted({0, int(0.75 * (count - 1)), count - 1}) if args.preview_only else range(count)
            for index in selected:
                mesh = trimesh.Trimesh(vertices=subject["vertices"][index], faces=faces, process=False)
                node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
                try:
                    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                finally:
                    scene.remove_node(node)
                picture = Image.fromarray(color, "RGBA")
                if args.supersample != 1:
                    picture = picture.resize((args.width, args.height), Image.Resampling.LANCZOS)
                check = validate_rgba(picture)
                pixels = np.asarray(picture)
                shade_std = float(pixels[pixels[:, :, 3] > 248, :3].astype(float).std(axis=0).mean())
                if not np.isfinite(shade_std) or shade_std < 2:
                    raise ValueError("Mesh lighting is flat; check directional light colors/intensities.")
                frame = int(subject["frames"][index])
                sample = int(subject["sample_ids"][index])
                grid_time = sample / args.sample_fps
                source_time = frame / subject["sampling"].source_fps
                name = f"qishi_{sample:05d}_t{grid_time:07.3f}_f{frame:06d}.png"
                picture.save(directory / name)
                rows.append({"image": name, "sample_index_0based": sample,
                             "grid_time_seconds": grid_time, "source_frame_0based": frame,
                             "phalp_frame_1based": frame + 1, "source_time_seconds": source_time,
                             "source_minus_grid_seconds": source_time - grid_time,
                             "source_track_id": int(subject["track"].source_track_ids[index]),
                             "tracking_status": "exact", "display_floor_offset": float(subject["floor"][index]),
                             "foreground_fraction": check["foreground_fraction"],
                             "transparent_fraction": check["transparent_fraction"],
                             "foreground_rgb_std": shade_std,
                             "alpha_bbox_xyxy": json.dumps(check["alpha_bbox_xyxy"]),
                             "is_move_endpoint": frame == subject["end"]})
                if len(rows) % 20 == 0 or frame == subject["end"]:
                    print(f"Rendered {subject['subject_id']}: {len(rows)}/{len(selected)}", flush=True)
            with (directory / "frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = {"subject_id": subject["subject_id"], "role": subject["role"],
                       "move_id": 1, "move_name": "qishi", "sample_fps": args.sample_fps,
                       "source_fps": subject["sampling"].source_fps,
                       "source_frame_count": subject["sampling"].source_frame_count,
                       "segment_start_frame_0based": subject["start"],
                       "segment_end_frame_0based": subject["end"],
                       "endpoint_grid_time_seconds": subject["endpoint_grid_time"],
                       "endpoint_source_time_seconds": subject["end"] / subject["sampling"].source_fps,
                       "rendered_frame_count": len(rows), "full_sample_count": count,
                       "source_video": str(Path(subject["video"]).resolve()),
                       "tracking": subject["tracking_provenance"],
                       "boundary_provenance": subject["boundary_provenance"],
                       "presentation": subject["presentation"],
                       "used_track_ids": list(subject["track"].used_track_ids),
                       "tracking_substitutions": 0, "image_size": [args.width, args.height],
                       "image_mode": "RGBA", "frames": rows}
            write_json(directory / "frames.json", summary)
            contact_sheet(directory, rows, subject["subject_id"])
            results.append({k: v for k, v in summary.items() if k != "frames"})
    finally:
        renderer.delete()
    return {"schema_version": "1.1", "status": "complete", "preview_only": args.preview_only,
            "view_mode": args.view_mode,
            "backend": "smplx.SMPLLayer + trimesh + pyrender/EGL; no generative image model",
            "sample_policy": "existing uniform_sample_source_indices; floor(k*source_fps/sample_fps); closed source-frame bounds",
            "model": file_record(args.model_path / "SMPL_NEUTRAL.pkl"),
            "camera": {"projection": "orthographic", "azimuth_degrees": args.azimuth,
                       "elevation_degrees": args.elevation, "pose": pose.tolist(),
                       "xmag": xmag, "ymag": ymag, "shared_for_all_frames_and_subjects": True},
            "canvas": {"width": args.width, "height": args.height, "supersampling": args.supersample,
                       "background": "transparent", "mesh_srgb": [0.65098, 0.74118, 0.85882]},
            "subjects": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path,
                        default=PROJECT / "reference_data/BV1WE411W7JB/reference_manifest.json")
    parser.add_argument("--student-alignment", type=Path,
                        default=PROJECT / "endpoint_metric_results/first3_five_students/student/10/alignment.json")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="New destination directory; default is selected by --view-mode.")
    parser.add_argument("--view-mode", choices=("initial-heading", "canonical-front"),
                        default="initial-heading",
                        help="Keep tracked root turns with initial yaw alignment, or render identity-root frontal poses.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SMPL_MODEL_PATH)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--azimuth", type=float, default=None,
                        help="Default 35 degrees for initial-heading, 0 for canonical-front.")
    parser.add_argument("--elevation", type=float, default=None,
                        help="Default 8 degrees for initial-heading, 0 for canonical-front.")
    parser.add_argument("--canonical-height", type=float, default=1.7)
    parser.add_argument("--preview-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_view(args)
    if args.sample_fps != 5.0 or min(args.width, args.height) < 128 or args.supersample < 1:
        raise ValueError("Expected 5 FPS, a canvas of at least 128 pixels and positive supersampling.")
    if not -80 < args.elevation < 80 or args.canonical_height <= 0:
        raise ValueError("Invalid camera elevation or canonical height.")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}; choose a new output directory.")
    import smplx
    import torch

    torch.set_num_threads(4)
    subjects = load_subjects(args)
    smpl = smplx.SMPLLayer(model_path=str(args.model_path), gender="neutral", num_betas=10).eval()
    for subject in subjects:
        reconstruct(subject, smpl, args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    result = render(subjects, smpl.faces, args)
    result["exporter"] = file_record(Path(__file__))
    write_json(args.output_root / "render_manifest.json", result)
    print(f"Saved {sum(s['rendered_frame_count'] for s in result['subjects'])} transparent PNGs to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
