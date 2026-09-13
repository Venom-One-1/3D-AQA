#!/usr/bin/env python
"""Run HMR2 image inference for all manually selected Tai Chi KeyPose images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from aqa3d.keypose_alignment import KeyPoseImage, discover_keypose_images


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FOUR_D_HUMANS_ROOT = Path("/home/sqw/Projects/4D-Humans")
DEFAULT_KEYPOSE_ROOT = Path("/home/sqw/VisualSearch/aqa/FrameData/KeyPose")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/sqw/VisualSearch/aqa/keypose_alignment_results/i8kMrJmAfjU"
)


def _load_hmr2_dependencies(four_d_humans_root: Path):
    sys.path.insert(0, str(four_d_humans_root))
    from detectron2.config import LazyConfig
    import hmr2
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
    from hmr2.utils import recursive_to
    from hmr2.utils.renderer import Renderer, cam_crop_to_full
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

    return {
        "LazyConfig": LazyConfig,
        "hmr2": hmr2,
        "CACHE_DIR_4DHUMANS": CACHE_DIR_4DHUMANS,
        "ViTDetDataset": ViTDetDataset,
        "DEFAULT_CHECKPOINT": DEFAULT_CHECKPOINT,
        "download_models": download_models,
        "load_hmr2": load_hmr2,
        "recursive_to": recursive_to,
        "Renderer": Renderer,
        "cam_crop_to_full": cam_crop_to_full,
        "DefaultPredictor_Lazy": DefaultPredictor_Lazy,
    }


def _prediction_path(output_root: Path, item: KeyPoseImage) -> Path:
    return output_root / "smpl_predictions" / item.relative_path.with_suffix(".npz")


def _overlay_path(output_root: Path, item: KeyPoseImage) -> Path:
    return output_root / "keypose_reconstruction_overlays" / item.relative_path


def _load_saved_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _render_overlay(
    image: np.ndarray,
    vertices: np.ndarray,
    camera_translation: np.ndarray,
    renderer,
    focal_length: float,
) -> np.ndarray:
    rgba = renderer.render_rgba_multiple(
        [vertices],
        cam_t=[camera_translation],
        render_res=np.asarray([image.shape[1], image.shape[0]]),
        mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
        scene_bg_color=(1, 1, 1),
        focal_length=float(focal_length),
    )
    rgb = image.astype(np.float32)[:, :, ::-1] / 255.0
    overlay = rgb * (1.0 - rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
    return np.clip(255.0 * overlay[:, :, ::-1], 0, 255).astype(np.uint8)


def _configure_detector(dependencies: dict, score_threshold: float):
    cfg_path = (
        Path(dependencies["hmr2"].__file__).parent
        / "configs"
        / "cascade_mask_rcnn_vitdet_h_75ep.py"
    )
    cfg = dependencies["LazyConfig"].load(str(cfg_path))
    cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for predictor in cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = min(score_threshold, 0.25)
    return dependencies["DefaultPredictor_Lazy"](cfg)


def _infer_one(
    item: KeyPoseImage,
    *,
    model,
    model_cfg,
    detector,
    renderer,
    dependencies: dict,
    device: torch.device,
    score_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    image = cv2.imread(str(item.image_path))
    if image is None:
        raise ValueError(f"Cannot read KeyPose image: {item.image_path}")

    detection = detector(image)["instances"]
    valid = (detection.pred_classes == 0) & (detection.scores >= score_threshold)
    boxes = detection.pred_boxes.tensor[valid].detach().cpu().numpy()
    scores = detection.scores[valid].detach().cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError(
            f"No person detected above score {score_threshold:.2f}: {item.image_path}"
        )
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    selected = int(np.argmax(areas))
    box = boxes[[selected]]

    dataset = dependencies["ViTDetDataset"](model_cfg, image, box)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)))
    batch = dependencies["recursive_to"](batch, device)
    with torch.no_grad():
        output = model(batch)

    params = output["pred_smpl_params"]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    image_size = batch["img_size"].float()
    scaled_focal = (
        model_cfg.EXTRA.FOCAL_LENGTH
        / model_cfg.MODEL.IMAGE_SIZE
        * image_size.max()
    )
    full_camera = dependencies["cam_crop_to_full"](
        output["pred_cam"],
        box_center,
        box_size,
        image_size,
        scaled_focal,
    )[0].detach().cpu().numpy()
    vertices = output["pred_vertices"][0].detach().cpu().numpy()
    prediction = {
        "body_pose": params["body_pose"][0].detach().cpu().numpy(),
        "global_orient": params["global_orient"][0].detach().cpu().numpy(),
        "betas": params["betas"][0].detach().cpu().numpy(),
        "pred_cam": output["pred_cam"][0].detach().cpu().numpy(),
        "full_camera_translation": full_camera,
        "vertices": vertices,
        "bbox_xyxy": box[0],
        "detection_score": np.asarray(float(scores[selected]), dtype=np.float32),
        "focal_length": np.asarray(float(scaled_focal), dtype=np.float32),
        "image_size_wh": np.asarray([image.shape[1], image.shape[0]], dtype=np.int32),
    }
    overlay = _render_overlay(
        image,
        vertices,
        full_camera,
        renderer,
        float(scaled_focal),
    )
    return prediction, overlay


def _write_archive(
    output_root: Path,
    keypose_root: Path,
    items: list[KeyPoseImage],
) -> Path:
    predictions = [
        _load_saved_prediction(_prediction_path(output_root, item))
        for item in items
    ]
    archive_path = output_root / "keypose_smpl.npz"
    np.savez_compressed(
        archive_path,
        image_paths=np.asarray([str(item.image_path) for item in items]),
        relative_paths=np.asarray([str(item.relative_path) for item in items]),
        move_ids=np.asarray([item.move_id for item in items], dtype=np.int16),
        move_names=np.asarray([item.move_name for item in items]),
        keypose_orders=np.asarray(
            [item.keypose_order for item in items],
            dtype=np.int16,
        ),
        source_frame_numbers=np.asarray(
            [item.source_frame_number for item in items],
            dtype=np.int32,
        ),
        body_poses=np.stack([item["body_pose"] for item in predictions]),
        global_orients=np.stack([item["global_orient"] for item in predictions]),
        betas=np.stack([item["betas"] for item in predictions]),
        bboxes_xyxy=np.stack([item["bbox_xyxy"] for item in predictions]),
        detection_scores=np.asarray(
            [float(item["detection_score"]) for item in predictions],
            dtype=np.float32,
        ),
    )

    rows = []
    for item, prediction in zip(items, predictions):
        rows.append(
            {
                "move_id": item.move_id,
                "move_name": item.move_name,
                "keypose_order": item.keypose_order,
                "source_frame_number": item.source_frame_number,
                "image_path": str(item.image_path),
                "relative_path": str(item.relative_path),
                "prediction_path": str(_prediction_path(output_root, item)),
                "overlay_path": str(_overlay_path(output_root, item)),
                "detection_score": float(prediction["detection_score"]),
            }
        )
    manifest_path = output_root / "keypose_smpl_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "keypose_smpl_summary.json").write_text(
        json.dumps(
            {
                "keypose_root": str(keypose_root),
                "keypose_count": len(items),
                "move_count": len(set(item.move_id for item in items)),
                "archive_path": str(archive_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypose-root", type=Path, default=DEFAULT_KEYPOSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--four-d-humans-root",
        type=Path,
        default=DEFAULT_FOUR_D_HUMANS_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detection-score", type=float, default=0.5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-image predictions that already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    items = discover_keypose_images(args.keypose_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = [
        item
        for item in items
        if args.force or not _prediction_path(args.output_root, item).is_file()
    ]

    if pending:
        dependencies = _load_hmr2_dependencies(args.four_d_humans_root)
        dependencies["download_models"](dependencies["CACHE_DIR_4DHUMANS"])
        checkpoint = args.checkpoint or Path(dependencies["DEFAULT_CHECKPOINT"])
        model, model_cfg = dependencies["load_hmr2"](str(checkpoint))
        device = torch.device(args.device)
        model = model.to(device).eval()
        detector = _configure_detector(dependencies, args.detection_score)
        renderer = dependencies["Renderer"](model_cfg, faces=model.smpl.faces)

        total_start = time.perf_counter()
        for index, item in enumerate(pending, start=1):
            started = time.perf_counter()
            prediction, overlay = _infer_one(
                item,
                model=model,
                model_cfg=model_cfg,
                detector=detector,
                renderer=renderer,
                dependencies=dependencies,
                device=device,
                score_threshold=args.detection_score,
            )
            prediction_path = _prediction_path(args.output_root, item)
            overlay_path = _overlay_path(args.output_root, item)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prediction_path, **prediction)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise RuntimeError(f"Failed to write overlay: {overlay_path}")
            elapsed = time.perf_counter() - started
            print(
                f"[{index:03d}/{len(pending):03d}] {item.relative_path} "
                f"{elapsed:.2f}s",
                flush=True,
            )
        print(
            f"HMR2 inference completed in {time.perf_counter() - total_start:.2f}s",
            flush=True,
        )
    else:
        print("All per-image SMPL predictions already exist; rebuilding archive.")

    archive_path = _write_archive(args.output_root, args.keypose_root, items)
    print(f"KeyPose SMPL archive: {archive_path}")


if __name__ == "__main__":
    main()
