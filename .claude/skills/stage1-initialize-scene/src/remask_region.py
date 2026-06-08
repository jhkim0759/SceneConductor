#!/usr/bin/env python3
"""
remask_region.py — Run SAM (ViT-H) directly with bbox or point prompts to add missing objects.

No LLM calls. Pure SAM inference + mask I/O.

Re-mask plan schema:
{
  "new_objects": [
    {"class": "floor_lamp", "bbox_xy": [0.72, 0.15, 0.82, 0.65], "reason": "visible near couch"},
    {"class": "plant",      "point_xy": [0.35, 0.55]}
  ]
}

bbox_xy: normalised [x0, y0, x1, y1] in [0, 1].
point_xy: normalised [x, y] in [0, 1].

Side-effects:
  - Appends masks/M+1.png, M+2.png, ... for each new object.
  - Updates masks/mask.png (label map).
  - Updates object_class.json.
  - Updates mask_attribute.json (via mask_attribute module).

Usage:
    python remask_region.py --scene_dir /path/to/scene --remask_plan plan.json [--gpu 0]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# SAM is imported lazily inside _load_predictor() so that --help works without
# the GroundedSAM conda env active.
# grounded-sam backend is vendored as a sibling folder inside this skill's src/
_GROUNDED_SAM_SKILL = Path(__file__).resolve().parent / "grounded-sam"
_SKILL_SAM_DIR = _GROUNDED_SAM_SKILL / "Grounded-Segment-Anything"
_SKILL_SAM_PKG_DIR = _SKILL_SAM_DIR / "segment_anything"

# Local sibling
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mask_attribute
from merge_masks import _validate_consistency

# SAM checkpoint candidates (searched in order)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DIRS = yaml.safe_load((_PROJECT_ROOT / "DIRECTORYS.yaml").read_text())


def _dir(key, default):
    p = Path(_DIRS.get(key, default))
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


_SAM_CKPT_CANDIDATES = [
    _dir("checkpoints_grounded_sam", "./checkpoints/grounded-sam") / "sam_vit_h_4b8939.pth",
]


def _find_sam_checkpoint() -> Path:
    for p in _SAM_CKPT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "SAM ViT-H checkpoint not found. Searched:\n"
        + "\n".join(f"  {p}" for p in _SAM_CKPT_CANDIDATES)
    )


def _load_predictor(gpu: int):
    """Lazy SAM import — only called when actually running inference."""
    # Try to add the skill-local segment_anything to sys.path first
    for p in (_SKILL_SAM_DIR, _SKILL_SAM_PKG_DIR):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        from segment_anything import sam_model_registry, SamPredictor as _SP
    except ImportError as e:
        print(
            "[remask_region] ERROR: 'segment_anything' is not importable.\n"
            "  Activate the conda env that has GroundedSAM installed, e.g.:\n"
            "    conda activate grounded-sam\n"
            "  or pass --conda_env_python /path/to/env/bin/python and re-run.\n"
            f"  Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    ckpt = _find_sam_checkpoint()
    print(f"[remask_region] Loading SAM from {ckpt} on {device}", file=sys.stderr)
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    sam.to(device)
    return _SP(sam)


def _existing_mask_ids(masks_dir: Path) -> list[int]:
    return sorted(
        int(f.stem) for f in masks_dir.glob("*.png")
        if f.stem.isdigit() and f.name != "mask.png"
    )


def _load_object_class(scene_dir: Path) -> dict[str, str]:
    oc_path = scene_dir / "object_class.json"
    if not oc_path.exists():
        return {}
    with open(oc_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if "objects" in raw:
        return {str(k): str(v) for k, v in raw["objects"].items()}
    return {str(k): str(v) for k, v in raw.items()}


def _save_object_class(scene_dir: Path, class_map: dict[str, str]) -> None:
    oc_path = scene_dir / "object_class.json"
    with open(oc_path, "w", encoding="utf-8") as fh:
        json.dump(class_map, fh, indent=2)


def _update_label_map(masks_dir: Path, new_id: int, binary_mask: np.ndarray) -> None:
    """Add new_id region to masks/mask.png (or create it)."""
    label_path = masks_dir / "mask.png"
    if label_path.exists():
        arr = np.array(Image.open(label_path)).astype(np.int32)
    else:
        arr = np.zeros(binary_mask.shape, dtype=np.int32)
    arr[binary_mask] = new_id
    Image.fromarray(arr.astype(np.uint8)).save(label_path)


def _pick_best_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """From SAM multimask output, pick the mask with highest score."""
    best_idx = int(np.argmax(scores))
    return masks[best_idx]


def apply_remask_plan(scene_dir: Path, remask_plan: dict, gpu: int = 0) -> list[int]:
    """
    Run SAM for each new_object in remask_plan.
    Returns list of new mask IDs assigned.
    """
    masks_dir = scene_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    # Find image
    image_path = None
    for name in ("image.png", "image.jpg", "image.jpeg"):
        candidate = scene_dir / name
        if candidate.exists():
            image_path = candidate
            break
    if image_path is None:
        raise FileNotFoundError(f"No image.png/jpg found in {scene_dir}")

    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    img_h, img_w = image_np.shape[:2]

    predictor = _load_predictor(gpu)
    predictor.set_image(image_np)

    existing_ids = _existing_mask_ids(masks_dir)
    existing_ids_before = list(existing_ids)
    next_id = (max(existing_ids) + 1) if existing_ids else 1

    class_map = _load_object_class(scene_dir)
    new_objects = remask_plan.get("new_objects", [])
    new_ids: list[int] = []

    for obj_spec in new_objects:
        cls = obj_spec.get("class", "unknown")

        if "bbox_xy" in obj_spec:
            x0n, y0n, x1n, y1n = obj_spec["bbox_xy"]
            box = np.array([x0n * img_w, y0n * img_h, x1n * img_w, y1n * img_h])
            masks_out, scores_out, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # (1, 4)
                multimask_output=True,
            )
        elif "point_xy" in obj_spec:
            px, py = obj_spec["point_xy"]
            point_coords = np.array([[px * img_w, py * img_h]])
            point_labels = np.array([1])  # foreground
            masks_out, scores_out, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        else:
            print(f"[remask_region] WARNING: obj_spec has no bbox_xy or point_xy: {obj_spec}", file=sys.stderr)
            continue

        binary = _pick_best_mask(masks_out, scores_out)  # (H, W) bool
        area = int(binary.sum())
        if area == 0:
            print(f"[remask_region] WARNING: SAM returned empty mask for {cls}, skipping", file=sys.stderr)
            continue

        mask_path = masks_dir / f"{next_id}.png"
        Image.fromarray((binary.astype(np.uint8) * 255)).save(mask_path)
        _update_label_map(masks_dir, next_id, binary)

        class_map[str(next_id)] = cls
        new_ids.append(next_id)
        print(f"[remask_region] Added mask {next_id} for '{cls}' (area={area}px)", file=sys.stderr)
        next_id += 1

    _save_object_class(scene_dir, class_map)
    mask_attribute.record_remask(scene_dir, remask_plan, new_ids)

    if new_ids and existing_ids_before:
        assert min(new_ids) > max(existing_ids_before), (
            f"[remask_region] FATAL: new ids {new_ids} overlap pre-existing "
            f"ids (max existing={max(existing_ids_before)})"
        )

    _validate_consistency(masks_dir, scene_dir / "object_class.json")

    print(f"[remask_region] Done. Added {len(new_ids)} masks: {new_ids}", file=sys.stderr)
    return new_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM with bbox/point prompts to add missing masks")
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--remask_plan", type=Path, required=True, help="Path to remask plan JSON")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    parser.add_argument("--conda_env_python", type=str, default=None,
                        help="Unused; present for interface documentation only")
    return parser.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    with open(args.remask_plan, "r", encoding="utf-8") as fh:
        remask_plan = json.load(fh)
    apply_remask_plan(scene_dir, remask_plan, gpu=args.gpu)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[remask_region] ERROR: {exc}", file=sys.stderr)
        raise
