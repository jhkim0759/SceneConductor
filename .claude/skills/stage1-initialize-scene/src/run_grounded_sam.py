#!/usr/bin/env python3
"""
Stage 1 — GroundedSAM runner.

Calls the existing grounded-sam/run_inference.py via subprocess (grounded-sam conda env),
then converts its 0-indexed native outputs to the pipeline's 1-indexed contract:

  <scene_dir>/masks/mask.png   — integer PNG, pixel value 0=bg 1..N=objects
  <scene_dir>/masks/1.png      — binary mask for object 1 (0 or 255)
  <scene_dir>/masks/2.png ...
  <scene_dir>/object_class.json — {"1": "chair", "2": "table", ...}

Usage:
    python run_grounded_sam.py \\
        --scene_dir /path/to/scene \\
        --prompt "chair. table. lamp." \\
        [--gpu 0] \\
        [--conda_env_name grounded-sam]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# ── DIRECTORYS.yaml (canonical machine-specific paths) ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())


def _dir(key, default):
    p = Path(_DIRS.get(key, default))
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


# ── Hardcoded paths (overridable via CLI) ────────────────────────────────────
_PROJECT_ROOT = _REPO_ROOT
# grounded-sam backend is vendored as a sibling folder inside this skill's src/
_GROUNDED_SAM_SKILL = Path(__file__).resolve().parent / "grounded-sam"
GROUNDED_SAM_SCRIPT = _GROUNDED_SAM_SKILL / "run_inference.py"
CONDA_ENV_NAME_DEFAULT = _DIRS["conda_envs"]["grounded-sam"]
_CHECKPOINTS_DIR = _dir("checkpoints_grounded_sam", "./checkpoints/grounded-sam")
GDINO_CKPT = _CHECKPOINTS_DIR / "groundingdino_swint_ogc.pth"
SAM_CKPT = _CHECKPOINTS_DIR / "sam_vit_h_4b8939.pth"


# ── Preflight checks ─────────────────────────────────────────────────────────

def preflight(conda_env_name: str, scene_dir: Path) -> None:
    errors = []

    if not conda_env_name:
        errors.append("Conda env name is empty")
    if not GROUNDED_SAM_SCRIPT.exists():
        errors.append(f"GroundedSAM script not found: {GROUNDED_SAM_SCRIPT}")
    if not GDINO_CKPT.exists():
        errors.append(f"GroundingDINO checkpoint missing: {GDINO_CKPT}")
    if not SAM_CKPT.exists():
        errors.append(f"SAM checkpoint missing: {SAM_CKPT}")

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        errors.append(f"Input image not found: {image_path}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


# ── Post-processing: native → pipeline contract ───────────────────────────────

def convert_outputs(native_dir: Path, scene_dir: Path) -> int:
    """
    Convert GroundedSAM native outputs in native_dir to pipeline layout.

    Native outputs:
        0.png, 1.png, ...    binary masks (0-indexed)
        mask.npy             float32 HxW label map (values 1..N)
        label.json           tags + mask list

    Pipeline outputs:
        <scene_dir>/masks/1.png ... N.png   (1-indexed binary masks)
        <scene_dir>/masks/mask.png          (integer PNG, mode L)
        <scene_dir>/object_class.json       ({"1": "chair", ...})

    Returns number of objects written (N).
    """
    import numpy as np
    from PIL import Image

    masks_out = scene_dir / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    # ── 1. Rename 0.png..N-1.png → 1.png..N.png ────────────────────────────
    native_pngs = sorted(
        [p for p in native_dir.iterdir() if p.stem.isdigit() and p.suffix == ".png"],
        key=lambda p: int(p.stem),
    )
    n_objects = len(native_pngs)
    if n_objects == 0:
        print("[WARN] GroundedSAM produced no individual mask PNGs.", file=sys.stderr)

    for native_png in native_pngs:
        one_indexed = int(native_png.stem) + 1
        dest = masks_out / f"{one_indexed}.png"
        shutil.copy2(native_png, dest)
        print(f"  Copied {native_png.name} → masks/{dest.name}")

    # ── 2. Convert mask.npy → mask.png (integer PNG, mode L) ────────────────
    npy_path = native_dir / "mask.npy"
    if not npy_path.exists():
        print(f"[WARN] mask.npy not found in {native_dir}", file=sys.stderr)
    else:
        arr = np.load(str(npy_path))          # float32 HxW, values 0..N
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode="L")
        img.save(str(masks_out / "mask.png"))
        print(f"  mask.npy → masks/mask.png  (shape={arr.shape}, max={arr.max():.0f})")

    # ── 3. Build object_class.json from label.json, aligned to actual PNGs ──
    # label.json holds the class for every box (1..N) that DINO detected, but
    # save_individual_masks may skip masks that are too small, so the actual
    # PNGs may be fewer. We therefore must filter by the ids of the PNGs that
    # exist on disk so object_class.json aligns 1:1 with the masks without stale entries.
    label_json = native_dir / "label.json"
    if not label_json.exists():
        print(f"[WARN] label.json not found in {native_dir}", file=sys.stderr)
        return n_objects

    with open(label_json, encoding="utf-8") as f:
        label_data = json.load(f)

    # The native PNG filename (stem) is the 0-indexed DINO idx. The copy step creates
    # 1-indexed files in masks/ as stem+1, so existing_ids is made 1-indexed to match.
    existing_ids = {int(p.stem) + 1 for p in native_pngs}

    # label.json["mask"] = [{"value": 0, "label": "background"}, {"value": 1, "label": "chair", ...}, ...]
    object_class: dict[str, str] = {}
    skipped_stale: list[int] = []
    for entry in label_data.get("mask", []):
        value = entry.get("value", 0)
        if value == 0:
            continue  # skip background
        if value not in existing_ids:
            skipped_stale.append(value)
            continue  # no PNG on disk for this id (mask was dropped by save_individual_masks)
        label = entry.get("label", "unknown").strip()
        object_class[str(value)] = label

    out_json = scene_dir / "object_class.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(object_class, f, indent=2)
    print(f"  object_class.json written: {object_class}")
    if skipped_stale:
        print(f"  [INFO] skipped {len(skipped_stale)} stale label.json entries with no PNG: ids={skipped_stale}")

    return n_objects


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 GroundedSAM runner — wraps run_inference.py with pipeline contract"
    )
    p.add_argument("--scene_dir", required=True, type=Path,
                   help="Scene directory containing image.png")
    p.add_argument("--prompt", required=True,
                   help='Text prompt, e.g. "chair. table. lamp."')
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index (default: 0)")
    p.add_argument("--conda_env_name", type=str,
                   default=CONDA_ENV_NAME_DEFAULT,
                   help="Name of the conda env to run GroundedSAM in (default from DIRECTORYS.yaml::conda_envs.grounded-sam)")
    # box / text thresholds lowered from 0.25 → 0.20 so faintly-matching
    # text→region detections (small wall posters, picture frames, partial
    # views) survive. Step 3.8 enrich + Mask-Evaluator filter the extra
    # noise downstream. iou_threshold left untouched (NMS dedup).
    p.add_argument("--box_threshold", type=float, default=0.20)
    p.add_argument("--text_threshold", type=float, default=0.20)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    conda_env_name = args.conda_env_name

    preflight(conda_env_name, scene_dir)

    # Run GroundedSAM in a temp dir so we don't pollute scene_dir
    with tempfile.TemporaryDirectory(prefix="grounded_sam_native_") as tmp_str:
        tmp_dir = Path(tmp_str)
        image_path = scene_dir / "image.png"

        cmd = [
            "conda", "run", "-n", conda_env_name, "python",
            str(GROUNDED_SAM_SCRIPT),
            "--image", str(image_path),
            "--prompt", args.prompt,
            "--output_dir", str(tmp_dir),
            "--box_threshold", str(args.box_threshold),
            "--text_threshold", str(args.text_threshold),
            "--iou_threshold", str(args.iou_threshold),
            "--device", "cuda",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        print(f"[run_grounded_sam] GPU={args.gpu}  scene={scene_dir}")
        print(f"[run_grounded_sam] Prompt: {args.prompt}")
        print(f"[run_grounded_sam] Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            print(f"[ERROR] run_inference.py exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

        n = convert_outputs(tmp_dir, scene_dir)
        print(f"\n[run_grounded_sam] Done. {n} object(s) written to {scene_dir}/masks/")
        print(f"[run_grounded_sam] object_class.json: {scene_dir}/object_class.json")


if __name__ == "__main__":
    main()
