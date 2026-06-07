#!/usr/bin/env python3
"""
finalize_layout.py — Stage 1 Step 7: Move all stage outputs into <scene_dir>/inputs/.

Contract IN (files at scene_dir top-level or already in inputs/ by some scripts):
    object_class_prompt.json
    object_class.json
    mask_attribute.json
    masks/
    object/            (if SAM3D wrote here instead of inputs/object/)
    layout_prediction.json
    layout-prediction.glb
    pointmap_xz.ply
    floor.obj
    merge_plan.json    (optional)
    remask_plan.json   (optional)
    verification_overlay.png  (optional)

Contract OUT (all under scene_dir/inputs/, image.png stays at top-level):
    inputs/object_class_prompt.json
    inputs/object_class.json
    inputs/mask_attribute.json
    inputs/masks/
    inputs/object/
    inputs/layout_prediction.json
    inputs/layout-prediction.glb
    inputs/pointmap_xz.ply
    inputs/floor.obj
    inputs/merge_plan.json           (if present)
    inputs/remask_plan.json          (if present)
    inputs/verification_overlay.png  (if present)

Files already inside inputs/ are left in place; top-level copies are removed only
after the inputs/ destination is confirmed.

Usage:
    python finalize_layout.py --scene_dir /path/to/scene
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


# Files/dirs that should move from top-level into inputs/
SINGLE_FILES = [
    "object_class_prompt.json",
    "object_class.json",
    "mask_attribute.json",
    "layout_prediction.json",
    "layout-prediction.glb",
    "pointmap_xz.ply",
    "floor.obj",
    "merge_plan.json",        # optional
    "remask_plan.json",       # optional
    "verification_overlay.png",  # optional
    "object_state_annotated_mask.png",  # optional
    "object_state.json",      # optional
    "overlap_pairs.json",     # optional — pixel-overlap pre-filter output (Step 3.6)
    "small_mask_candidates.json",  # optional — small-mask hints (Step 3.7)
]

DIRS = [
    "masks",
    "object",
    "thumbnails",             # optional
]


def move_item(src: Path, dst: Path) -> bool:
    """Move src to dst. Returns True if moved, False if src doesn't exist."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # dst already there — only overwrite if src is newer or different size
        if src.stat().st_size != dst.stat().st_size:
            print(f"  [OVERWRITE] {dst.name} (size mismatch)")
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        else:
            print(f"  [SKIP-SAME] {dst.name} already in inputs/ and same size — removing top-level copy")
            if src != dst:
                if src.is_dir():
                    shutil.rmtree(src)
                else:
                    src.unlink()
            return True
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        shutil.rmtree(src)
    else:
        shutil.move(str(src), dst)
    print(f"  [MOVED]  {src.name} -> inputs/{dst.name}")
    return True


def update_layout_paths(inputs_dir: Path):
    """Rewrite absolute paths in layout_prediction.json to point into inputs/."""
    lp = inputs_dir / "layout_prediction.json"
    if not lp.exists():
        return
    with open(lp, encoding="utf-8") as f:
        layout = json.load(f)

    changed = False
    new_meshes = []
    for mesh_path in layout.get("meshes", []):
        p = Path(mesh_path)
        # If path points to top-level object/ or floor.obj, remap to inputs/
        # Heuristic: if the path's parent is scene_dir and not already inside inputs/
        if "inputs" not in p.parts:
            # Try to find the file in inputs/
            fname = p.name
            parent_name = p.parent.name  # e.g. "object" or the scene dir name
            if parent_name == "object":
                new_p = inputs_dir / "object" / fname
            elif fname == "floor.obj":
                new_p = inputs_dir / "floor.obj"
            else:
                new_p = inputs_dir / fname
            new_meshes.append(str(new_p))
            if str(new_p) != mesh_path:
                changed = True
        else:
            new_meshes.append(mesh_path)

    if changed:
        layout["meshes"] = new_meshes
        with open(lp, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=4)
        print(f"  [PATCHED] layout_prediction.json mesh paths updated to inputs/")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Step 7 — finalize layout into inputs/")
    parser.add_argument("--scene_dir", required=True, type=Path)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    if not scene_dir.is_dir():
        print(f"[ERROR] scene_dir not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    inputs_dir = scene_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[finalize_layout] Scene: {scene_dir}")
    print(f"[finalize_layout] Target: {inputs_dir}\n")

    # Step 1: Move top-level files into inputs/
    print("--- Moving files ---")
    for fname in SINGLE_FILES:
        src = scene_dir / fname
        dst = inputs_dir / fname
        if src == dst:
            continue  # already in inputs/ (shouldn't happen but be safe)
        move_item(src, dst)

    # Step 2: Move top-level dirs into inputs/
    print("\n--- Moving directories ---")
    for dname in DIRS:
        src = scene_dir / dname
        dst = inputs_dir / dname
        if src == dst or not src.exists():
            # Check if already in inputs/
            if dst.exists():
                print(f"  [ALREADY]  {dname}/ already in inputs/")
            continue
        move_item(src, dst)

    # Step 3: Patch layout_prediction.json mesh paths
    print("\n--- Patching layout paths ---")
    update_layout_paths(inputs_dir)

    # Step 4: Verify expected outputs
    print("\n--- Verification ---")
    required = [
        "object_class_prompt.json",
        "object_class.json",
        "mask_attribute.json",
        "masks",
        "object",
        "layout_prediction.json",
        "layout-prediction.glb",
        "pointmap_xz.ply",
        "floor.obj",
    ]
    missing = []
    for name in required:
        p = inputs_dir / name
        if p.exists():
            print(f"  [OK]  inputs/{name}")
        else:
            print(f"  [MISSING]  inputs/{name}")
            missing.append(name)

    optional = ["merge_plan.json", "remask_plan.json", "verification_overlay.png"]
    for name in optional:
        p = inputs_dir / name
        status = "OK" if p.exists() else "absent (optional)"
        print(f"  [{status}]  inputs/{name}")

    # Verify image.png stays at top-level
    if (scene_dir / "image.png").exists():
        print(f"  [OK]  image.png (top-level, not moved)")
    else:
        print(f"  [WARN]  image.png not found at top-level!")
        missing.append("image.png")

    if missing:
        print(f"\n[finalize_layout] WARNING: {len(missing)} required file(s) missing: {missing}")
        sys.exit(1)
    else:
        print(f"\n[finalize_layout] All required outputs verified in inputs/")
        print(f"[finalize_layout] Done.")


if __name__ == "__main__":
    main()
