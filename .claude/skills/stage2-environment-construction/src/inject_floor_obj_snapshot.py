#!/usr/bin/env python3
"""
inject_floor_obj_snapshot.py

Post-process step for the Stage-2 build snapshot ONLY.
Copies blend/blender_scene.blend -> blend/stage2-sub-build.blend, then injects
floor.obj (from layout_prediction.json) into the snapshot at the pose that
convert.py would have assigned if it hadn't dropped the floor entry.

The LIVE blend/blender_scene.blend is never touched.

CLI:
    python3 inject_floor_obj_snapshot.py <scene_dir> [--blender <path>]

Exit codes:
    0  success (or graceful no-op fallback)
    1  hard error (source blend missing)
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Blender binary resolution — mirrors verify_roundtrip.py in this repo.
# Priority: CLI --blender > $BLENDER_BIN env > $BLENDER env > DIRECTORYS.yaml
# > PATH search.
# ---------------------------------------------------------------------------

def _resolve_blender(override: str | None) -> str:
    if override:
        return override
    env_bin = os.environ.get("BLENDER_BIN") or os.environ.get("BLENDER")
    if env_bin:
        return env_bin
    # Try DIRECTORYS.yaml (repo canonical machine-specific path).
    try:
        _repo_root = Path(__file__).resolve().parents[4]
        _dirs_path = _repo_root / "DIRECTORYS.yaml"
        if _dirs_path.is_file():
            import yaml
            _dirs = yaml.safe_load(_dirs_path.read_text(encoding="utf-8"))
            yaml_bin = _dirs.get("blender_bin")
            if yaml_bin:
                yaml_path = Path(yaml_bin)
                if not yaml_path.is_absolute():
                    yaml_path = (_repo_root / yaml_path).resolve()
                if yaml_path.exists():
                    return str(yaml_path)
    except Exception:
        pass
    found = shutil.which("blender")
    if found:
        return found
    return "blender"  # last resort — let the subprocess error speak for itself


# ---------------------------------------------------------------------------
# Import math helpers from convert.py without copy-pasting.
# ---------------------------------------------------------------------------

def _import_convert_helpers():
    """Add convert.py's directory to sys.path and import the needed helpers."""
    convert_dir = str(
        Path(__file__).resolve().parents[2] / "stage2-sub-pointmap-to-separable-stage"
        / "src"
    )
    if convert_dir not in sys.path:
        sys.path.insert(0, convert_dir)
    from convert import (  # noqa: PLC0415
        convert_vec,
        convert_rotation,
        rotation_matrix_to_euler_xyz,
        parse_scale,
    )
    return convert_vec, convert_rotation, rotation_matrix_to_euler_xyz, parse_scale


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compute_floor_transform(layout_json: dict, k: float):
    """Replicate exactly what convert.py would have done for oid=='floor'.

    Steps (all matching convert.py lines 540-570 + 726-735):
      1. convert_vec(translation)   -> Blender location (P @ v)
      2. convert_rotation(rotation) -> R_bl = P @ R_tm @ P
         rotation_matrix_to_euler_xyz(R_bl) -> euler radians
      3. No Z+180 fix (floor is exempt, convert.py line 552)
      4. parse_scale(scale_raw) -> s; scale = [s, s, s]
      5. location = [v * k for v in location]  (world-scale, line 727-734)
         rotation and per-object scale are NOT multiplied by k.

    Returns (location, euler_radians, scale, euler_deg).
    """
    convert_vec, convert_rotation, rotation_matrix_to_euler_xyz, parse_scale = (
        _import_convert_helpers()
    )

    ids = layout_json.get("object_id", [])
    if "floor" not in ids:
        raise ValueError("No 'floor' entry in layout_prediction.json object_id list")

    fi = ids.index("floor")
    translation = layout_json["translation"][fi]
    rotation_raw = layout_json["rotation"][fi]
    scale_raw = layout_json["scale"][fi]

    # Step 1 — coordinate conversion
    location = convert_vec(translation)

    # Step 2 — rotation conversion
    R_bl = convert_rotation(rotation_raw)
    euler = rotation_matrix_to_euler_xyz(R_bl)
    # (euler is [rx, ry, rz] in radians)

    # Step 3 — floor is exempt from Z+180 fix (convert.py line 552)
    # Nothing to do here.

    # Step 4 — scale
    s = parse_scale(scale_raw)
    scale = [s, s, s]

    # Step 5 — world-scale: multiply location by k (rotation + per-obj scale untouched)
    location = [v * k for v in location]

    euler_deg = [math.degrees(a) for a in euler]
    return location, euler, scale, euler_deg


def read_world_scale_k(blender_scene_json_path: str) -> float:
    """Read the final world-scale factor k from blender_scene.json.

    The field is meta.world_scale_factor (convert.py line 814).
    Falls back to 1.0 if the file or field is absent.
    """
    try:
        with open(blender_scene_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        k = float(data["meta"]["world_scale_factor"])
        return k
    except (OSError, KeyError, ValueError, TypeError):
        return 1.0


def _blender_inject_script(
    target_blend: str,
    floor_obj_path: str,
    location: list,
    euler_rad: list,
    scale: list,
) -> str:
    """Return a Python script string to be run inside Blender headlessly.

    The script:
      1. Opens target_blend.
      2. Imports floor.obj.
      3. Wraps imported roots under an Empty named 'floor', mirrors the
         import_and_place() pattern used in build.py.
      4. Sets the Empty's transform to (location, euler_rad, scale).
      5. Saves the blend in-place.
    """
    # Inline the OBJ import pattern from build.py lines 139-143 verbatim.
    script = f"""\
import bpy, sys, os, math
from mathutils import Euler, Vector, Matrix

target_blend = {target_blend!r}
floor_obj_path = {floor_obj_path!r}
location  = {location!r}
euler_rad = {euler_rad!r}
scale     = {scale!r}

# Open the snapshot blend.
bpy.ops.wm.open_mainfile(filepath=target_blend)

# --- Import floor.obj (pattern from build.py) ---
before = set(bpy.data.objects.keys())
try:
    bpy.ops.wm.obj_import(filepath=floor_obj_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=floor_obj_path)
after = set(bpy.data.objects.keys())
new_names = after - before
new_objs = [bpy.data.objects[n] for n in new_names]

# Remove any cameras/lights that came in with the OBJ.
to_delete = [o for o in new_objs if o.type in {{"CAMERA", "LIGHT"}}]
for obj in to_delete:
    bpy.data.objects.remove(obj, do_unlink=True)
new_objs = [o for o in new_objs if o.type not in {{"CAMERA", "LIGHT"}}]

if not new_objs:
    print("[inject-floor-obj] WARNING: no mesh objects found in floor.obj — snapshot saved without floor")
else:
    # Determine root objects (parent not in the imported set).
    imported_set = set(new_objs)
    roots = [o for o in new_objs if (o.parent is None or o.parent not in imported_set)]

    # Create an Empty named 'floor' to carry the JSON transform,
    # matching build.py's import_and_place() convention (line 357).
    empty = bpy.data.objects.new("floor", None)
    bpy.context.collection.objects.link(empty)
    for r in roots:
        if r.parent is not None:
            r.parent = None
        r.parent = empty
        r.matrix_parent_inverse = Matrix.Identity(4)

    empty.rotation_mode = "XYZ"
    empty.location       = Vector(location)
    empty.rotation_euler = Euler(euler_rad, "XYZ")
    empty.scale          = Vector(scale)

    print(f"[inject-floor-obj] placed 'floor' empty: "
          f"loc={{list(empty.location)}} rot={{list(empty.rotation_euler)}} scale={{list(empty.scale)}}")

# Save in-place (overwrites stage2-sub-build.blend).
bpy.ops.wm.save_mainfile()
print("[inject-floor-obj] blend saved")
"""
    return script


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy blender_scene.blend -> stage2-sub-build.blend and inject "
            "floor.obj into the snapshot at the pose specified by layout_prediction.json."
        )
    )
    parser.add_argument("scene_dir", help="Path to the scene directory")
    parser.add_argument(
        "--blender",
        default=None,
        help=(
            "Path to the Blender binary. Defaults to $BLENDER_BIN, $BLENDER, "
            "DIRECTORYS.yaml blender_bin, then PATH search."
        ),
    )
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    source_blend = os.path.join(scene_dir, "blend", "blender_scene.blend")
    target_blend = os.path.join(scene_dir, "blend", "stage2-sub-build.blend")
    layout_json_path = os.path.join(scene_dir, "inputs", "layout_prediction.json")
    blender_scene_json = os.path.join(scene_dir, "json", "blender_scene.json")

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if not os.path.isfile(source_blend):
        print(
            f"[inject-floor-obj] ERROR: source blend not found: {source_blend}",
            file=sys.stderr,
        )
        sys.exit(1)

    def _graceful_fallback(reason: str):
        print(f"[inject-floor-obj] WARNING: {reason} — copying blend without floor injection")
        shutil.copy2(source_blend, target_blend)
        print(f"[inject-floor-obj] snapshot copied: {target_blend}")

    # Check layout_prediction.json
    if not os.path.isfile(layout_json_path):
        _graceful_fallback(f"layout_prediction.json not found: {layout_json_path}")
        return

    with open(layout_json_path, "r", encoding="utf-8") as fh:
        layout_json = json.load(fh)

    if "floor" not in layout_json.get("object_id", []):
        _graceful_fallback("layout_prediction.json has no 'floor' entry")
        return

    # Resolve and verify floor.obj path
    ids = layout_json["object_id"]
    fi = ids.index("floor")
    raw_mesh_path = layout_json["meshes"][fi]
    floor_obj_path = raw_mesh_path if os.path.isfile(raw_mesh_path) else None

    if floor_obj_path is None:
        # Try canonical location: <scene_dir>/inputs/floor.obj
        cand = os.path.join(scene_dir, "inputs", "floor.obj")
        if os.path.isfile(cand):
            floor_obj_path = cand
        else:
            _graceful_fallback(
                f"floor.obj not found (tried {raw_mesh_path!r} and {cand!r})"
            )
            return

    # ------------------------------------------------------------------
    # Compute floor transform (replicating convert.py exactly)
    # ------------------------------------------------------------------
    k = read_world_scale_k(blender_scene_json)
    location, euler_rad, scale, euler_deg_vals = compute_floor_transform(layout_json, k)

    # ------------------------------------------------------------------
    # Step 1: copy source -> target (idempotent — always overwrites)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(target_blend), exist_ok=True)
    shutil.copy2(source_blend, target_blend)

    # ------------------------------------------------------------------
    # Step 2: spawn Blender subprocess to inject floor.obj into snapshot
    # ------------------------------------------------------------------
    blender_bin = _resolve_blender(args.blender)

    inject_script = _blender_inject_script(
        target_blend=target_blend,
        floor_obj_path=floor_obj_path,
        location=location,
        euler_rad=euler_rad,
        scale=scale,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_inject_floor.py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(inject_script)
        tmp_script_path = tmp.name

    try:
        cmd = [blender_bin, "--background", "--python", tmp_script_path]
        rc = subprocess.call(cmd)
    finally:
        try:
            os.unlink(tmp_script_path)
        except OSError:
            pass

    if rc != 0:
        print(
            f"[inject-floor-obj] ERROR: Blender subprocess exited with code {rc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    loc_str = f"({location[0]:.4f},{location[1]:.4f},{location[2]:.4f})"
    rot_str = f"({euler_deg_vals[0]:.2f},{euler_deg_vals[1]:.2f},{euler_deg_vals[2]:.2f})"
    sc_str  = f"({scale[0]:.4f},{scale[1]:.4f},{scale[2]:.4f})"
    print(
        f"[inject-floor-obj] OK: floor imported at "
        f"loc={loc_str} rot_deg={rot_str} scale={sc_str} k={k:.6f}"
    )


if __name__ == "__main__":
    main()
