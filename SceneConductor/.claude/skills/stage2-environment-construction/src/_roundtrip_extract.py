"""
_roundtrip_extract.py — Blender in-process extraction helper for verify_roundtrip.py.

Runs inside Blender (--background). Given an already-opened .blend file and a
path to a roundtrip JSON copy, calls the same exporters used by the pipeline to
re-populate:
  - point_cloud block  (update_blender_scene_json_pc.write_point_cloud_block)
  - stage block        (update_blender_scene_json.build_stage_block from polygon_v2.json)
  - env blocks         (export_env_to_json.export_env_to_json)

The roundtrip JSON is a copy of the original blender_scene.json and must
already exist before this script is called. This script overwrites only the
blocks owned by each extractor — all other fields are preserved.

Usage (driven by verify_roundtrip.py):
    blender --background original.blend --python _roundtrip_extract.py -- \\
        --scene-json /tmp/.../blender_scene.roundtrip.json \\
        --scene-dir  /path/to/scene_dir

The --scene-dir is the ORIGINAL scene dir (for resolving polygon_v2.json, PLY
path, etc.). The roundtrip JSON lives in a tmp dir but references assets in
the original scene dir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse args after the "--" separator
# ---------------------------------------------------------------------------

if "--" in sys.argv:
    _script_args = sys.argv[sys.argv.index("--") + 1:]
else:
    _script_args = []

_parser = argparse.ArgumentParser(prog="_roundtrip_extract.py")
_parser.add_argument("--scene-json", required=True,
                     help="Path to blender_scene.roundtrip.json (in tmp dir)")
_parser.add_argument("--scene-dir", required=True,
                     help="Path to the ORIGINAL scene dir (for polygon_v2.json, PLY, etc.)")
_args = _parser.parse_args(_script_args)

_SCENE_JSON = Path(_args.scene_json).resolve()
_SCENE_DIR = Path(_args.scene_dir).resolve()

LOG = "[roundtrip_extract]"

# ---------------------------------------------------------------------------
# sys.path augmentation — point to all skill scripts dirs
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent.resolve()
_SKILLS_DIR = _THIS_DIR.parent.parent.parent  # SceneConductor/.claude/skills/

_PATH_ADDITIONS = [
    _THIS_DIR,
    _SKILLS_DIR / "stage2-sub-pointmap-to-separable-stage" / "src",
    _SKILLS_DIR / "stage2-sub-env-enhance" / "src",
]
for _p in _PATH_ADDITIONS:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Import bpy (available only inside Blender)
# ---------------------------------------------------------------------------
import bpy  # noqa: E402

print(f"{LOG} Blender version: {bpy.app.version_string}")
print(f"{LOG} scene_json={_SCENE_JSON}")
print(f"{LOG} scene_dir={_SCENE_DIR}")

# ---------------------------------------------------------------------------
# Load current roundtrip JSON (we will update it in place)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: Path, data: dict) -> None:
    """Atomic write via sibling tempfile + rename."""
    import tempfile
    parent = path.parent
    fd, tmp = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Block 1: point_cloud — re-extract from scene
# ---------------------------------------------------------------------------

def _extract_point_cloud(scene_json: dict) -> None:
    """
    Re-populate point_cloud block from live bpy state or from polygon_v2.json.

    If the .blend contains a PointCloud_XZ mesh, count its vertices and update
    num_vertices. PLY path and import config are preserved from the existing JSON
    (they cannot be recovered from bpy state alone without re-reading the PLY).
    """
    # Get the existing point_cloud block (copied from the original JSON).
    pc = scene_json.get("point_cloud")
    if pc is None:
        print(f"{LOG} point_cloud block absent in roundtrip JSON — skipping")
        return

    # Count vertices from the live PointCloud_XZ mesh (most reliable source).
    pc_obj_name = pc.get("name", "PointCloud_XZ")
    pc_obj = bpy.data.objects.get(pc_obj_name)
    if pc_obj is not None and pc_obj.type == "MESH":
        n_verts = len(pc_obj.data.vertices)
        pc["num_vertices"] = n_verts
        print(f"{LOG} point_cloud: counted {n_verts} vertices from '{pc_obj_name}'")
    else:
        # Fall back to PLY header parsing if the mesh is not in the scene.
        ply_rel = pc.get("ply_path", "")
        ply_abs = _SCENE_DIR / ply_rel
        if ply_abs.exists():
            n_verts = _count_ply_vertices_header(ply_abs)
            if n_verts is not None:
                pc["num_vertices"] = n_verts
                print(f"{LOG} point_cloud: PLY header count={n_verts}")
            else:
                print(f"{LOG} point_cloud: could not count PLY vertices")
        else:
            print(f"{LOG} point_cloud: PLY not found at {ply_abs} — num_vertices unchanged")

    scene_json["point_cloud"] = pc


def _count_ply_vertices_header(ply_path: Path) -> int | None:
    """Read only the PLY ASCII header and return vertex count."""
    try:
        with ply_path.open("rb") as fh:
            magic = fh.read(3)
            if magic != b"ply":
                return None
            fh.seek(0)
            vertex_count = None
            for raw_line in fh:
                line = raw_line.decode("ascii", errors="replace").rstrip("\r\n").strip()
                if line.startswith("element vertex"):
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            vertex_count = int(parts[2])
                        except ValueError:
                            pass
                if line == "end_header":
                    break
        return vertex_count
    except Exception as exc:
        print(f"{LOG} WARN: PLY header read error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Block 2: stage — re-populate from live bpy Stage collection + polygon_v2.json
# ---------------------------------------------------------------------------

def _extract_stage(scene_json: dict) -> None:
    """
    Re-populate stage block.

    Strategy: read polygon_v2.json from scene_dir (the canonical source for
    polygon geometry), then read floor_z/ceiling_z from the live Floor/Ceiling
    objects in the Stage collection (most accurate post-edit values).

    If polygon_v2.json is absent, the existing stage block is preserved as-is.
    """
    polygon_path = _SCENE_DIR / "polygon_v2.json"
    if not polygon_path.exists():
        print(f"{LOG} stage: polygon_v2.json not found — stage block preserved as-is")
        return

    try:
        from update_blender_scene_json import build_stage_block
    except ImportError as exc:
        print(f"{LOG} stage: could not import build_stage_block ({exc}) — stage preserved")
        return

    try:
        polygon = json.loads(polygon_path.read_text())
    except Exception as exc:
        print(f"{LOG} stage: failed to read polygon_v2.json: {exc} — stage preserved")
        return

    # Read wall/floor/ceiling thicknesses from existing stage block if present,
    # otherwise use defaults.
    existing_stage = scene_json.get("stage", {})
    wall_t = existing_stage.get("wall_thickness", 0.25)
    floor_t = existing_stage.get("floor_thickness", 0.30)
    ceiling_t = existing_stage.get("ceiling_thickness", 0.30)

    stage_block = build_stage_block(polygon, wall_t, floor_t, ceiling_t)

    # Override floor_z / ceiling_z from live Blender objects if available.
    # The Floor mesh top face is at floor_z; Ceiling mesh bottom face at ceiling_z.
    floor_obj = bpy.data.objects.get("Floor")
    ceiling_obj = bpy.data.objects.get("Ceiling")

    if floor_obj is not None and floor_obj.type == "MESH":
        bpy.context.view_layer.update()
        zs = [floor_obj.matrix_world @ v.co for v in floor_obj.data.vertices]
        if zs:
            floor_z_live = max(v.z for v in zs)
            stage_block["floor_z"] = round(floor_z_live, 6)
            print(f"{LOG} stage: floor_z from mesh = {floor_z_live:.4f}")

    if ceiling_obj is not None and ceiling_obj.type == "MESH":
        bpy.context.view_layer.update()
        zs = [ceiling_obj.matrix_world @ v.co for v in ceiling_obj.data.vertices]
        if zs:
            ceiling_z_live = min(v.z for v in zs)
            stage_block["ceiling_z"] = round(ceiling_z_live, 6)
            print(f"{LOG} stage: ceiling_z from mesh = {ceiling_z_live:.4f}")

    # Preserve openings[] from existing stage block (they are not in polygon_v2.json).
    if "openings" in existing_stage:
        stage_block["openings"] = existing_stage["openings"]

    scene_json["stage"] = stage_block
    print(f"{LOG} stage: rebuilt from polygon_v2.json "
          f"({len(stage_block.get('wall_objects', []))} walls, "
          f"floor_z={stage_block.get('floor_z')}, "
          f"ceiling_z={stage_block.get('ceiling_z')})")


# ---------------------------------------------------------------------------
# Block 3: env blocks — lighting, world, stage_materials, render, compositor
# ---------------------------------------------------------------------------

def _extract_env(scene_json: dict) -> None:
    """
    Re-populate env blocks from live bpy state using export_env_to_json.

    Writes directly to _SCENE_JSON (the roundtrip copy).
    """
    try:
        from export_env_to_json import export_env_to_json
    except ImportError as exc:
        print(f"{LOG} env: could not import export_env_to_json ({exc}) — env blocks preserved")
        return

    try:
        summary = export_env_to_json(
            str(_SCENE_JSON),
            overwrite_if_present=True,
            include_compositor=True,
        )
        print(f"{LOG} env: exported {summary}")
    except Exception as exc:
        print(f"{LOG} env: export_env_to_json failed: {exc} — env blocks preserved")


# ---------------------------------------------------------------------------
# Run extraction sequence
# ---------------------------------------------------------------------------

scene_json = _load_json(_SCENE_JSON)

print(f"{LOG} --- Extracting point_cloud block ---")
_extract_point_cloud(scene_json)
_save_json(_SCENE_JSON, scene_json)

print(f"{LOG} --- Extracting stage block ---")
# Reload in case prior save changed things.
scene_json = _load_json(_SCENE_JSON)
_extract_stage(scene_json)
_save_json(_SCENE_JSON, scene_json)

print(f"{LOG} --- Extracting env blocks (lighting/world/stage_materials/render) ---")
# export_env_to_json writes directly to the file, so no need to reload/save.
_extract_env(scene_json)

print(f"{LOG} Extraction complete. Roundtrip JSON: {_SCENE_JSON}")
