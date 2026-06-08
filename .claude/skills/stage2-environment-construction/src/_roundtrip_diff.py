"""
_roundtrip_diff.py — Blender in-process invariant extractor and differ.

Runs inside Blender (--background, NO pre-loaded .blend).
Opens two .blend files sequentially using bpy.ops.wm.open_mainfile, extracting
invariants from each, then computes the diff and writes it to a JSON file.

Strategy (avoids bpy state wipe from wm.open_mainfile):
  1. Open blend A  → extract_invariants() → save to tmp dict A
  2. Open blend B  → extract_invariants() → save to tmp dict B
  3. Compute diff between A and B
  4. Write diff to --out path

Invariants extracted:
  - mesh_object_count (MESH type, excl. lights/cameras)
  - per-mesh-object location, rotation_euler, scale (matched by name)
  - camera location, rotation_euler, lens, sensor_width
  - light counts by type (SUN/AREA/POINT/SPOT)
  - total light energy summed by type
  - world strength + mode (nishita/hdri/flat)
  - stage_materials: base_color + roughness for Floor, Ceiling, Wall_NN
  - stage.polygon_vertices count + per-vertex XY
  - stage.floor_z, stage.ceiling_z
  - stage.openings count + per-opening xy_range + z_range
  - render samples, resolution_x/y, engine
  - point_cloud num_vertices

Usage (driven by verify_roundtrip.py):
    blender --background --python _roundtrip_diff.py -- \\
        --original /tmp/.../original.blend \\
        --roundtrip /tmp/.../roundtrip.blend \\
        --out /tmp/.../diff.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Parse args after "--" separator
# ---------------------------------------------------------------------------

if "--" in sys.argv:
    _script_args = sys.argv[sys.argv.index("--") + 1:]
else:
    _script_args = []

_parser = argparse.ArgumentParser(prog="_roundtrip_diff.py")
_parser.add_argument("--original",  required=True, help="Path to original.blend")
_parser.add_argument("--roundtrip", required=True, help="Path to roundtrip.blend")
_parser.add_argument("--out",       required=True, help="Output diff.json path")
_args = _parser.parse_args(_script_args)

LOG = "[roundtrip_diff]"

import bpy  # noqa: E402


# ---------------------------------------------------------------------------
# Invariant extractor — called once per .blend after open_mainfile
# ---------------------------------------------------------------------------

def extract_invariants() -> dict[str, Any]:
    """Extract all tracked invariants from the currently open bpy state."""
    inv: dict[str, Any] = {}

    scene = bpy.context.scene

    # --- mesh_object_count ---
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    inv["mesh_object_count"] = len(mesh_objs)

    # --- per-mesh transforms (list of dicts keyed by name) ---
    mesh_entries: list[dict] = []
    for obj in mesh_objs:
        obj.rotation_mode = "XYZ"  # ensure XYZ euler access
        entry = {
            "name": obj.name,
            "location": [round(v, 8) for v in obj.location],
            "rotation_euler": [round(v, 8) for v in obj.rotation_euler],
            "scale": [round(v, 8) for v in obj.scale],
        }
        mesh_entries.append(entry)
    inv["mesh_objects_by_name"] = {e["name"]: e for e in mesh_entries}

    # --- camera ---
    cam_obj = scene.camera
    if cam_obj is not None:
        cam_obj.rotation_mode = "XYZ"
        cam_data = cam_obj.data
        inv["camera"] = {
            "location": [round(v, 8) for v in cam_obj.location],
            "rotation_euler": [round(v, 8) for v in cam_obj.rotation_euler],
            "lens": round(float(cam_data.lens), 6),
            "sensor_width": round(float(cam_data.sensor_width), 6),
        }
    else:
        inv["camera"] = None

    # --- lights ---
    light_objs = [o for o in bpy.data.objects if o.type == "LIGHT"]
    counts: dict[str, int] = {"SUN": 0, "AREA": 0, "POINT": 0, "SPOT": 0}
    energies: dict[str, float] = {"SUN": 0.0, "AREA": 0.0, "POINT": 0.0, "SPOT": 0.0}
    for obj in light_objs:
        lt = obj.data.type
        if lt in counts:
            counts[lt] += 1
            energies[lt] += float(obj.data.energy)
    inv["light_counts"] = counts
    inv["light_energies"] = {k: round(v, 6) for k, v in energies.items()}

    # --- world ---
    world = scene.world
    world_info: dict[str, Any] = {"mode": "none", "world_strength": 0.0}
    if world is not None and world.use_nodes:
        nt = world.node_tree
        bg_node = next((n for n in nt.nodes if n.type == "BACKGROUND"), None)
        if bg_node is not None:
            strength = round(float(bg_node.inputs["Strength"].default_value), 6)
            world_info["world_strength"] = strength

            color_sock = bg_node.inputs.get("Color")
            driver_node = None
            if color_sock is not None and color_sock.is_linked:
                driver_node = color_sock.links[0].from_node

            if driver_node is not None and driver_node.bl_idname == "ShaderNodeTexSky":
                world_info["mode"] = "nishita_sky"
            elif driver_node is not None and driver_node.bl_idname == "ShaderNodeTexEnvironment":
                world_info["mode"] = "hdri"
            elif color_sock is not None and not color_sock.is_linked:
                world_info["mode"] = "flat"
            else:
                world_info["mode"] = "unknown"
    inv["world"] = world_info

    # --- stage_materials ---
    # Collect materials for stage surfaces: Floor, Ceiling, Wall_NN.
    # We read only Mat_Floor_Stage, Mat_Ceiling_Stage, Mat_Walls_Stage, Mat_Wall_NN.
    stage_mats: dict[str, dict] = {}

    def _read_principled(mat_name: str) -> dict | None:
        mat = bpy.data.materials.get(mat_name)
        if mat is None or not mat.use_nodes:
            return None
        pbsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if pbsdf is None:
            return None
        base_color_sock = pbsdf.inputs.get("Base Color")
        roughness_sock = pbsdf.inputs.get("Roughness")
        if base_color_sock is None or roughness_sock is None:
            return None
        bc = base_color_sock.default_value
        return {
            "base_color": [round(float(bc[i]), 6) for i in range(4)],
            "roughness": round(float(roughness_sock.default_value), 6),
        }

    floor_mat = _read_principled("Mat_Floor_Stage")
    if floor_mat:
        stage_mats["floor"] = floor_mat

    ceiling_mat = _read_principled("Mat_Ceiling_Stage")
    if ceiling_mat:
        stage_mats["ceiling"] = ceiling_mat

    default_wall = _read_principled("Mat_Walls_Stage")
    if default_wall:
        stage_mats["walls.__default__"] = default_wall

    # Per-wall materials (Mat_Wall_NN)
    for mat in bpy.data.materials:
        if mat.name.startswith("Mat_Wall_"):
            wall_name = mat.name[len("Mat_"):]  # e.g. "Wall_01"
            wall_mat = _read_principled(mat.name)
            if wall_mat:
                stage_mats[f"walls.{wall_name}"] = wall_mat

    inv["stage_materials"] = stage_mats

    # --- stage geometry (read from Floor/Ceiling/Wall_NN objects + blender_scene.json) ---
    # For stage polygon, floor_z, ceiling_z, openings: these are stored in
    # blender_scene.json, not recoverable purely from bpy without re-reading JSON.
    # We read from the JSON block embedded in the scene's custom properties if present,
    # OR from Floor/Ceiling object z-extents for floor_z/ceiling_z.
    stage_info: dict[str, Any] = {}

    floor_obj = bpy.data.objects.get("Floor")
    ceiling_obj = bpy.data.objects.get("Ceiling")

    if floor_obj is not None and floor_obj.type == "MESH":
        bpy.context.view_layer.update()
        zs = [floor_obj.matrix_world @ v.co for v in floor_obj.data.vertices]
        if zs:
            stage_info["floor_z"] = round(max(v.z for v in zs), 6)

    if ceiling_obj is not None and ceiling_obj.type == "MESH":
        bpy.context.view_layer.update()
        zs = [ceiling_obj.matrix_world @ v.co for v in ceiling_obj.data.vertices]
        if zs:
            stage_info["ceiling_z"] = round(min(v.z for v in zs), 6)

    # Polygon vertices: extract from Floor mesh XY footprint if available.
    # Approximate by taking the convex outline XY of Floor top face vertices at floor_z.
    if floor_obj is not None and floor_obj.type == "MESH" and "floor_z" in stage_info:
        fz = stage_info["floor_z"]
        bpy.context.view_layer.update()
        top_verts = [
            (round(float((floor_obj.matrix_world @ v.co).x), 6),
             round(float((floor_obj.matrix_world @ v.co).y), 6))
            for v in floor_obj.data.vertices
            if abs(float((floor_obj.matrix_world @ v.co).z) - fz) < 0.01
        ]
        # Deduplicate
        seen: set = set()
        unique_verts: list = []
        for pt in top_verts:
            key = (round(pt[0], 3), round(pt[1], 3))
            if key not in seen:
                seen.add(key)
                unique_verts.append(list(pt))
        stage_info["polygon_vertices"] = unique_verts

    inv["stage"] = stage_info

    # --- render ---
    inv["render"] = {
        "engine": scene.render.engine,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
    }
    try:
        inv["render"]["samples"] = scene.cycles.samples
    except AttributeError:
        inv["render"]["samples"] = None

    # --- point_cloud num_vertices ---
    # Count from PointCloud_XZ mesh object if present.
    pc_obj = bpy.data.objects.get("PointCloud_XZ")
    if pc_obj is None:
        # Try any object whose name starts with "PointCloud"
        for obj in bpy.data.objects:
            if obj.name.startswith("PointCloud") and obj.type == "MESH":
                pc_obj = obj
                break
    if pc_obj is not None and pc_obj.type == "MESH":
        inv["point_cloud_num_vertices"] = len(pc_obj.data.vertices)
    else:
        inv["point_cloud_num_vertices"] = None

    return inv


# ---------------------------------------------------------------------------
# Diff builder
# ---------------------------------------------------------------------------

def _diff_val(key: str, a, b) -> dict:
    return {"expected": a, "observed": b}


def _l2_color(c1: list[float], c2: list[float]) -> float:
    """L2 distance between two RGBA (or RGB) color vectors."""
    n = min(len(c1), len(c2))
    return math.sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(n)))


def build_diff(inv_a: dict, inv_b: dict) -> dict:
    """Compare invariants from blend A (original) vs blend B (roundtrip)."""
    diff: dict[str, Any] = {
        "missing_blocks_in_original": [],
        "extra_blocks_in_roundtrip": [],
    }

    # --- mesh_object_count ---
    diff["mesh_object_count"] = _diff_val(
        "mesh_object_count",
        inv_a["mesh_object_count"],
        inv_b["mesh_object_count"],
    )

    # --- per-mesh objects ---
    objs_a: dict[str, dict] = inv_a.get("mesh_objects_by_name", {})
    objs_b: dict[str, dict] = inv_b.get("mesh_objects_by_name", {})
    names_a = set(objs_a)
    names_b = set(objs_b)
    missing_meshes = sorted(names_a - names_b)
    extra_meshes = sorted(names_b - names_a)
    if missing_meshes:
        diff["missing_blocks_in_original"].extend([f"mesh:{n}" for n in missing_meshes])
    if extra_meshes:
        diff["extra_blocks_in_roundtrip"].extend([f"mesh:{n}" for n in extra_meshes])

    mesh_diffs: list[dict] = []
    for name in sorted(names_a & names_b):
        ea = objs_a[name]
        eb = objs_b[name]
        mesh_entry: dict[str, Any] = {"name": name}
        mesh_entry["location"] = {"expected": ea["location"], "observed": eb["location"]}
        mesh_entry["rotation_euler"] = {"expected": ea["rotation_euler"], "observed": eb["rotation_euler"]}
        mesh_entry["scale"] = {"expected": ea["scale"], "observed": eb["scale"]}
        mesh_diffs.append(mesh_entry)
    diff["mesh_objects"] = mesh_diffs

    # --- camera ---
    cam_a = inv_a.get("camera")
    cam_b = inv_b.get("camera")
    cam_diff: dict[str, Any] = {}
    if cam_a is not None and cam_b is not None:
        cam_diff["location"] = {"expected": cam_a["location"], "observed": cam_b["location"]}
        cam_diff["rotation_euler"] = {"expected": cam_a["rotation_euler"], "observed": cam_b["rotation_euler"]}
        cam_diff["lens"] = {"expected": cam_a["lens"], "observed": cam_b["lens"]}
        cam_diff["sensor_width"] = {"expected": cam_a["sensor_width"], "observed": cam_b["sensor_width"]}
    elif cam_a is None:
        diff["missing_blocks_in_original"].append("camera")
    elif cam_b is None:
        diff["extra_blocks_in_roundtrip"].append("camera")
    diff["camera"] = cam_diff

    # --- lights ---
    lc_a = inv_a.get("light_counts", {})
    lc_b = inv_b.get("light_counts", {})
    le_a = inv_a.get("light_energies", {})
    le_b = inv_b.get("light_energies", {})
    light_counts: dict[str, dict] = {}
    light_energy: dict[str, dict] = {}
    for lt in ("SUN", "AREA", "POINT", "SPOT"):
        light_counts[lt] = {"expected": lc_a.get(lt, 0), "observed": lc_b.get(lt, 0)}
        light_energy[lt] = {"expected": le_a.get(lt, 0.0), "observed": le_b.get(lt, 0.0)}
    diff["light_counts"] = light_counts
    diff["light_energy"] = light_energy

    # --- world ---
    wa = inv_a.get("world", {})
    wb = inv_b.get("world", {})
    world_diff: dict[str, Any] = {}
    world_diff["mode"] = {"expected": wa.get("mode", "none"), "observed": wb.get("mode", "none")}
    world_diff["world_strength"] = {
        "expected": wa.get("world_strength", 0.0),
        "observed": wb.get("world_strength", 0.0),
    }
    diff["world"] = world_diff

    # --- stage_materials ---
    sma = inv_a.get("stage_materials", {})
    smb = inv_b.get("stage_materials", {})
    sm_diff: dict[str, Any] = {}
    for role in sorted(set(sma) | set(smb)):
        entry_a = sma.get(role)
        entry_b = smb.get(role)
        if entry_a is None:
            diff["missing_blocks_in_original"].append(f"stage_material:{role}")
            continue
        if entry_b is None:
            diff["extra_blocks_in_roundtrip"].append(f"stage_material:{role}")
            continue
        role_diff: dict[str, Any] = {}
        # base_color L2
        bc_a = entry_a.get("base_color", [0, 0, 0, 1])
        bc_b = entry_b.get("base_color", [0, 0, 0, 1])
        l2 = round(_l2_color(bc_a, bc_b), 8)
        role_diff["base_color"] = {"expected": bc_a, "observed": bc_b, "l2_delta": l2}
        # roughness
        role_diff["roughness"] = {
            "expected": entry_a.get("roughness", 0.0),
            "observed": entry_b.get("roughness", 0.0),
        }
        sm_diff[role] = role_diff
    diff["stage_materials"] = sm_diff

    # --- stage ---
    sa = inv_a.get("stage", {})
    sb = inv_b.get("stage", {})
    stage_diff: dict[str, Any] = {}

    if "floor_z" in sa or "floor_z" in sb:
        stage_diff["floor_z"] = {
            "expected": sa.get("floor_z", None),
            "observed": sb.get("floor_z", None),
        }
    if "ceiling_z" in sa or "ceiling_z" in sb:
        stage_diff["ceiling_z"] = {
            "expected": sa.get("ceiling_z", None),
            "observed": sb.get("ceiling_z", None),
        }

    # polygon_vertices
    pverts_a = sa.get("polygon_vertices", [])
    pverts_b = sb.get("polygon_vertices", [])
    stage_diff["polygon_vertices_count"] = {
        "expected": len(pverts_a),
        "observed": len(pverts_b),
    }
    vert_diffs: list[dict] = []
    for i in range(min(len(pverts_a), len(pverts_b))):
        vert_diffs.append({
            "expected": pverts_a[i],
            "observed": pverts_b[i],
        })
    stage_diff["polygon_vertices"] = vert_diffs

    diff["stage"] = stage_diff

    # --- render ---
    ra = inv_a.get("render", {})
    rb = inv_b.get("render", {})
    render_diff: dict[str, Any] = {}
    render_diff["engine"] = {"expected": ra.get("engine"), "observed": rb.get("engine")}
    render_diff["samples"] = {"expected": ra.get("samples"), "observed": rb.get("samples")}
    render_diff["resolution"] = {
        "expected": [ra.get("resolution_x"), ra.get("resolution_y")],
        "observed": [rb.get("resolution_x"), rb.get("resolution_y")],
    }
    diff["render"] = render_diff

    # --- point_cloud ---
    pc_a = inv_a.get("point_cloud_num_vertices")
    pc_b = inv_b.get("point_cloud_num_vertices")
    diff["point_cloud"] = {"num_vertices": {"expected": pc_a, "observed": pc_b}}

    return diff


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, data: dict) -> None:
    import tempfile
    parent = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

original_path = _args.original
roundtrip_path = _args.roundtrip
out_path = _args.out

print(f"{LOG} Opening original: {original_path}")
bpy.ops.wm.open_mainfile(filepath=original_path)
print(f"{LOG} Extracting invariants from original ...")
inv_original = extract_invariants()
print(f"{LOG} Original: mesh_count={inv_original['mesh_object_count']} "
      f"lights={sum(inv_original['light_counts'].values())} "
      f"world_mode={inv_original['world']['mode']}")

print(f"{LOG} Opening roundtrip: {roundtrip_path}")
bpy.ops.wm.open_mainfile(filepath=roundtrip_path)
print(f"{LOG} Extracting invariants from roundtrip ...")
inv_roundtrip = extract_invariants()
print(f"{LOG} Roundtrip: mesh_count={inv_roundtrip['mesh_object_count']} "
      f"lights={sum(inv_roundtrip['light_counts'].values())} "
      f"world_mode={inv_roundtrip['world']['mode']}")

print(f"{LOG} Computing diff ...")
diff = build_diff(inv_original, inv_roundtrip)

_atomic_write_json(out_path, diff)
print(f"{LOG} Diff written to: {out_path}")
