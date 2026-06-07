"""
build_stage_v2.py
-----------------
Blender script — build Floor + Wall_NN (per WALL edge) + Ceiling from polygon_v2.json.

CLI (after `--`):
  --scene-dir PATH     absolute path to scene folder

Library API (importable from another Blender Python context):
  from build_stage_v2 import build_from_polygon_dict
  report = build_from_polygon_dict(stage_dict, "blender_scene.blend")
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy   # type: ignore
import bmesh # type: ignore

# ---------------------------------------------------------------------------
# Required keys in stage_dict
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = (
    "polygon_vertices",
    "floor_z",
    "ceiling_z",
    "wall_thickness",
    "floor_thickness",
    "ceiling_thickness",
)


# ---------------------------------------------------------------------------
# Collection / material helpers
# ---------------------------------------------------------------------------

def ensure_stage_collection() -> "bpy.types.Collection":
    coll = bpy.data.collections.get("Stage")
    if coll is None:
        coll = bpy.data.collections.new("Stage")
        bpy.context.scene.collection.children.link(coll)
    return coll


def link_exclusive(obj: "bpy.types.Object", target_coll: "bpy.types.Collection") -> None:
    for c in list(obj.users_collection):
        if c != target_coll:
            c.objects.unlink(obj)
    if target_coll not in obj.users_collection:
        target_coll.objects.link(obj)


def ensure_material(name: str) -> "bpy.types.Material":
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.85, 1.0)
            bsdf.inputs["Roughness"].default_value = 1.0
    mat.use_fake_user = True
    return mat


def _assign_material(mesh: "bpy.types.Mesh", name: str) -> None:
    mat = ensure_material(name)
    if mesh.materials:
        mesh.materials[0] = mat
    else:
        mesh.materials.append(mat)


def _remove_existing_stage() -> None:
    for obj_name in ("Floor", "Ceiling"):
        obj = bpy.data.objects.get(obj_name)
        if obj is not None:
            m = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if m is not None and m.users == 0:
                bpy.data.meshes.remove(m, do_unlink=True)
    for obj in [o for o in list(bpy.data.objects) if o.name.startswith("Wall_")]:
        m = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if m is not None and m.users == 0:
            bpy.data.meshes.remove(m, do_unlink=True)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def build_slab(
    name: str,
    xy_vertices: list[list[float]],
    base_z: float,
    thickness: float,
    extrude_down: bool,
    material_name: str,
    stage_coll: "bpy.types.Collection",
) -> "bpy.types.Object":
    """Build a flat n-gon slab (Floor or Ceiling) at base_z, extruded by thickness."""
    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    link_exclusive(obj, stage_coll)

    bm = bmesh.new()
    verts = [bm.verts.new((x, y, base_z)) for x, y in xy_vertices]
    bm.verts.ensure_lookup_table()
    face = bm.faces.new(verts)
    face.normal_update()
    if face.normal.z < 0:
        bmesh.ops.reverse_faces(bm, faces=[face])
        bm.faces.ensure_lookup_table()
        face = bm.faces[0]
        face.normal_update()

    ret = bmesh.ops.extrude_face_region(bm, geom=[face])
    new_verts = [e for e in ret["geom"] if isinstance(e, bmesh.types.BMVert)]
    dz = -thickness if extrude_down else thickness
    bmesh.ops.translate(bm, verts=new_verts, vec=(0.0, 0.0, dz))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    bm.to_mesh(mesh)
    bm.free()
    _assign_material(mesh, material_name)
    return obj


def compute_inward_normal_ccw(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float]:
    """For a CCW polygon, inward normal of edge A→B = (B-A) rotated +90°, normalized."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    nx, ny = -dy, dx
    mag = math.sqrt(nx * nx + ny * ny) + 1e-12
    return (nx / mag, ny / mag)


def build_wall(
    name: str,
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    floor_z: float,
    ceiling_z: float,
    thickness: float,
    inward_normal: tuple[float, float],
    material_name: str,
    stage_coll: "bpy.types.Collection",
) -> "bpy.types.Object":
    """Build one wall quad with inward-facing normal; Solidify offset=-1.0; apply modifier."""
    ax, ay = a_xy
    bx, by = b_xy
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj  = bpy.data.objects.new(name, mesh)
    link_exclusive(obj, stage_coll)

    bm = bmesh.new()
    bvs = [bm.verts.new(p) for p in (
        (ax, ay, floor_z), (bx, by, floor_z),
        (bx, by, ceiling_z), (ax, ay, ceiling_z),
    )]
    bm.verts.ensure_lookup_table()
    face = bm.faces.new(bvs)
    face.normal_update()

    fx, fy = face.normal.x, face.normal.y
    dot = fx * inward_normal[0] + fy * inward_normal[1]
    if dot < 0:
        bmesh.ops.reverse_faces(bm, faces=[face])
        bm.faces.ensure_lookup_table()
        face = bm.faces[0]
        face.normal_update()

    bm.to_mesh(mesh)
    bm.free()
    _assign_material(mesh, material_name)

    sol = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    sol.thickness         = thickness
    sol.offset            = -1.0
    sol.use_even_offset   = True
    sol.use_quality_normals = True

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=sol.name)
    obj.select_set(False)
    return obj


# ---------------------------------------------------------------------------
# Opening helpers (preserved for external callers; not exercised by CLI)
# ---------------------------------------------------------------------------

def _apply_opening_to_wall(
    obj: "bpy.types.Object",
    opening: dict,
    wall_thickness: float,
    errors: list[str],
) -> bool:
    """Cut a rectangular opening through a solidified wall using bmesh bisect ops.

    Returns True on success, False on failure (error appended to errors).
    Preserved for Stage-2 callers that pass openings[]; not used by this skill's CLI.
    """
    try:
        (x0, y0), (x1, y1) = opening["xy_range"]
        z0, z1 = opening["z_range"]
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"opening {opening.get('id','?')}: malformed fields: {exc}")
        return False

    opening_id = opening.get("id", "?")
    aabb_x_min, aabb_x_max = min(x0, x1), max(x0, x1)
    aabb_y_min, aabb_y_max = min(y0, y1), max(y0, y1)
    aabb_z_min, aabb_z_max = min(z0, z1), max(z0, z1)

    mesh = obj.data
    mw   = obj.matrix_world
    mw_inv = mw.inverted()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(mw)
    saved_bm = bmesh.new()
    saved_bm.from_mesh(mesh)

    def _bisect4(bm_t: bmesh.types.BMesh) -> None:
        normals_xy = []
        for f in bm_t.faces:
            zs = [v.co.z for v in f.verts]
            if max(zs) - min(zs) > 0.1:
                normals_xy.append((f.normal.x, f.normal.y))
        if normals_xy:
            avg_nx = sum(n[0] for n in normals_xy) / len(normals_xy)
            avg_ny = sum(n[1] for n in normals_xy) / len(normals_xy)
            mag = math.sqrt(avg_nx**2 + avg_ny**2) + 1e-12
            avg_nx /= mag; avg_ny /= mag
        else:
            avg_nx, avg_ny = 0.0, 0.0
        tang_x, tang_y = -avg_ny, avg_nx

        def proj_t(px, py):
            return px * tang_x + py * tang_y

        tvals = [proj_t(aabb_x_min, aabb_y_min), proj_t(aabb_x_min, aabb_y_max),
                 proj_t(aabb_x_max, aabb_y_min), proj_t(aabb_x_max, aabb_y_max)]
        t_min, t_max = min(tvals), max(tvals)

        for plane_co, plane_no in [
            ((0, 0, aabb_z_min), (0, 0, -1)),
            ((0, 0, aabb_z_max), (0, 0, 1)),
            ((t_min*tang_x, t_min*tang_y, 0), (-tang_x, -tang_y, 0)),
            ((t_max*tang_x, t_max*tang_y, 0), (tang_x, tang_y, 0)),
        ]:
            bmesh.ops.bisect_plane(
                bm_t,
                geom=list(bm_t.verts)+list(bm_t.edges)+list(bm_t.faces),
                plane_co=plane_co, plane_no=plane_no,
                clear_outer=False, clear_inner=False, use_snap_center=False,
            )
            bm_t.verts.ensure_lookup_table()
            bm_t.edges.ensure_lookup_table()
            bm_t.faces.ensure_lookup_table()

    _bisect4(bm)

    SHRINK = 0.005
    interior = [
        f for f in bm.faces
        if (aabb_x_min+SHRINK <= sum(v.co.x for v in f.verts)/len(f.verts) <= aabb_x_max-SHRINK
            and aabb_y_min+SHRINK <= sum(v.co.y for v in f.verts)/len(f.verts) <= aabb_y_max-SHRINK
            and aabb_z_min+SHRINK <= sum(v.co.z for v in f.verts)/len(f.verts) <= aabb_z_max-SHRINK)
    ]
    if not interior:
        errors.append(f"opening {opening_id}: no interior faces found; check xy_range vs wall extents")
        bm.free(); saved_bm.free()
        return False

    bmesh.ops.delete(bm, geom=interior, context="FACES_ONLY")
    bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    boundary = [e for e in bm.edges if len(e.link_faces) < 2]
    if boundary:
        try:
            bmesh.ops.holes_fill(bm, edges=boundary, sides=4)
        except Exception:
            try:
                bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=boundary)
            except Exception:
                pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    bad = [e for e in bm.edges if len(e.link_faces) != 2]
    if bad:
        errors.append(f"opening {opening_id}: non-manifold ({len(bad)} edges); reverted")
        bm.free()
        mesh.clear_geometry()
        saved_bm.to_mesh(mesh)
        saved_bm.free()
        return False

    bm.transform(mw_inv)
    bm.to_mesh(mesh)
    bm.free(); saved_bm.free()
    mesh.update()
    print(f"[build_stage_v2] Applied opening {opening_id} to {obj.name}.")
    return True


# ---------------------------------------------------------------------------
# PointCloud_XZ removal
# ---------------------------------------------------------------------------

def remove_pointcloud() -> None:
    """Delete any PointCloud_XZ object and its mesh datablock."""
    pc_objs = [o for o in list(bpy.data.objects)
               if o.name == "PointCloud_XZ" or o.name.startswith("PointCloud_XZ.")]
    for obj in pc_objs:
        mesh_name = obj.data.name if obj.data else None
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh_name and mesh_name in bpy.data.meshes:
            m = bpy.data.meshes[mesh_name]
            if m.users == 0:
                bpy.data.meshes.remove(m, do_unlink=True)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m, do_unlink=True)


# ---------------------------------------------------------------------------
# Public library API  (do not change signature — depended on by stage2-sub-pointmap-to-separable-stage
# and stage-op-executor)
# ---------------------------------------------------------------------------

def build_from_polygon_dict(
    stage_dict: dict,
    blend_path: str,
    *,
    save: bool = True,
    replace_existing: bool = True,
) -> dict:
    """Rebuild the Stage collection (Floor, Wall_NN, Ceiling) from a stage dict.

    Must be called from inside a Blender process (imports bpy).

    Parameters
    ----------
    stage_dict:
        The "stage" block from blender_scene.json. Required keys:
          polygon_vertices, floor_z, ceiling_z, wall_thickness,
          floor_thickness, ceiling_thickness.
        Optional keys:
          open_edges   — list of {from,to} dicts OR list of int edge indices.
          openings     — list of opening descriptors (window/door/arch cuts).
          wall_edges   — informational; not consumed here (edges are derived from
                         open_edges).
    blend_path:
        Absolute path for the save destination (.blend).
    save:
        If True (default), save the .blend after building.
    replace_existing:
        If True (default), delete existing Floor/Wall_NN/Ceiling first.

    Returns
    -------
    dict with keys: wall_names, floor_z, ceiling_z, openings_applied,
                    manifold_ok, errors
    """
    missing = [k for k in _REQUIRED_KEYS if k not in stage_dict]
    if missing:
        raise ValueError(f"build_from_polygon_dict: stage_dict missing required key(s): {missing}")

    verts: list[list[float]] = stage_dict["polygon_vertices"]
    floor_z:           float = float(stage_dict["floor_z"])
    ceiling_z:         float = float(stage_dict["ceiling_z"])
    wall_thickness:    float = float(stage_dict["wall_thickness"])
    floor_thickness:   float = float(stage_dict["floor_thickness"])
    ceiling_thickness: float = float(stage_dict["ceiling_thickness"])

    # Decode open_edges — support both {from,to} dicts and bare int (edge-index).
    open_edges_raw = stage_dict.get("open_edges", [])
    open_edge_pairs: set[tuple[int, int]] = set()
    n_verts = len(verts)
    for item in open_edges_raw:
        if isinstance(item, dict):
            open_edge_pairs.add((int(item["from"]), int(item["to"])))
        elif isinstance(item, int):
            v_from = item % n_verts
            v_to   = (item + 1) % n_verts
            open_edge_pairs.add((v_from, v_to))

    openings: list[dict] = stage_dict.get("openings", [])
    errors:   list[str]  = []

    if replace_existing:
        _remove_existing_stage()

    ensure_material("Mat_Floor")
    ensure_material("Mat_Wall")
    ensure_material("Mat_Ceiling")
    stage_coll = ensure_stage_collection()

    # Floor
    build_slab("Floor", verts, floor_z, floor_thickness, extrude_down=True,
               material_name="Mat_Floor", stage_coll=stage_coll)

    # Walls — one per non-OPEN edge
    wall_index = 0
    wall_names: list[str] = []
    wall_edge_map: dict[str, dict] = {}

    for i in range(n_verts):
        v_from = i
        v_to   = (i + 1) % n_verts
        if (v_from, v_to) in open_edge_pairs or (v_to, v_from) in open_edge_pairs:
            continue
        wall_index += 1
        a = (float(verts[v_from][0]), float(verts[v_from][1]))
        b = (float(verts[v_to][0]),   float(verts[v_to][1]))
        in_n = compute_inward_normal_ccw(a, b)
        name = f"Wall_{wall_index:02d}"
        build_wall(name, a, b, floor_z, ceiling_z, wall_thickness, in_n,
                   "Mat_Wall", stage_coll)
        wall_names.append(name)
        wall_edge_map[name] = {"a_xy": a, "b_xy": b, "floor_z": floor_z, "ceiling_z": ceiling_z}

    # Ceiling
    build_slab("Ceiling", verts, ceiling_z, ceiling_thickness, extrude_down=False,
               material_name="Mat_Ceiling", stage_coll=stage_coll)

    bpy.context.view_layer.update()

    # Openings (cuts) — preserved for external callers; not exercised by this skill's CLI
    openings_applied = 0
    for opening in openings:
        wall_name  = opening.get("wall_name")
        opening_id = opening.get("id", "?")
        if not wall_name:
            errors.append(f"opening {opening_id}: missing 'wall_name'; skipped")
            continue
        wall_obj = bpy.data.objects.get(wall_name)
        if wall_obj is None:
            errors.append(f"opening {opening_id}: wall '{wall_name}' not found; skipped")
            continue
        ok = _apply_opening_to_wall(wall_obj, opening, wall_thickness, errors)
        if ok:
            openings_applied += 1

    bpy.context.view_layer.update()

    if save:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"[build_stage_v2] Saved {blend_path}")

    # Manifold check
    manifold_ok = True
    for wname in wall_names:
        wobj = bpy.data.objects.get(wname)
        if wobj is None:
            continue
        bm_chk = bmesh.new()
        bm_chk.from_mesh(wobj.data)
        bad = [e for e in bm_chk.edges if len(e.link_faces) != 2]
        bm_chk.free()
        if bad:
            manifold_ok = False
            break

    return {
        "wall_names":       wall_names,
        "floor_z":          floor_z,
        "ceiling_z":        ceiling_z,
        "openings_applied": openings_applied,
        "manifold_ok":      manifold_ok,
        "errors":           errors,
    }


# ---------------------------------------------------------------------------
# CLI implementation
# ---------------------------------------------------------------------------

def _merge_stage_block(
    json_path: Path,
    verts: list,
    wall_names: list[str],
    polygon_data: dict,
    floor_z: float,
    ceiling_z: float,
) -> None:
    """Merge the stage block into blender_scene.json per the schema (§11)."""
    scene_json = json.loads(json_path.read_text()) if json_path.exists() else {}

    scene_json["stage"] = {
        "polygon_vertices":    polygon_data.get("polygon_vertices", verts),
        "polygon_centroid_xy": polygon_data.get("polygon_centroid_xy", None),
        "floor_z":             floor_z,
        "ceiling_z":           ceiling_z,
        "wall_thickness":      polygon_data.get("wall_thickness", 0.25),
        "floor_thickness":     polygon_data.get("floor_thickness", 0.30),
        "ceiling_thickness":   polygon_data.get("ceiling_thickness", 0.30),
        "wall_objects":        wall_names,
        "wall_edges":          polygon_data.get("wall_edges", []),
        "open_edges":          polygon_data.get("open_edges", []),
        "openings":            polygon_data.get("openings", []),
        "camera_xy":           polygon_data.get("camera_xy", [0.0, 0.0]),
        "camera_source":       polygon_data.get("camera_source", "blender_camera"),
        "source_frame":        polygon_data.get("source_frame", "blend_world"),
        "buffer_m":            polygon_data.get("buffer_m", 0.08),
        "rect_angle_deg":      polygon_data.get("rect_angle_deg", polygon_data.get("yaw_deg", 0.0)),
        "generator":           "stage2-sub-pointmap-to-separable-stage",
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(scene_json, indent=2))


def _build_cli(scene_dir: Path) -> int:
    # Locate polygon_v2.json
    poly_new    = scene_dir / "json" / "polygon_v2.json"
    poly_legacy = scene_dir / "polygon_v2.json"
    if poly_new.exists():
        polygon_path = poly_new
    elif poly_legacy.exists():
        print(f"[build_stage_v2] WARNING: using legacy path {poly_legacy}", file=sys.stderr)
        polygon_path = poly_legacy
    else:
        print(f"[build_stage_v2] ERROR: polygon_v2.json not found at {poly_new}", file=sys.stderr)
        return 2

    polygon = json.loads(polygon_path.read_text())

    # polygon_v2.json uses "polygon_vertices" (already the new key)
    verts     = polygon["polygon_vertices"]
    floor_z   = float(polygon["floor_z"])
    ceiling_z = float(polygon["ceiling_z"])

    open_edges_dicts = polygon.get("open_edges", [])

    stage_dict = {
        "polygon_vertices":  verts,
        "floor_z":           floor_z,
        "ceiling_z":         ceiling_z,
        "wall_thickness":    float(polygon.get("wall_thickness", 0.25)),
        "floor_thickness":   float(polygon.get("floor_thickness", 0.30)),
        "ceiling_thickness": float(polygon.get("ceiling_thickness", 0.30)),
        "open_edges":        open_edges_dicts,
        "openings":          [],
    }

    blend_path = str(bpy.data.filepath)
    report = build_from_polygon_dict(stage_dict, blend_path, save=False)

    # Remove PointCloud_XZ
    remove_pointcloud()

    # Merge stage block into blender_scene.json
    json_path = scene_dir / "json" / "blender_scene.json"
    _merge_stage_block(json_path, verts, report["wall_names"], polygon, floor_z, ceiling_z)
    print(f"[build_stage_v2] Merged stage block into {json_path}")

    # Save
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    n_open = len([e for e in open_edges_dicts])
    print(
        f"[build_stage_v2] floor=Floor walls={len(report['wall_names'])} "
        f"openings_skipped={n_open} ceiling=Ceiling "
        f"pointcloud_removed=True manifold_ok={report['manifold_ok']}"
    )
    if report["errors"]:
        for err in report["errors"]:
            print(f"  WARNING: {err}")
    return 0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    return _build_cli(args.scene_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
