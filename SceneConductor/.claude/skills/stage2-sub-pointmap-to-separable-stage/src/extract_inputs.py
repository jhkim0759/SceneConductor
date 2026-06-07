"""Blender script: delete stale stage geometry, ensure materials, export pointcloud/camera/bbox data."""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore

SAMPLE_SIZE = 60000
SAMPLE_SEED = 0

MAT_DEFAULTS = {
    "Mat_Floor":   (0.85, 0.85, 0.85, 1.0),
    "Mat_Wall":    (0.85, 0.85, 0.85, 1.0),
    "Mat_Ceiling": (0.85, 0.85, 0.85, 1.0),
}

STAGE_EXACT   = {"Room_Shell", "floor", "Floor", "Ceiling", "Stage"}
STAGE_PREFIX  = ("Wall_", "Stage_", "Floor.", "Ceiling.", "Wall.")


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    return ap.parse_args(argv)


def is_stage(name):
    return name in STAGE_EXACT or any(name.startswith(p) for p in STAGE_PREFIX)


def ensure_materials():
    for name, rgba in MAT_DEFAULTS.items():
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = rgba
                bsdf.inputs["Roughness"].default_value = 1.0
        mat.use_fake_user = True


def delete_stage_objects():
    stale = [o for o in list(bpy.data.objects) if is_stage(o.name)]
    for obj in stale:
        mesh = obj.data if obj.type == "MESH" else None
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh, do_unlink=True)
    # Unlink empty Stage collection if present
    stage_coll = bpy.data.collections.get("Stage")
    if stage_coll is not None and len(stage_coll.objects) == 0:
        if stage_coll.name in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.unlink(stage_coll)


def export_pointcloud(slug, scale_factor):
    pc = bpy.data.objects["PointCloud_XZ"]
    mw = pc.matrix_world
    n = len(pc.data.vertices)
    all_coords = np.empty((n, 3), dtype=np.float64)
    for i, v in enumerate(pc.data.vertices):
        wv = mw @ v.co
        # PointCloud_XZ is imported by build.py (ply_import.py) with axis_remap
        # forward=Z,up=Y and world_scale_applied=meta.world_scale_factor already
        # baked into vertex.co. Read world coords as-is — no second axis swap or
        # scale (the historical (-x, z, y) * scale was a double transform that
        # inflated pc_z by ≈ scale_factor², saved only by the 8.5 m ceiling cap).
        all_coords[i] = (wv.x, wv.y, wv.z)
    rng = np.random.default_rng(SAMPLE_SEED)
    k = min(SAMPLE_SIZE, n)
    idx = rng.choice(n, k, replace=False) if k < n else np.arange(n)
    xy = all_coords[idx, :2].astype(np.float32)
    np.save(f"/tmp/{slug}_pointcloud_xy.npy", xy)
    z = all_coords[:, 2]
    with open(f"/tmp/{slug}_pc_z.json", "w") as f:
        json.dump({"min": float(z.min()), "max": float(z.max())}, f)


def export_camera(slug):
    cam = bpy.data.objects.get("Camera") or bpy.context.scene.camera
    loc = cam.matrix_world.translation
    # Yaw: angle of camera forward direction projected to XY plane.
    # Blender camera looks down local -Z; forward_world = R @ (0,0,-1).
    R = cam.matrix_world.to_3x3()
    fwd = R @ Vector((0.0, 0.0, -1.0))
    yaw_deg = math.degrees(math.atan2(-fwd.x, -fwd.y))
    with open(f"/tmp/{slug}_camera.json", "w") as f:
        json.dump({
            "x": float(loc.x),
            "y": float(loc.y),
            "z": float(loc.z),
            "yaw_deg": float(yaw_deg),
        }, f)


def export_object_bboxes(slug):
    result = {}
    for obj in bpy.data.objects:
        if not obj.name.startswith("geometry_"):
            continue
        mw = obj.matrix_world
        corners = [[*(mw @ Vector(c))] for c in obj.bound_box]
        result[obj.name] = corners
    with open(f"/tmp/{slug}_object_bboxes.json", "w") as f:
        json.dump(result, f)


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    slug = scene_dir.name
    blend_path = str(scene_dir / "blend" / "blender_scene.blend")

    scene_json = json.loads((scene_dir / "json" / "blender_scene.json").read_text())
    scale_factor = float(scene_json.get("meta", {}).get("world_scale_factor", 1.0))

    ensure_materials()
    delete_stage_objects()
    export_pointcloud(slug, scale_factor)
    export_camera(slug)
    export_object_bboxes(slug)

    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[extract_inputs] slug={slug} scale={scale_factor:.3f} blend_saved={blend_path}")


if __name__ == "__main__":
    main()
