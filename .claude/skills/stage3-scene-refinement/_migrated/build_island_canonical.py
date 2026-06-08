#!/usr/bin/env python3
"""Clean canonical-space island builder for scene-relation-graph.

Design (single rigid transform, applied uniformly):

  1. Read the anchor object's WORLD rigid pose (R, T).  No scale.
       M_anchor = Translate(T) @ Rotate(R)
  2. Compute the SINGLE inverse rigid transform:
       M_inv = M_anchor^-1
  3. Apply M_inv UNIFORMLY to every member's loc+rot:
       new_loc, new_rot = decompose( M_inv @ old_rigid_matrix )
     -> anchor lands at (loc=0, rot=0), members keep their RELATIVE poses.
     -> scale is NOT part of the rigid transform; left untouched.
  4. Delete everything outside the keep-set (group + Floor + lights + Camera).
  5. Save island.blend + metadata.json.  Metadata records M_anchor so the
     inverse transform (canonical -> world) is just `M_anchor @ canonical`.

This intentionally REPLACES scene-operation-planner/build_group_island.py for
relation-graph use only — the other script has a Z double-shift bug when its
--anchor-z-floor flag is on.  We don't touch it because other skills depend
on its (round-trippable, but visually broken) behavior.

Run via:
    blender --background <main_scene.blend> \
        --python <THIS> -- \
        --output-blend <island.blend> \
        --metadata    <metadata.json> \
        --group-name  G1 \
        --anchor-id   obj_6 \
        --member-ids  obj_2 obj_4 obj_5 obj_8 obj_27 obj_28
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


def _find_args() -> list[str]:
    if "--" not in sys.argv:
        raise SystemExit("missing -- before script args")
    return sys.argv[sys.argv.index("--") + 1:]


def _loc_rot_matrix(loc: mathutils.Vector, rot_euler: mathutils.Euler) -> mathutils.Matrix:
    """Compose a 4x4 rigid (no scale) matrix from location + euler rotation."""
    return mathutils.Matrix.Translation(loc) @ rot_euler.to_matrix().to_4x4()


def _delete_outside_keep_set(keep_names: set[str]) -> int:
    queue = [bpy.data.objects[n] for n in keep_names if n in bpy.data.objects]
    extended = set(keep_names)
    while queue:
        o = queue.pop()
        for child in o.children:
            if child.name not in extended:
                extended.add(child.name)
                queue.append(child)
    removed = 0
    for name in [o.name for o in bpy.data.objects if o.name not in extended]:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
            removed += 1
    return removed


def main() -> None:
    raw = _find_args()
    p = argparse.ArgumentParser()
    p.add_argument("--output-blend", required=True)
    p.add_argument("--metadata",     required=True)
    p.add_argument("--group-name",   required=True)
    p.add_argument("--anchor-id",    required=True)
    p.add_argument("--member-ids",   required=True, nargs="+",
                   help="Other group members. May INCLUDE the anchor-id (treated as anchor).")
    # By default the island is STRIPPED: existing stage (Floor/Wall*/Ceiling)
    # and ALL existing lights are deleted, and only a single basic Sun is added.
    # Pass --keep-original-stage / --keep-original-lights to preserve them.
    p.add_argument("--keep-original-stage",  action="store_true", default=False)
    p.add_argument("--keep-original-lights", action="store_true", default=False)
    p.add_argument("--no-add-sun", action="store_true", default=False,
                   help="Skip adding a basic Sun light.")
    p.add_argument("--use-scene-camera", default=None, metavar="CAMERA_OBJ_NAME",
                   help=(
                       "When set, skip synthetic-camera creation and instead use the "
                       "named camera object from the source .blend as the scene camera. "
                       "The camera pose is left exactly as-is (no movement/rotation). "
                       "If the named object is not found, falls back to the synthetic camera."
                   ))
    args = p.parse_args(raw)

    anchor_name = args.anchor_id
    member_names = list(args.member_ids)
    if anchor_name not in member_names:
        member_names.append(anchor_name)

    anchor = bpy.data.objects.get(anchor_name)
    if anchor is None:
        raise SystemExit(f"anchor object {anchor_name!r} not found in current .blend")

    members = []
    for name in member_names:
        o = bpy.data.objects.get(name)
        if o is None:
            print(f"[island] WARN: member {name!r} not found, skipping", file=sys.stderr)
            continue
        members.append(o)

    # --- Step 1: snapshot anchor's WORLD rigid pose (R, T)
    anchor_loc = anchor.location.copy()
    anchor_rot = anchor.rotation_euler.copy()
    M_anchor = _loc_rot_matrix(anchor_loc, anchor_rot)

    # --- Step 2: inverse rigid transform — the SINGLE matrix applied to all members
    M_inv = M_anchor.inverted()

    # --- Step 2.5: snapshot each member's pre-transform loc/rot/scale
    snapshot = {
        o.name: {
            "world_location":       list(o.location),
            "world_rotation_euler": list(o.rotation_euler),
            "world_scale":          list(o.scale),
        }
        for o in members
    }

    # --- Step 3a: decide what to keep
    # Default behavior: STRIPPED island.
    #   - Members + (optionally) Camera kept.
    #   - Existing stage (Floor/Wall_*/Ceiling) deleted unless --keep-original-stage.
    #   - All existing LIGHT objects deleted unless --keep-original-lights.
    # We then add ONE basic Sun and ONE basic Floor plane (unless suppressed).
    STAGE_NAMES = {"Floor", "Ceiling"}

    def _is_original_stage(o):
        return (o.name in STAGE_NAMES) or o.name.startswith("Wall_")

    keep = {anchor.name, *(m.name for m in members), "Camera"}
    # When --use-scene-camera names a camera that differs from "Camera", ensure it
    # is not deleted before Step 4d references it.
    if args.use_scene_camera:
        keep.add(args.use_scene_camera)
    if args.keep_original_stage:
        for o in bpy.data.objects:
            if _is_original_stage(o):
                keep.add(o.name)
    if args.keep_original_lights:
        for o in bpy.data.objects:
            if o.type == "LIGHT":
                keep.add(o.name)

    n_removed = _delete_outside_keep_set(keep)
    print(f"[island] kept {len(keep)} objects, deleted {n_removed}")

    # --- Step 3b: apply M_inv UNIFORMLY to each member's loc+rot (scale untouched)
    members_meta = {}
    for o in members:
        old_loc = mathutils.Vector(snapshot[o.name]["world_location"])
        old_rot = mathutils.Euler(snapshot[o.name]["world_rotation_euler"], "XYZ")
        old_rigid = _loc_rot_matrix(old_loc, old_rot)
        new_rigid = M_inv @ old_rigid
        new_loc = new_rigid.translation
        new_rot = new_rigid.to_euler("XYZ")

        o.location = (new_loc.x, new_loc.y, new_loc.z)
        o.rotation_mode = "XYZ"
        o.rotation_euler = (new_rot.x, new_rot.y, new_rot.z)
        # o.scale intentionally unchanged — scale is not part of the rigid transform.

        members_meta[o.name] = {
            "world_location":       snapshot[o.name]["world_location"],
            "world_rotation_euler": snapshot[o.name]["world_rotation_euler"],
            "world_scale":          snapshot[o.name]["world_scale"],
            "canonical_location":   list(new_loc),
            "canonical_rotation_euler": list(new_rot),
            "canonical_scale":      snapshot[o.name]["world_scale"],
            "is_anchor":            (o.name == anchor.name),
        }
    bpy.context.view_layer.update()

    # --- Step 4a: compute the canonical-space bbox of the group (for camera + ground sizing)
    xs, ys, zs = [0.0], [0.0], [0.0]
    for o in members:
        xs.append(o.location.x); ys.append(o.location.y); zs.append(o.location.z)
        d = list(o.dimensions) if hasattr(o, "dimensions") else [0, 0, 0]
        xs += [o.location.x - d[0], o.location.x + d[0]]
        ys += [o.location.y - d[1], o.location.y + d[1]]
        zs += [o.location.z - d[2], o.location.z + d[2]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    extent = max(max(xs) - min(xs), max(ys) - min(ys), 4.0) + 2.0
    floor_z = min(zs) - 0.05  # just under the lowest member geometry

    # --- Step 4b: add a basic Sun light (unless suppressed)
    if not args.no_add_sun:
        sun_data = bpy.data.lights.new(name="Island_Sun", type="SUN")
        sun_data.energy = 3.0
        sun = bpy.data.objects.new("Island_Sun", sun_data)
        sun.location = (extent * 0.5, -extent * 0.5, max(zs) + extent * 0.6)
        # tilt so it points roughly toward the group center
        sun.rotation_euler = (math.radians(45), 0.0, math.radians(30))
        bpy.context.scene.collection.objects.link(sun)

    # --- Step 4d: set up a camera for the canonical view
    if args.use_scene_camera:
        # Use the named camera from the source .blend — no movement or rotation.
        # This makes the rendered island view match the reference image viewpoint.
        scene_cam = bpy.data.objects.get(args.use_scene_camera)
        if scene_cam is not None and scene_cam.type == "CAMERA":
            bpy.context.scene.camera = scene_cam
            print(f"[island] using scene camera {args.use_scene_camera!r} (pose unchanged)")
        else:
            print(
                f"[island] WARNING: --use-scene-camera {args.use_scene_camera!r} not found "
                f"or not a CAMERA object — falling back to synthetic camera",
                file=sys.stderr,
            )
            args.use_scene_camera = None  # trigger fallback below

    if not args.use_scene_camera:
        # Synthetic 3/4-overhead framing camera (original behavior).
        cam = bpy.data.objects.get("Camera")
        if cam is None:
            cam_data = bpy.data.cameras.new("__island_cam")
            cam = bpy.data.objects.new("__island_cam", cam_data)
            bpy.context.scene.collection.objects.link(cam)
        cam.location = (cx + extent * 0.6, cy - extent * 0.6, max(zs) + extent * 0.4)
        cam.rotation_euler = (math.radians(60), 0.0, math.radians(45))
        bpy.context.scene.camera = cam

    # --- Step 5: save island.blend
    out_blend = Path(args.output_blend).resolve()
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"[island] saved {out_blend}")

    # --- Step 6: write metadata. M_anchor is the inverse-of-our-inverse, i.e.
    # the single transform that re-projects canonical -> world for every member.
    metadata = {
        "group_name":   args.group_name,
        "anchor_id":    anchor.name,
        "convention":   "single_rigid_xform",
        "transform_note": (
            "world = M_anchor @ canonical  (uniform per island).  "
            "M_anchor = Translate(anchor_world_origin) @ Euler(anchor_world_rotation_euler).  "
            "Scale is NOT part of the rigid xform; each object's scale is preserved as-is."
        ),
        "anchor_world_origin":         list(anchor_loc),
        "anchor_world_rotation_euler": list(anchor_rot),
        "M_anchor_4x4":                [list(row) for row in M_anchor],
        "M_inv_4x4":                   [list(row) for row in M_inv],
        "members":      members_meta,
        "island_blend": str(out_blend),
    }
    meta_path = Path(args.metadata).resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[island] metadata -> {meta_path}")


main()
