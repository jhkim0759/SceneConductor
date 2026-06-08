"""Inner Blender script for merge_island.py — run by Blender headless.

Usage:
    blender -b <scene_blend> --python merge_island_blender_inner.py -- <selected_groups_json>

For each group in selected_groups.json:
  - Opens group_dir/island.blend as a library.
  - For each non-anchor member obj_id:
      canonical_matrix = island.blend's obj_id.matrix_world
      world_matrix = M_anchor @ canonical_matrix
      Decompose world_matrix -> location, rotation_euler (XYZ), scale.
      Write onto bpy.data.objects[obj_id] in the main scene.
  - Anchor object: untouched.
Saves the scene blend in place.
Never touches blend/blender_scene.blend.

Prints per-group summaries. Logs warnings for missing objects; does not abort.

Name-conflict note: when loading island.blend into a scene that already has objects
with the same names (e.g., obj_9 exists in both), Blender appends .001 suffixes.
We resolve this by snapshotting data_from.objects (original names) and data_to.objects
(loaded references, possibly renamed) in parallel and building a name->matrix dict
using the original names as keys.

matrix_world note: loaded objects that are not yet linked to a scene collection
report identity as matrix_world.  We link them temporarily to the active collection,
call view_layer.update(), then read matrix_world before unlinking.
"""
import json
import sys
import traceback
from pathlib import Path

import bpy
import mathutils


def parse_mat4(rows: list) -> mathutils.Matrix:
    return mathutils.Matrix([
        [rows[0][0], rows[0][1], rows[0][2], rows[0][3]],
        [rows[1][0], rows[1][1], rows[1][2], rows[1][3]],
        [rows[2][0], rows[2][1], rows[2][2], rows[2][3]],
        [rows[3][0], rows[3][1], rows[3][2], rows[3][3]],
    ])


def get_island_transforms(island_path: str) -> dict[str, mathutils.Matrix]:
    """Load island.blend as a library; return {original_obj_name: matrix_world}.

    Blender renames loaded objects if the scene already has objects with the same
    name (obj_9 -> obj_9.001).  We capture the original names from data_from before
    the load and pair them positionally with the loaded references in data_to.

    Loaded objects are NOT linked to any collection by default, so matrix_world
    reports as identity until they are linked.  We link each one temporarily,
    call view_layer.update(), capture the matrix, then unlink and remove.
    """
    transforms: dict[str, mathutils.Matrix] = {}
    loaded_objects: list = []

    try:
        with bpy.data.libraries.load(island_path, link=False) as (data_from, data_to):
            original_names: list[str] = list(data_from.objects)
            data_to.objects = list(data_from.objects)

        # Build (original_name, loaded_obj) pairs
        pairs = list(zip(original_names, data_to.objects))
        active_col = bpy.context.collection  # active collection in the scene

        for orig_name, loaded_obj in pairs:
            if loaded_obj is None:
                continue
            loaded_objects.append(loaded_obj)

            # Link to the active collection so matrix_world is computed
            if loaded_obj.name not in [o.name for o in active_col.objects]:
                active_col.objects.link(loaded_obj)

        # One update pass for all newly linked objects
        bpy.context.view_layer.update()

        # Now read matrices
        for orig_name, loaded_obj in pairs:
            if loaded_obj is None:
                continue
            transforms[orig_name] = loaded_obj.matrix_world.copy()

    except Exception as exc:
        print(f"[merge_island] WARNING: failed to load {island_path}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # Clean up: remove all loaded objects
        for ob in loaded_objects:
            if ob is None:
                continue
            ob_name = ob.name
            real_ob = bpy.data.objects.get(ob_name)
            if real_ob is None:
                continue
            # Unlink from all collections
            for col in bpy.data.collections:
                if real_ob.name in [o.name for o in col.objects]:
                    col.objects.unlink(real_ob)
            for sc in bpy.data.scenes:
                if real_ob.name in [o.name for o in sc.collection.objects]:
                    sc.collection.objects.unlink(real_ob)
            bpy.data.objects.remove(real_ob, do_unlink=True)

    return transforms


def main():
    argv = sys.argv
    try:
        sep = argv.index("--")
        selected_groups_json = argv[sep + 1]
    except (ValueError, IndexError):
        print("[merge_island] ERROR: pass <selected_groups.json> after '--'", file=sys.stderr)
        sys.exit(1)

    data = json.loads(Path(selected_groups_json).read_text())
    groups = data.get("groups", [])

    if not groups:
        print("[merge_island] no groups — nothing to merge")
        bpy.ops.wm.save_mainfile()
        sys.exit(0)

    scene_blend_path = bpy.data.filepath
    print(f"[merge_island] working blend: {scene_blend_path}")

    for group in groups:
        group_id = group["group_id"]
        group_dir = Path(group["group_dir"])
        anchor_id = group["anchor_id"]
        # Prefer the explicit chosen refined island.blend from the manifest
        # (e.g. simple_refiner/iter_K/island.blend). Fall back to the group_dir
        # baseline only when the caller did not supply one — that baseline is
        # the un-refined snapshot and using it would silently discard all
        # island-refiner iterations.
        explicit_island = group.get("island_blend") or group.get("island")
        if explicit_island:
            island_path = str(explicit_island)
        else:
            island_path = str(group_dir / "island.blend")
        meta_path = group_dir / "metadata.json"

        # Read metadata for M_anchor and member info
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f"[merge_island] WARNING: cannot read {meta_path}: {exc} — skipping group {group_id}")
            continue

        m_anchor_rows = meta.get("M_anchor_4x4")
        if m_anchor_rows is None:
            print(f"[merge_island] WARNING: M_anchor_4x4 missing in {meta_path} — skipping {group_id}")
            continue

        M_anchor = parse_mat4(m_anchor_rows)
        members_meta = meta.get("members", {})

        # Load canonical transforms from island.blend (keyed by original island names)
        island_transforms = get_island_transforms(island_path)
        if not island_transforms:
            print(f"[merge_island] WARNING: no transforms loaded from {island_path} — skipping {group_id}")
            continue

        updated_count = 0
        for obj_id, member_info in members_meta.items():
            is_anchor = member_info.get("is_anchor", False)
            if is_anchor:
                continue  # Anchor is authoritative in the scene; never touch it

            # Get canonical matrix from island.blend (keyed by original name)
            M_canonical = island_transforms.get(obj_id)
            if M_canonical is None:
                print(f"[merge_island] WARNING: {obj_id} not found in island.blend — skipping")
                continue

            # Get scene object (the existing one in the scene, keyed by original name)
            scene_obj = bpy.data.objects.get(obj_id)
            if scene_obj is None:
                print(f"[merge_island] WARNING: {obj_id} not found in scene blend — skipping")
                continue

            # Compose world transform: M_world = M_anchor @ M_canonical
            M_world = M_anchor @ M_canonical

            # Decompose into loc / rot / scale
            loc, rot_quat, scale = M_world.decompose()
            rot_euler = rot_quat.to_euler("XYZ")

            scene_obj.location = loc
            if scene_obj.rotation_mode != "XYZ":
                print(f"[merge_island_blender_inner] warning: forcing {scene_obj.name}.rotation_mode XYZ (was {scene_obj.rotation_mode})", file=sys.stderr)
                scene_obj.rotation_mode = "XYZ"
            scene_obj.rotation_euler = rot_euler
            scene_obj.scale = scale

            updated_count += 1

        print(
            f"[merge_island] merged {group_id}: {updated_count} members updated "
            f"(anchor={anchor_id} untouched)"
        )

    # Save the scene blend in place
    bpy.ops.wm.save_mainfile()
    print(f"[merge_island] saved {scene_blend_path}")
    sys.exit(0)


main()
