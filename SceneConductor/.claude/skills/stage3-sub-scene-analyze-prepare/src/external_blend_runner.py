"""Blender runner that applies one modification op to a .blend file.

Invoke via:
    blender --background <input.blend> --python external_blend_runner.py -- <op_json> <output.blend>

`op_json` is a JSON file with an "action" key plus action-specific args.
Result is written to `<op_json>.result` so the caller can read it.
"""
import json
import sys
import traceback

import bpy


def _find_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- separator before args")
    after = argv[argv.index("--") + 1:]
    if len(after) < 2:
        raise SystemExit("expected: <op_json> <output_blend>")
    return after[0], after[1]


def _serialize_object(obj):
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "children": [c.name for c in obj.children],
        "location": list(obj.location),
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "dimensions": list(obj.dimensions),
    }


def _apply_action(op):
    action = op["action"]

    if action == "list_objects":
        prefix = op.get("name_prefix", "obj_")
        objs = [_serialize_object(o) for o in bpy.data.objects if o.name.startswith(prefix)]
        objs.sort(key=lambda d: d["name"])
        return {"action": action, "success": True, "objects": objs}

    if action == "inspect_object":
        name = op["obj_name"]
        o = bpy.data.objects.get(name)
        if o is None:
            return {"action": action, "success": False, "message": f"not found: {name}"}
        return {"action": action, "success": True, "object": _serialize_object(o)}

    if action == "update_layout":
        name = op["obj_name"]
        loc = op["location"]
        o = bpy.data.objects.get(name)
        if o is None:
            return {"action": action, "success": False, "message": f"not found: {name}"}
        before = list(o.location)
        o.location = loc
        bpy.context.view_layer.update()
        return {"action": action, "success": True, "obj_name": name, "before": before, "after": list(o.location)}

    if action == "update_rotation":
        name = op["obj_name"]
        rot = op["rotation_euler"]
        o = bpy.data.objects.get(name)
        if o is None:
            return {"action": action, "success": False, "message": f"not found: {name}"}
        before = list(o.rotation_euler)
        o.rotation_mode = "XYZ"
        o.rotation_euler = rot
        bpy.context.view_layer.update()
        return {"action": action, "success": True, "obj_name": name, "before": before, "after": list(o.rotation_euler)}

    if action == "update_size":
        name = op["obj_name"]
        scl = op["scale"]
        o = bpy.data.objects.get(name)
        if o is None:
            return {"action": action, "success": False, "message": f"not found: {name}"}
        before = list(o.scale)
        o.scale = scl
        bpy.context.view_layer.update()
        return {"action": action, "success": True, "obj_name": name, "before": before, "after": list(o.scale)}

    if action == "render":
        out_png = op["output_png"]
        view = op.get("view", "top")  # "top" or "front" or "persp"
        res = op.get("resolution", [800, 600])
        # Patterns of objects to hide during render. Stage walls/ceiling/lights
        # block external cameras — hide by default for scenes that have them.
        hide_patterns = op.get(
            "hide_patterns",
            ["Wall_", "Ceiling", "Portal_Window_", "Area_Fill", "Practical_", "Class_Light_"],
        )
        scene = bpy.context.scene
        scene.render.image_settings.file_format = "PNG"
        scene.render.resolution_x = res[0]
        scene.render.resolution_y = res[1]
        scene.render.filepath = out_png
        scene.render.film_transparent = False
        scene.render.engine = "BLENDER_WORKBENCH"
        try:
            scene.display.shading.light = "STUDIO"
            scene.display.shading.color_type = "OBJECT"
            scene.display.shading.show_xray = False
            scene.display.shading.studio_light = "Default"
            scene.display.render_aa = "FXAA"
        except Exception:
            pass

        # Hide stage geometry, remember prior visibility to restore later
        hidden_state = []
        for o in bpy.data.objects:
            if any(o.name.startswith(p) for p in hide_patterns):
                hidden_state.append((o, o.hide_render))
                o.hide_render = True

        cam = bpy.data.objects.get("__render_cam")
        if cam is None:
            cam_data = bpy.data.cameras.new("__render_cam")
            cam = bpy.data.objects.new("__render_cam", cam_data)
            scene.collection.objects.link(cam)
        cam.data.type = "ORTHO"
        # Compute scene bbox from obj_* parents (location is on the empty)
        xs, ys, zs = [], [], []
        for o in bpy.data.objects:
            if o.name.startswith("obj_") and not o.hide_render:
                xs.append(o.location.x); ys.append(o.location.y); zs.append(o.location.z)
        if not xs:
            for o in bpy.data.objects:
                if o.type == "MESH" and not o.hide_render:
                    xs.append(o.location.x); ys.append(o.location.y); zs.append(o.location.z)
        if not xs:
            xs = ys = zs = [0.0]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        extent_x = max(xs) - min(xs)
        extent_y = max(ys) - min(ys)
        extent = max(extent_x, extent_y, 4.0) + 3.0
        cam.data.ortho_scale = extent
        cam.data.clip_start = 0.01
        cam.data.clip_end = 200.0
        if view == "top":
            cam.location = (cx, cy, 50.0)
            cam.rotation_euler = (0.0, 0.0, 0.0)
        elif view == "front":
            cam.location = (cx, min(ys) - 15.0, max(zs) + 2.0)
            cam.rotation_euler = (1.5708, 0.0, 0.0)
        else:
            cam.location = (cx + extent, cy - extent, extent / 2)
            cam.rotation_euler = (1.0, 0.0, 0.785)
        scene.camera = cam
        bpy.ops.render.render(write_still=True)

        # Restore visibility
        for o, prev in hidden_state:
            o.hide_render = prev

        return {
            "action": action,
            "success": True,
            "output_png": out_png,
            "view": view,
            "hidden_count": len(hidden_state),
        }

    if action == "metrics":
        # Geometric evaluation: count out-of-bounds + pairwise collisions for obj_*
        prefix = op.get("name_prefix", "obj_")
        room_bbox = op.get("room_bbox")  # [[xmin, ymin, zmin], [xmax, ymax, zmax]] or None
        oob_tolerance = op.get("oob_tolerance", 0.05)
        collision_tolerance = op.get("collision_tolerance", 0.001)

        targets = [o for o in bpy.data.objects if o.name.startswith(prefix)]

        # Auto-derive room bbox from Floor + Wall_* if not provided
        if room_bbox is None:
            xs, ys, zs = [], [], []
            for o in bpy.data.objects:
                if o.name == "Floor" or o.name.startswith("Wall_") or o.name == "Ceiling":
                    for corner in o.bound_box:
                        wc = o.matrix_world @ __import__("mathutils").Vector(corner)
                        xs.append(wc.x); ys.append(wc.y); zs.append(wc.z)
            if xs:
                room_bbox = [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]
            else:
                room_bbox = [[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]

        # Compute world-space bbox for each target (use mesh-children bound_box if obj is empty)
        def world_bbox(obj):
            xs, ys, zs = [], [], []
            mesh_descendants = []
            stack = [obj]
            while stack:
                cur = stack.pop()
                if cur.type == "MESH":
                    mesh_descendants.append(cur)
                stack.extend(cur.children)
            if not mesh_descendants:
                p = obj.location
                return [(p.x, p.y, p.z), (p.x, p.y, p.z)]
            for m in mesh_descendants:
                for corner in m.bound_box:
                    wc = m.matrix_world @ __import__("mathutils").Vector(corner)
                    xs.append(wc.x); ys.append(wc.y); zs.append(wc.z)
            return [(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))]

        # OOB check
        oob_objects = []
        for o in targets:
            (xmn, ymn, zmn), (xmx, ymx, zmx) = world_bbox(o)
            r_lo, r_hi = room_bbox
            if (xmn < r_lo[0] - oob_tolerance or ymn < r_lo[1] - oob_tolerance or
                zmn < r_lo[2] - oob_tolerance or xmx > r_hi[0] + oob_tolerance or
                ymx > r_hi[1] + oob_tolerance or zmx > r_hi[2] + oob_tolerance):
                oob_objects.append(o.name)

        # Pairwise AABB overlap (volume-based) — fast geometric BBL check
        bboxes = {o.name: world_bbox(o) for o in targets}

        def overlap_volume(a, b):
            (axn, ayn, azn), (axx, ayx, azx) = a
            (bxn, byn, bzn), (bxx, byx, bzx) = b
            dx = max(0.0, min(axx, bxx) - max(axn, bxn))
            dy = max(0.0, min(ayx, byx) - max(ayn, byn))
            dz = max(0.0, min(azx, bzx) - max(azn, bzn))
            return dx * dy * dz

        names = list(bboxes.keys())
        collisions = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                v = overlap_volume(bboxes[names[i]], bboxes[names[j]])
                if v > collision_tolerance:
                    collisions.append({"a": names[i], "b": names[j], "volume_m3": round(v, 5)})

        return {
            "action": action,
            "success": True,
            "Nobj": len(targets),
            "OOB_count": len(oob_objects),
            "OOB_objects": oob_objects,
            "BBL_count": len(collisions),
            "collisions": sorted(collisions, key=lambda d: -d["volume_m3"])[:20],
            "room_bbox": room_bbox,
        }

    if action == "remove_object":
        name = op["obj_name"]
        o = bpy.data.objects.get(name)
        if o is None:
            return {"action": action, "success": False, "message": f"not found: {name}"}
        # Recursively collect descendants so we delete the whole subtree
        to_delete = []
        stack = [o]
        while stack:
            cur = stack.pop()
            to_delete.append(cur)
            stack.extend(cur.children)
        names = [x.name for x in to_delete]
        for x in to_delete:
            bpy.data.objects.remove(x, do_unlink=True)
        return {"action": action, "success": True, "obj_name": name, "removed": names}

    return {"action": action, "success": False, "message": f"unknown action: {action}"}


def main():
    op_json, output_blend = _find_args()
    with open(op_json) as f:
        op = json.load(f)

    try:
        result = _apply_action(op)
    except Exception:
        result = {"action": op.get("action"), "success": False, "message": traceback.format_exc()}

    if result.get("success") and op["action"] not in ("list_objects", "inspect_object", "render"):
        try:
            bpy.ops.wm.save_as_mainfile(filepath=output_blend)
            result["output"] = output_blend
        except Exception:
            result["success"] = False
            result["message"] = "save failed: " + traceback.format_exc()

    with open(op_json + ".result", "w") as f:
        json.dump(result, f, indent=2)


main()
