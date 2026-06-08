"""Blender runner that applies one modification op to a .blend file.

Invoke via:
    blender --background <input.blend> --python external_blend_runner.py -- <op_json> <output.blend>

`op_json` is a JSON file with an "action" key plus action-specific args.
Result is written to `<op_json>.result` so the caller can read it.
"""
import json
import os
import sys
import traceback

import bpy
import numpy as np

UP_AXIS = 2  # Blender Z-up


def _find_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- separator before args")
    after = argv[argv.index("--") + 1:]
    if len(after) < 2:
        raise SystemExit("expected: <op_json> <output_blend>")
    return after[0], after[1]


def _local_scaled_dims(obj):
    """Return [dx, dy, dz] = bbox extents in obj's LOCAL frame multiplied by obj.scale.

    Invariant to the object's current rotation/translation — gives the slab's
    PHYSICAL (long_axis, depth, height) regardless of how it's oriented in world.
    For thin slabs (posters/chalkboards) min(dims) ≈ true depth. Critical for
    wall placement where world-frame bbox would be inflated by yaw rotation.
    """
    import numpy as _np
    from mathutils import Vector as _Vec
    inv_root = obj.matrix_world.inverted()
    pts = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if o.type == "MESH" and o.data and len(o.data.vertices) > 0:
            mw = o.matrix_world
            for v in o.data.vertices:
                pts.append((inv_root @ (mw @ v.co))[:])
        stack.extend(o.children)
    if not pts:
        return None
    A = _np.array(pts, dtype=_np.float64)
    unit_dims = A.max(0) - A.min(0)
    sx, sy, sz = obj.scale
    return unit_dims * _np.array([abs(sx), abs(sy), abs(sz)])


def _gather_world_triangles(obj):
    """Return (V, T): world-space verts (N,3) and triangle indices (Tn,3) over obj + descendants."""
    all_V, all_T = [], []
    offset = 0
    stack = [obj]
    while stack:
        o = stack.pop()
        if o.type == "MESH" and o.data and len(o.data.vertices) > 0:
            mw = o.matrix_world
            verts = np.array([(mw @ v.co)[:] for v in o.data.vertices], dtype=np.float64)
            tris = []
            for poly in o.data.polygons:
                vs = list(poly.vertices)
                for i in range(1, len(vs) - 1):
                    tris.append([offset + vs[0], offset + vs[i], offset + vs[i + 1]])
            if tris:
                all_V.append(verts)
                all_T.append(np.array(tris, dtype=np.int64))
                offset += len(verts)
        stack.extend(o.children)
    if not all_V:
        return None, None
    return np.concatenate(all_V, axis=0), np.concatenate(all_T, axis=0)


def _sample_points(V, T, n, rng):
    """Area-weighted barycentric sampling on the triangle mesh — matches trimesh.sample()."""
    if T is None or len(T) == 0:
        return V[:1].repeat(n, axis=0)
    tri_v = V[T]
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total = areas.sum()
    if total < 1e-12:
        return np.tile(V.mean(axis=0, keepdims=True), (n, 1))
    probs = areas / total
    idx = rng.choice(len(T), size=n, p=probs)
    u = rng.random(n)
    v = rng.random(n)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v
    return (w[:, None] * tri_v[idx, 0]
            + u[:, None] * tri_v[idx, 1]
            + v[:, None] * tri_v[idx, 2])


def _min_pair_diff(A, B, chunk=256):
    """Chunked chamfer-style: find min ||A_i - B_j|| and the diff vector (B_j - A_i)."""
    best_d, best_diff = np.inf, np.zeros(3)
    for i in range(0, A.shape[0], chunk):
        Ac = A[i:i + chunk]
        diff = B[None, :, :] - Ac[:, None, :]      # (c, M, 3)  vector from A to B
        d = np.linalg.norm(diff, axis=-1)          # (c, M)
        ij = np.unravel_index(d.argmin(), d.shape)
        if d[ij] < best_d:
            best_d = float(d[ij])
            best_diff = diff[ij].copy()
    return best_d, best_diff


def _move_closer_chamfer(anchor_pts, moving_pts):
    """Snap moving so its closest point coincides with anchor's closest point."""
    _, diff = _min_pair_diff(anchor_pts, moving_pts)   # diff = moving_closest - anchor_closest
    return -diff                                       # translate moving by -diff to coincide


def _place_on_top(anchor_pts, moving_pts):
    """Align moving's bottom (min Z) with anchor's top (max Z)."""
    delta = np.zeros(3)
    delta[UP_AXIS] = anchor_pts[:, UP_AXIS].max() - moving_pts[:, UP_AXIS].min()
    return delta


def _project_kissing(anchor_pts, moving_pts, direction):
    """Translate moving along `direction` so the two surfaces just touch.

    direction points from anchor toward moving (or any chosen axis).
    """
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return np.zeros(3)
    d = d / n
    proj_a = anchor_pts @ d
    proj_m = moving_pts @ d
    # moving's nearest-to-anchor face along d == anchor's furthest face along d
    return d * float(proj_a.max() - proj_m.min())


_AXIS_DIRS = {
    "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "+y": (0, 1, 0), "-y": (0, -1, 0),
    "+z": (0, 0, 1), "-z": (0, 0, -1),
    "x":  (-1, 0, 0),  # legacy alias (move in -X)
    "z":  (0, 0, 1),   # legacy alias
    "above": (0, 0, 1), "below": (0, 0, -1),
}


def _resolve_direction(rel):
    if isinstance(rel, (list, tuple)):
        return np.array(rel, dtype=np.float64)
    if rel in _AXIS_DIRS:
        return np.array(_AXIS_DIRS[rel], dtype=np.float64)
    raise ValueError(f"Unknown relation/direction: {rel!r}")


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

    if action == "flip_yaw_180":
        import math
        name = op["obj_name"]
        o = bpy.data.objects.get(name)
        if o is None:
            return {"action": action, "success": False, "message": f"not found: {name}"}
        o.rotation_mode = "XYZ"
        before = list(o.rotation_euler)
        new_rz = math.atan2(
            math.sin(before[2] + math.pi),
            math.cos(before[2] + math.pi),
        )
        o.rotation_euler = (before[0], before[1], new_rz)
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

    if action == "attach_to_wall":
        wall_name = op["wall_obj"]
        moving_name = op["moving_obj"]
        polygon_path = op.get("polygon_path")
        clearance = float(op.get("clearance", 0.02))
        forced_t = op.get("t_along_m")
        preserve_rotation = bool(op.get("preserve_rotation", False))

        moving = bpy.data.objects.get(moving_name)
        if moving is None:
            return {"action": action, "success": False, "message": f"object not found: {moving_name}"}

        if not polygon_path:
            blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else None
            if blend_dir:
                cand = os.path.join(os.path.dirname(blend_dir), "json", "polygon_v2.json")
                if os.path.isfile(cand):
                    polygon_path = cand

        # Axis-aligned fallback normals for Wall_01..Wall_04 when polygon_v2.json is unavailable.
        # Used when polygon_path is missing or the wall is not listed in the polygon data.
        # Wall_01: inward normal +X, Wall_02: +Y, Wall_03: -X, Wall_04: -Y.
        _WALL_FALLBACK = {
            "Wall_01": (1.0, 0.0, 0.0, 1.0, 0.0),   # (nx, ny, tx, ty, tangent_angle)
            "Wall_02": (0.0, 1.0, 0.0, 1.0, 1.5707963267948966),
            "Wall_03": (-1.0, 0.0, 0.0, -1.0, 3.141592653589793),
            "Wall_04": (0.0, -1.0, 0.0, -1.0, -1.5707963267948966),
        }

        use_fallback = False
        if not polygon_path or not os.path.isfile(polygon_path):
            if wall_name in _WALL_FALLBACK:
                use_fallback = True
            else:
                return {"action": action, "success": False,
                        "message": f"polygon_path not provided and auto-discovery failed (blend={bpy.data.filepath!r})"}

        if use_fallback:
            nx, ny, tx, ty, tangent_angle = _WALL_FALLBACK[wall_name]
            wall_thickness = 0.25
            # Determine v_from from the wall object's bounding box in Blender
            wall_obj_bl = bpy.data.objects.get(wall_name)
            if wall_obj_bl is not None:
                import mathutils as _mu
                corners = [wall_obj_bl.matrix_world @ _mu.Vector(c) for c in wall_obj_bl.bound_box]
                xs = [c.x for c in corners]
                ys = [c.y for c in corners]
                # wall surface coordinate: for normal +Y wall, the surface is at min(y)
                if nx == 1.0:   # Wall_01: surface at max(x) of wall mesh (inner face)
                    wall_surface_coord = max(xs)
                elif nx == -1.0:  # Wall_03: surface at min(x) of wall mesh
                    wall_surface_coord = min(xs)
                elif ny == 1.0:   # Wall_02: surface at max(y) of wall mesh
                    wall_surface_coord = max(ys)
                else:             # Wall_04: surface at min(y) of wall mesh
                    wall_surface_coord = min(ys)
            else:
                wall_surface_coord = 0.0
            # Build synthetic v_from so that signed-distance computation works
            if nx != 0.0:
                v_from = [wall_surface_coord, 0.0]
            else:
                v_from = [0.0, wall_surface_coord]
            L = 100.0  # effectively infinite along-wall length for fallback
        else:
            with open(polygon_path) as f:
                poly = json.load(f)
            verts = poly["polygon_vertices"]
            edges = poly["wall_edges"]
            wall_thickness = float(poly.get("wall_thickness", 0.25))

            signed_area = 0.0
            for i in range(len(verts)):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % len(verts)]
                signed_area += x1 * y2 - x2 * y1
            ccw = signed_area > 0

            # --- Wall re-selection for AMBIGUOUS assignments ---
            # The object's CURRENT (GALP) rotation already encodes which wall it
            # belongs to: its broad face is parallel to that wall. When the wall
            # assignment is ambiguous, attach to the wall the rotation matches
            # (the one the object is flattest against) so the rotation-aligned wall
            # and the attach wall are the SAME. Otherwise a frame can be rotated to
            # one wall's axis but pushed onto a different (nearer) wall, ending up
            # edge-on. Picks the candidate with the smallest perpendicular extent.
            if op.get("wall_ambiguous"):
                # Candidate walls: explicit list if the op carries one, else every
                # wall (the planner-revised plan drops wall_candidates, so we must
                # not depend on it).
                cand_walls = op.get("wall_candidates") or [e["object"] for e in edges]
                Vsel, _ = _gather_world_triangles(moving)
                if Vsel is not None and len(Vsel) > 0:
                    Vxy = Vsel[:, :2]
                    cen = Vxy.mean(axis=0)
                    rel_sel = Vxy - cen
                    best_w = None
                    best_ext = None
                    for cand in cand_walls:
                        ce = next((e for e in edges if e["object"] == cand), None)
                        if ce is None:
                            continue
                        cvf = verts[ce["from"]]
                        cvt = verts[ce["to"]]
                        ctx, cty = cvt[0] - cvf[0], cvt[1] - cvf[1]
                        cL = float(np.hypot(ctx, cty))
                        if cL < 1e-8:
                            continue
                        ctx, cty = ctx / cL, cty / cL
                        cnx, cny = (-cty, ctx) if ccw else (cty, -ctx)
                        # Distance gate: only walls the object actually sits in front
                        # of and near (closest vertex within ~2 m of the inner face,
                        # not behind it) — excludes the parallel opposite wall.
                        d_perp = (Vxy - np.array(cvf)) @ np.array([cnx, cny])
                        closest = float(d_perp.min())
                        if closest < -0.5 or closest > 2.0:
                            continue
                        # Among near walls, the rotation-matched wall is the one the
                        # object's broad face is parallel to → smallest perp extent.
                        proj = rel_sel[:, 0] * cnx + rel_sel[:, 1] * cny
                        ext = float(proj.max() - proj.min())
                        if best_ext is None or ext < best_ext:
                            best_ext = ext
                            best_w = cand
                    if best_w is not None and best_w != wall_name:
                        print(f"[attach_to_wall] ambiguous wall re-selected by rotation: "
                              f"{wall_name} -> {best_w} (perp_extent={best_ext:.3f}m)")
                        wall_name = best_w
                        # t_along_m was computed for the original wall; drop it so the
                        # along-wall position is preserved from the object's current
                        # location on the newly chosen wall (pure perpendicular push).
                        forced_t = None

            wall_edge = next((e for e in edges if e["object"] == wall_name), None)
            if wall_edge is None:
                if wall_name in _WALL_FALLBACK:
                    use_fallback = True
                    nx, ny, tx, ty, tangent_angle = _WALL_FALLBACK[wall_name]
                    wall_obj_bl = bpy.data.objects.get(wall_name)
                    if wall_obj_bl is not None:
                        import mathutils as _mu
                        corners = [wall_obj_bl.matrix_world @ _mu.Vector(c) for c in wall_obj_bl.bound_box]
                        xs = [c.x for c in corners]; ys = [c.y for c in corners]
                        if nx == 1.0:   wall_surface_coord = max(xs)
                        elif nx == -1.0: wall_surface_coord = min(xs)
                        elif ny == 1.0:  wall_surface_coord = max(ys)
                        else:            wall_surface_coord = min(ys)
                    else:
                        wall_surface_coord = 0.0
                    v_from = [wall_surface_coord, 0.0] if nx != 0.0 else [0.0, wall_surface_coord]
                    L = 100.0
                else:
                    return {"action": action, "success": False,
                            "message": f"wall {wall_name} not in {polygon_path}"}
            else:
                v_from = verts[wall_edge["from"]]
                v_to = verts[wall_edge["to"]]
                tx0, ty0 = v_to[0] - v_from[0], v_to[1] - v_from[1]
                L = float(np.hypot(tx0, ty0))
                if L < 1e-8:
                    return {"action": action, "success": False, "message": f"wall {wall_name} has zero length"}
                tx, ty = tx0 / L, ty0 / L
                if ccw:
                    nx, ny = -ty, tx
                else:
                    nx, ny = ty, -tx
                tangent_angle = float(np.arctan2(ty0, tx0))

        before_rot = list(moving.rotation_euler)
        before_loc = list(moving.location)

        # STEP 1 — rotation alignment (rz = wall tangent angle), rx/ry preserved.
        # Skipped when preserve_rotation=True so the object keeps its current orientation.
        if not preserve_rotation:
            # Pick yaw candidate with MINIMUM rotation change from current yaw.
            import math
            current_yaw = before_rot[2]
            candidates = [
                tangent_angle,
                tangent_angle + math.pi / 2,
                tangent_angle + math.pi,
                tangent_angle + 3 * math.pi / 2,
            ]
            # Normalize candidates to [0, 2π).
            candidates = [c % (2 * math.pi) for c in candidates]
            # Circular distance: shortest angle between two orientations.
            def circular_dist(a, b):
                return abs((a - b + math.pi) % (2 * math.pi) - math.pi)
            distances = [circular_dist(c, current_yaw) for c in candidates]
            best_idx = distances.index(min(distances))
            chosen_yaw = candidates[best_idx]
            rotation_delta_deg = math.degrees(distances[best_idx])
            print(f"[attach_to_wall] tangent_angle={math.degrees(tangent_angle):.1f}°, "
                  f"current_yaw={math.degrees(current_yaw):.1f}°, "
                  f"chosen candidate idx={best_idx}, rotation_delta={rotation_delta_deg:.1f}°")

            moving.rotation_mode = "XYZ"
            moving.rotation_euler = (before_rot[0], before_rot[1], chosen_yaw)
            bpy.context.view_layer.update()

        # STEP 2 — gather world-space mesh vertices (after optional rotation).
        Vm, _ = _gather_world_triangles(moving)
        if Vm is None:
            if not preserve_rotation:
                moving.rotation_euler = before_rot
                bpy.context.view_layer.update()
            return {"action": action, "success": False, "message": "no mesh under moving object"}

        # STEP 3 — signed perpendicular distance from wall line for each vert,
        # with positive direction = inward (toward room interior).
        diff_xy = Vm[:, :2] - np.array([v_from[0], v_from[1]])
        d_signed = diff_xy[:, 0] * nx + diff_xy[:, 1] * ny
        d_min = float(d_signed.min())  # most-outward vertex (closest to / past the wall)

        # STEP 4 — translate so the most-outward vertex sits `clearance` inward
        # of the INNER wall surface. polygon_vertices lie on the inner wall face
        # (the wall mesh is built with Solidify offset=-1.0, extruded outward away
        # from the room), so d_signed=0 is the inner face and target_perp=clearance
        # yields a flush mount with only a small z-fight clearance gap (default 2cm).
        # NOTE: this is NOT relative to the wall centerline — do not add wall_thickness/2.
        # Along-wall (t_along) position: use forced_t when given, else preserve current.
        target_perp = clearance
        delta_perp = target_perp - d_min

        if forced_t is not None:
            # Decompose current centroid into perp + along components, then override along.
            dx = before_loc[0] - v_from[0]
            dy = before_loc[1] - v_from[1]
            current_perp_dist = dx * nx + dy * ny
            new_perp_dist = current_perp_dist + delta_perp
            forced_t_clamped = max(0.0, min(L, float(forced_t)))
            target_x = v_from[0] + forced_t_clamped * tx + new_perp_dist * nx
            target_y = v_from[1] + forced_t_clamped * ty + new_perp_dist * ny
        else:
            # Purely perpendicular push; along-wall position preserved from before_loc.
            target_x = before_loc[0] + delta_perp * nx
            target_y = before_loc[1] + delta_perp * ny

        moving.location = (target_x, target_y, before_loc[2])
        bpy.context.view_layer.update()

        diff_after = (target_x - v_from[0], target_y - v_from[1])
        t_along = max(0.0, min(L, diff_after[0] * tx + diff_after[1] * ty))

        return {
            "action": action, "success": True,
            "wall_obj": wall_name, "moving_obj": moving_name,
            "before_location": before_loc, "after_location": list(moving.location),
            "before_rotation": before_rot, "after_rotation": list(moving.rotation_euler),
            "tangent_angle_deg": float(np.degrees(tangent_angle)),
            "wall_thickness_m": wall_thickness,
            "clearance_m": clearance,
            "preserve_rotation": preserve_rotation,
            "forced_t_along_m": forced_t,
            "d_min_m": d_min,
            "delta_perp_m": delta_perp,
            "target_perp_m": target_perp,
            "t_along_m": t_along, "wall_length_m": L,
            "polygon_winding": "fallback" if use_fallback else ("CCW" if ccw else "CW"),
            "polygon_path": polygon_path,
        }

    if action == "attach":
        anchor_name = op["anchor_obj"]
        moving_name = op["moving_obj"]
        relation = op.get("relation", "attached_to")
        n_samples = int(op.get("n_samples", 2000))

        anchor = bpy.data.objects.get(anchor_name)
        moving = bpy.data.objects.get(moving_name)
        if anchor is None or moving is None:
            return {"action": action, "success": False,
                    "message": f"object not found (anchor={anchor_name}, moving={moving_name})"}

        Va, Ta = _gather_world_triangles(anchor)
        Vm, Tm = _gather_world_triangles(moving)
        if Va is None or Vm is None:
            return {"action": action, "success": False, "message": "no mesh geometry under one of the objects"}

        rng = np.random.default_rng(0)
        pa = _sample_points(Va, Ta, n_samples, rng)
        pm = _sample_points(Vm, Tm, n_samples, rng)

        rel = relation if isinstance(relation, (list, tuple)) else str(relation).lower()
        if rel == "on":
            delta = _place_on_top(pa, pm)
        elif rel in ("attached_to", "next_to"):
            delta = _move_closer_chamfer(pa, pm)
        else:
            direction = _resolve_direction(rel)
            delta = _project_kissing(pa, pm, direction)

        before = list(moving.location)
        moving.location = (before[0] + float(delta[0]),
                           before[1] + float(delta[1]),
                           before[2] + float(delta[2]))
        bpy.context.view_layer.update()
        return {
            "action": action, "success": True,
            "anchor_obj": anchor_name, "moving_obj": moving_name,
            "relation": relation,
            "delta": [float(x) for x in delta],
            "before": before, "after": list(moving.location),
            "n_anchor_pts": int(len(pa)), "n_moving_pts": int(len(pm)),
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

    if action == "delete_object":
        obj_name = op["obj_name"]
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            return {"action": action, "success": False, "error": f"object '{obj_name}' not found"}
        # Recursively collect all descendants (leaves first so parent unlinks are safe)
        to_delete = []
        def _collect(o):
            for c in list(o.children):
                _collect(c)
            to_delete.append(o)
        _collect(obj)
        # Track mesh data to clean up orphan meshes after object removal
        mesh_data_to_check = set()
        for o in to_delete:
            if o.type == "MESH" and o.data is not None:
                mesh_data_to_check.add(o.data)
            bpy.data.objects.remove(o, do_unlink=True)
        # Remove now-orphan mesh datablocks
        removed_meshes = 0
        for m in mesh_data_to_check:
            if m.users == 0:
                bpy.data.meshes.remove(m)
                removed_meshes += 1
        return {
            "action": action, "success": True,
            "obj_name": obj_name,
            "deleted_objects": len(to_delete),
            "removed_meshes": removed_meshes,
        }

    return {"action": action, "success": False, "message": f"unknown action: {action}"}


READ_ONLY_ACTIONS = {"list_objects", "inspect_object", "render", "metrics"}


def main():
    op_json, output_blend = _find_args()
    with open(op_json) as f:
        payload = json.load(f)

    is_list = isinstance(payload, list)
    ops = payload if is_list else [payload]

    results = []
    any_mutating_success = False
    for op in ops:
        try:
            r = _apply_action(op)
        except Exception:
            r = {"action": op.get("action"), "success": False, "message": traceback.format_exc()}
        results.append(r)
        if r.get("success") and op.get("action") not in READ_ONLY_ACTIONS:
            any_mutating_success = True
        # Best-effort apply: a single failed op (e.g. "object not found" for an
        # object merged away upstream) leaves Blender state intact, so we keep
        # going instead of abandoning the remaining ops. Without this, the batch
        # was order-dependent — a phantom-object op early in the plan would skip
        # every valid op after it. Per-op failures are still recorded in results.

    save_ok = True
    if any_mutating_success:
        try:
            bpy.ops.wm.save_as_mainfile(filepath=output_blend)
        except Exception:
            save_ok = False
            for r in results:
                r["success"] = False
                r["message"] = "save failed: " + traceback.format_exc()

    if is_list:
        out = {
            "success": save_ok and all(r.get("success") for r in results) and len(results) == len(ops),
            "n_ops": len(ops),
            "n_executed": len(results),
            "results": results,
            "output": output_blend if (any_mutating_success and save_ok) else None,
        }
    else:
        out = results[0]
        if any_mutating_success and save_ok:
            out["output"] = output_blend

    with open(op_json + ".result", "w") as f:
        json.dump(out, f, indent=2)


main()
