"""
graph_tool_planner.py — Emit tool-based ops from relation_graph without LLM.

Reads:
  inputs/relation_graph.json
  json/blend_info.json
  json/polygon_v2.json
  json/object_state.json          (per-object attached_to lists)
  inputs/object_class.json        (label_id -> class_name)

Emits:
  json/graph_ops.json
  json/wall_ambiguous.json        (hook for future LLM tie-break)

Edge-type → action mapping:
  [NEW] per-object wall_set       → attach_to_wall  (replaces mounted_on_same_wall path)
  on_top_of                       → attach          (each member, anchor from group["anchor"])
  seated_around                   → skip            (needs island refinement; not handled here)
  co_illuminates                  → skip
  adjacent_to                     → skip
  custom                          → skip

Wall resolution uses nearest point-to-segment perpendicular distance from the
object's (x,y) to each wall edge in polygon_v2 (not midpoint distance, not
a side-string direction).

Usage:
    python3 graph_tool_planner.py --scene-dir /path/to/scene_dir
    python3 graph_tool_planner.py --scene-dir /path/to/scene_dir --output /custom/graph_ops.json
"""

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Blend-info path resolution (Stage-3 vs Stage-2 fallback)
# ---------------------------------------------------------------------------

def _resolve_blend_info_path(scene_dir: Path) -> Path:
    """Resolve which blend_info.json to load.

    Stage 3 ``json/blend_info.json`` is the canonical source, but a known
    extract_blend_info.py bug occasionally produces a file with
    ``categories.objects == []``. When that happens we fall back to
    ``inputs/blend_info.json`` (the Stage 2 dump, which is the same schema
    but populated). If both exist and the Stage 3 one is populated, prefer
    Stage 3 because it reflects post-Stage-2 edits.
    """
    stage3 = scene_dir / "json" / "blend_info.json"
    stage2 = scene_dir / "inputs" / "blend_info.json"
    if stage3.is_file():
        try:
            import json as _json
            data = _json.loads(stage3.read_text())
            objs = (data.get("categories", {}) or {}).get("objects", [])
            if objs:  # non-empty → trust stage3
                return stage3
        except Exception:
            pass
        # stage3 exists but is broken/empty → fall through to stage2
        print(
            f"WARNING: {stage3} has categories.objects == []. "
            f"Falling back to {stage2} (Stage 2 dump).",
            file=sys.stderr,
        )
    if stage2.is_file():
        return stage2
    # neither exists → return stage3 path so the loader raises a clear error
    return stage3


# ---------------------------------------------------------------------------
# Class taxonomy — copied verbatim from heuristic_planner.py
# (which copied from src/blend_ops/session_runner/scene_analysis.py)
# Copied — NOT imported — for cross-skill decoupling. Keep in sync manually.
# ---------------------------------------------------------------------------

WALL_CLASSES = [
    "chalkboard", "blackboard", "whiteboard", "poster", "picture frame",
    "painting", "mirror", "clock", "board", "tv", "television",
]

# Maximum allowed distance (m) from object origin to its nearest wall.
# Beyond this, treat the object as free-standing — the upstream
# attached_to=wall label (from object_state.json) is presumed mis-classified.
WALL_ATTACH_MAX_DIST_M = 1.5


def _class_matches(class_name: str, taxonomy: list) -> bool:
    """Case-insensitive substring/word match.

    Returns True if any taxonomy keyword appears as a substring of the
    (lower-cased) class name. This lets multi-word class names match a
    single taxonomy keyword: "board poster" matches "board" or "poster";
    "chalkboard bulletin board" matches "chalkboard" or "board".
    """
    if not class_name:
        return False
    cl = class_name.lower()
    for kw in taxonomy:
        if kw in cl:
            return True
    return False


# ---------------------------------------------------------------------------
# Wall frame & spacing computation
# ---------------------------------------------------------------------------

def compute_wall_frame(wall_obj: str, polygon_v2: dict) -> tuple | None:
    """
    Reconstruct the wall frame (v_from, v_to, L, t_unit) for the given wall_obj.

    Returns: (v_from [x,y], v_to [x,y], L, t_unit [x,y]) or None if wall not found.

    The tangent t_unit points from v_from toward v_to; projection of a point p
    onto the wall edge gives t = (p - v_from) · t_unit ∈ [0, L].
    """
    verts = polygon_v2.get("polygon_vertices", [])
    wall_edges = polygon_v2.get("wall_edges", [])

    if not verts or not wall_edges:
        return None

    for edge in wall_edges:
        if edge.get("object") != wall_obj:
            continue

        from_idx = edge.get("from")
        to_idx = edge.get("to")
        if from_idx is None or to_idx is None or from_idx >= len(verts) or to_idx >= len(verts):
            continue

        v_from = verts[from_idx]
        v_to = verts[to_idx]

        # Compute L and t_unit
        dx = v_to[0] - v_from[0]
        dy = v_to[1] - v_from[1]
        L = (dx**2 + dy**2) ** 0.5

        if L < 1e-6:
            return None

        t_unit = [dx / L, dy / L]
        return (v_from, v_to, L, t_unit)

    return None


def get_object_width(obj_name: str, blend_info: dict, all_objects: dict | None = None) -> float | None:
    """
    Compute the along-wall half-width of an object by recursively examining child meshes.

    Returns: 0.5 * max(child_dim_x, child_dim_y) over all mesh descendants, or None.

    The empty object itself has dimensions [0,0,0]; we recursively search its
    descendants for MESH objects with non-zero dimensions.
    """
    if not blend_info:
        return None

    # Build object index once if not provided
    if all_objects is None:
        categories = blend_info.get("categories", {})
        all_objects = {}
        for category in ["objects", "world", "geometry_meshes"]:
            for obj in categories.get(category, []):
                all_objects[obj.get("name")] = obj

    obj_info = all_objects.get(obj_name)
    if not obj_info:
        return None

    # Recursively collect max extent from descendants
    def collect_max_extent(node_name: str) -> float:
        node = all_objects.get(node_name)
        if not node:
            return 0.0

        max_e = 0.0

        # Check if this node is a MESH with non-zero dimensions
        if node.get("type") == "MESH":
            dims = node.get("dimensions", [0, 0, 0])
            if len(dims) >= 2:
                extent = max(dims[0], dims[1])
                max_e = max(max_e, extent)

        # Recurse into children
        for child_name in node.get("children", []):
            max_e = max(max_e, collect_max_extent(child_name))

        return max_e

    max_extent = collect_max_extent(obj_name)

    if max_extent < 1e-6:
        return None

    return 0.5 * max_extent


def compute_spaced_t_along_m(
    members: list,
    wall_obj: str,
    wall_frame: tuple,
    blend_info: dict,
) -> dict:
    """
    Compute non-overlapping t_along_m positions for wall members.

    Args:
        members: list of object names
        wall_obj: target wall object name
        wall_frame: (v_from, v_to, L, t_unit) from compute_wall_frame
        blend_info: blend info dict

    Returns: {obj_name: t_along_m} for successfully positioned objects.

    Strategy:
    1. Compute t_i = (loc_i - v_from) · t_unit for each member.
    2. Compute half-width w_i for each member from child dimensions.
    3. Sort by t_i, then apply forward greedy spacing with 0.05 m margin.
    4. Clamp final t values to [0, L], shifting left if overflow.
    """
    v_from, v_to, L, t_unit = wall_frame

    categories = blend_info.get("categories", {})
    all_objects = {}
    for category in ["objects", "world", "geometry_meshes"]:
        for obj in categories.get(category, []):
            all_objects[obj.get("name")] = obj

    # Collect member positions and widths
    member_data = []
    for obj in members:
        obj_info = all_objects.get(obj)
        if not obj_info:
            return {}

        loc = obj_info.get("location", [0, 0, 0])
        if len(loc) < 2:
            return {}

        # Project onto wall
        dx_from = loc[0] - v_from[0]
        dy_from = loc[1] - v_from[1]
        t_i = dx_from * t_unit[0] + dy_from * t_unit[1]

        # Get width (pass all_objects to avoid re-indexing)
        w_i = get_object_width(obj, blend_info, all_objects)
        if w_i is None:
            w_i = 0.15  # fallback

        member_data.append({
            "obj": obj,
            "t_orig": t_i,
            "w": w_i,
        })

    if not member_data:
        return {}

    # Sort by original t_i
    member_data.sort(key=lambda x: x["t_orig"])

    # Forward greedy spacing with 0.05 m margin
    margin = 0.05
    assigned = []
    for i, md in enumerate(member_data):
        if i == 0:
            # First member stays roughly at original position
            t_new = md["t_orig"]
        else:
            # Enforce minimum spacing from previous
            prev_t = assigned[-1]["t_assigned"]
            prev_w = assigned[-1]["w"]
            t_new = max(md["t_orig"], prev_t + prev_w + md["w"] + margin)

        assigned.append({
            "obj": md["obj"],
            "w": md["w"],
            "t_assigned": t_new,
        })

    # Check for overflow and shift if necessary
    max_t = max(a["t_assigned"] for a in assigned)
    if max_t > L:
        overflow = max_t - L
        for a in assigned:
            a["t_assigned"] = max(0.0, a["t_assigned"] - overflow)

    # Clamp to [0, L]
    result = {}
    for a in assigned:
        t_final = max(0.0, min(L, a["t_assigned"]))
        result[a["obj"]] = round(t_final, 4)

    return result


# ---------------------------------------------------------------------------
# Defense-in-depth: refresh stale wall_mount_evidence from bev_objects.json
# ---------------------------------------------------------------------------

# Same threshold used in stage2 compute_polygon.classify_edges.
_WM_NEAR_THRESH = 0.45  # metres


def _refresh_wall_mount_evidence(
    polygon_v2: dict,
    object_state: dict | None,
    scene_dir: Path,
) -> None:
    """Recompute wall_mount_evidence in-memory using the same 2-criteria gate
    as the patched compute_polygon.classify_edges.

    This is a defense-in-depth guard for scenes whose polygon_v2.json was
    produced BEFORE the evidence-gating patch landed in Stage 2.  The fix
    runs entirely in Stage 3 — it never writes polygon_v2.json to disk.

    Gate (mirrors compute_polygon.WM_NEAR_THRESH logic):
        n_close = count of hull vertices within _WM_NEAR_THRESH of the edge
        d_centroid = distance from hull centroid to the edge
        PASSES if  n_close >= 2  OR  d_centroid <= _WM_NEAR_THRESH * 2.5

    Hull source priority:
        1. bev_objects.json (hull_xy — mesh-derived, same source as Stage 2)
        2. blend_info.json  (AABB half-widths → 4-corner box, fallback)

    Args:
        polygon_v2:    The in-memory polygon_v2 dict (mutated in-place).
        object_state:  Loaded object_state.json dict (used to find wall-mounted
                       objects and their class).  May be None.
        scene_dir:     Scene root path — used to locate bev_objects.json and
                       blend_info.json for hull data.
    """
    wall_edges = polygon_v2.get("wall_edges", [])
    verts = polygon_v2.get("polygon_vertices", [])
    if not wall_edges or not verts:
        return

    # ── Collect wall-mounted object ids (same gate as build_wall_set) ──────
    wall_obj_ids: set[str] = set()
    if object_state:
        for obj_entry in object_state.get("objects", []):
            obj_id = obj_entry.get("obj_id", "")
            if obj_id and "wall" in obj_entry.get("attached_to", []):
                wall_obj_ids.add(obj_id)
    # Also include any object explicitly listed in polygon_v2 wall_mount_objects
    for wmo in polygon_v2.get("wall_mount_objects", []):
        wmo_id = wmo.get("id", "")
        if wmo_id:
            wall_obj_ids.add(wmo_id)

    if not wall_obj_ids:
        return

    # ── Build hull_xy lookup: obj_id → list of [x, y] ───────────────────
    hulls: dict[str, list] = {}

    # Priority 1: bev_objects.json (mesh-derived convex hull)
    bev_path = scene_dir / "json" / "bev_objects.json"
    if bev_path.exists():
        try:
            bev_data = json.loads(bev_path.read_text(encoding="utf-8"))
            for o in bev_data.get("objects", []):
                oid = o.get("id", "")
                hxy = o.get("hull_xy")
                if oid and hxy:
                    hulls[oid] = [[float(p[0]), float(p[1])] for p in hxy]
        except Exception as exc:
            print(
                f"[graph_tool] WARNING — could not parse bev_objects.json for hull "
                f"refresh: {exc}",
                file=sys.stderr,
            )

    # Priority 2: blend_info.json AABB fallback for objects missing from bev_objects
    missing = wall_obj_ids - set(hulls.keys())
    if missing:
        blend_info_path = _resolve_blend_info_path(scene_dir)
        if blend_info_path.exists():
            try:
                blend_data = json.loads(blend_info_path.read_text(encoding="utf-8"))
                cats = blend_data.get("categories", {})
                all_blend_objs: dict = {}
                for cat in ["objects", "world", "geometry_meshes"]:
                    for obj in cats.get(cat, []):
                        all_blend_objs[obj.get("name", "")] = obj

                def _collect_max_dims(node_name: str) -> tuple[float, float]:
                    """Recursively find max (dim_x, dim_y) among MESH descendants."""
                    node = all_blend_objs.get(node_name)
                    if not node:
                        return 0.0, 0.0
                    mx, my = 0.0, 0.0
                    if node.get("type") == "MESH":
                        dims = node.get("dimensions", [0, 0, 0])
                        if len(dims) >= 2:
                            mx, my = float(dims[0]), float(dims[1])
                    for child in node.get("children", []):
                        cx, cy = _collect_max_dims(child)
                        mx, my = max(mx, cx), max(my, cy)
                    return mx, my

                for oid in missing:
                    node = all_blend_objs.get(oid)
                    if not node:
                        continue
                    loc = node.get("location", [0.0, 0.0, 0.0])
                    ox, oy = float(loc[0]), float(loc[1])
                    dx, dy = _collect_max_dims(oid)
                    if dx < 1e-6 and dy < 1e-6:
                        # No usable AABB — use a single-point hull at the object origin
                        hulls[oid] = [[ox, oy]]
                    else:
                        hw, hd = dx / 2.0, dy / 2.0
                        hulls[oid] = [
                            [ox - hw, oy - hd],
                            [ox + hw, oy - hd],
                            [ox + hw, oy + hd],
                            [ox - hw, oy + hd],
                        ]
            except Exception as exc:
                print(
                    f"[graph_tool] WARNING — could not parse blend_info for hull "
                    f"refresh: {exc}",
                    file=sys.stderr,
                )

    # Build class lookup from object_state for the evidence rows
    obj_class: dict[str, str] = {}
    if object_state:
        for obj_entry in object_state.get("objects", []):
            oid = obj_entry.get("obj_id", "")
            cls = obj_entry.get("category", "")
            if oid:
                obj_class[oid] = cls
    # Supplement from polygon_v2 wall_mount_objects
    for wmo in polygon_v2.get("wall_mount_objects", []):
        wmo_id = wmo.get("id", "")
        wmo_cls = wmo.get("class", "")
        if wmo_id and wmo_id not in obj_class:
            obj_class[wmo_id] = wmo_cls

    thresh = _WM_NEAR_THRESH
    gate_centroid = thresh * 2.5

    # ── Wipe existing evidence lists and recompute ───────────────────────
    for edge in wall_edges:
        edge["wall_mount_evidence"] = []

    for obj_id in sorted(wall_obj_ids):
        hull = hulls.get(obj_id)
        if not hull:
            continue  # no geometry data — skip
        n_hull = len(hull)
        if n_hull == 0:
            continue

        centroid_x = sum(p[0] for p in hull) / n_hull
        centroid_y = sum(p[1] for p in hull) / n_hull
        cls = obj_class.get(obj_id, "")

        for edge in wall_edges:
            fi = edge.get("from")
            ti = edge.get("to")
            if fi is None or ti is None or fi >= len(verts) or ti >= len(verts):
                continue
            ax, ay = verts[fi][0], verts[fi][1]
            bx, by = verts[ti][0], verts[ti][1]

            dists = [
                _pt_seg_dist_inline(p[0], p[1], ax, ay, bx, by)
                for p in hull
            ]
            n_close = sum(1 for d in dists if d <= thresh)
            d_centroid = _pt_seg_dist_inline(centroid_x, centroid_y, ax, ay, bx, by)

            if n_close >= 2 or d_centroid <= gate_centroid:
                edge["wall_mount_evidence"].append(
                    {"id": obj_id, "class": cls, "dist_m": round(min(dists), 3)}
                )

    n_walls = len(wall_edges)
    n_objs = len(wall_obj_ids)
    print(
        f"[graph_tool] refreshed wall_mount_evidence: {n_walls} walls × {n_objs} wall-mounted objects"
    )


def _pt_seg_dist_inline(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Inline duplicate of _pt_seg_dist used before that function is defined."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


# ---------------------------------------------------------------------------
# Wall resolution — NEW: nearest point-to-segment distance
# ---------------------------------------------------------------------------

def _pt_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Perpendicular (clamped) distance from point (px,py) to segment (a→b)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def nearest_wall(obj_xy: list, polygon_v2: dict, obj_id: str | None = None) -> tuple:
    """
    Find the wall an object should attach to.

    Priority:
    1. If obj_id is provided AND it appears in any wall_edges[].wall_mount_evidence[],
       return that wall directly with wall_ambiguous=False (the polygon-builder
       already verified geometric proximity when emitting evidence).
    2. Otherwise fall back to point-to-segment distance from obj_xy.

    Args:
        obj_xy: [x, y] position of the object
        polygon_v2: polygon_v2.json dict
        obj_id: optional object id ("obj_3") for evidence lookup

    Returns:
        (wall_obj, nearest_dist, wall_ambiguous, wall_candidates, all_dists)
        ── wall_obj      — name of selected wall (str), or None if no walls found
        ── nearest_dist  — distance in metres
        ── wall_ambiguous — True if 2nd-nearest is within margin
        ── wall_candidates — [nearest_wall, second_wall]
        ── all_dists     — [(dist, wall_name), ...] sorted ascending

    Ambiguity criterion (geometric fallback only):
        second_dist - nearest_dist < 0.5 m
        OR second_dist < 1.2 * nearest_dist
    """
    wall_edges = polygon_v2.get("wall_edges", [])
    verts = polygon_v2.get("polygon_vertices", [])

    if not verts or not wall_edges:
        return (None, float("inf"), False, [], [])

    # ── Build full candidate list (wall_index, wall_name, perp_dist) ─────────
    px, py = obj_xy[0], obj_xy[1]
    cands = []  # (dist, wall_name, wall_index)
    for i, edge in enumerate(wall_edges):
        from_idx = edge.get("from")
        to_idx = edge.get("to")
        wall_name = edge.get("object", "")
        if from_idx is None or to_idx is None:
            continue
        if from_idx >= len(verts) or to_idx >= len(verts):
            continue
        v0 = verts[from_idx]
        v1 = verts[to_idx]
        d = _pt_seg_dist(px, py, v0[0], v0[1], v1[0], v1[1])
        cands.append((d, wall_name, i))

    if not cands:
        return (None, float("inf"), False, [], [])

    # ── Build whitelist from wall_mount_evidence ──────────────────────────
    whitelist = set()
    if obj_id:
        for i, edge in enumerate(wall_edges):
            for ev in edge.get("wall_mount_evidence", []) or []:
                if ev.get("id") == obj_id:
                    whitelist.add(i)

    # ── Priority 1: whitelist non-empty → pick closest whitelisted wall ───
    if whitelist:
        wl_cands = [(d, wn, wi) for d, wn, wi in cands if wi in whitelist]
        wl_cands.sort(key=lambda x: x[0])
        winner_dist, winner_name, _ = wl_cands[0]
        runner_up = wl_cands[1] if len(wl_cands) > 1 else None
        wall_ambiguous = (
            runner_up is not None and (runner_up[0] - winner_dist < 0.5)
        )
        all_dists = [(d, wn) for d, wn, _ in sorted(cands, key=lambda x: x[0])]
        candidates = [winner_name] + ([runner_up[1]] if runner_up else [])
        print(
            f"graph_tool_planner: {obj_id} → {winner_name} via "
            f"wall_mount_evidence whitelist (dist {winner_dist:.3f}m, "
            f"ambiguous={wall_ambiguous})",
            file=sys.stderr,
        )
        return (winner_name, winner_dist, wall_ambiguous, candidates, all_dists)

    # ── Priority 2: geometric nearest wall (no whitelist) ────────────────
    cands.sort(key=lambda x: x[0])
    nearest_dist, nearest_wall_obj, _ = cands[0]
    all_dists = [(d, wn) for d, wn, _ in cands]

    if len(cands) < 2:
        return (nearest_wall_obj, nearest_dist, False, [nearest_wall_obj], all_dists)

    second_dist, second_wall_obj, _ = cands[1]
    ambiguous = (second_dist - nearest_dist < 0.5) or (second_dist < 1.2 * nearest_dist)
    candidates = [nearest_wall_obj, second_wall_obj]
    return (nearest_wall_obj, nearest_dist, ambiguous, candidates, all_dists)


# ---------------------------------------------------------------------------
# Per-object wall-mounted determination
# ---------------------------------------------------------------------------

def build_on_top_of_surface_members(relation_graph: dict) -> set:
    """
    Build the set of objects that are members of an on_top_of group whose
    anchor is an object (not Floor/floor). These objects rest on a surface
    and must NOT receive wall attachment even if their class is wall-like.

    Mirrors the exclusion logic in heuristic_planner._relation_exclusions.
    """
    excluded = set()
    for group in relation_graph.get("groups", []):
        if group.get("edge_type") != "on_top_of":
            continue
        anchor = str(group.get("anchor", "")).lower()
        if anchor == "floor":
            continue  # rests on the floor — not an exclusion
        # anchor is an object (e.g. "obj_19") — members are surface-resting
        excluded.update(group.get("members", []))
    return excluded


def build_class_lookup(object_state: dict, object_class: dict) -> dict:
    """
    Build {obj_id: class_name} from object_state.json (label_id ↔ obj_id mapping)
    and object_class.json (label_id → class_name).

    Returns empty dict if either input is None/empty.
    """
    if not object_state or not object_class:
        return {}
    obj_to_class = {}
    for obj_entry in object_state.get("objects", []):
        obj_id = obj_entry.get("obj_id", "")
        label_id = obj_entry.get("label_id")
        if not obj_id or label_id is None:
            continue
        cls = object_class.get(str(label_id))
        if cls:
            obj_to_class[obj_id] = cls
    return obj_to_class


def build_wall_set(
    object_state: dict,
    obj_to_class: dict,
    on_top_of_surface_members: set,
) -> dict:
    """
    Determine which objects are wall-mounted and the basis for each decision.

    Returns: {obj_id: basis_string} for all wall-mounted objects.

    An object is wall-mounted if:
      PRIMARY  — object_state.json attached_to contains "wall"
      FALLBACK — object's class matches WALL_CLASSES
    AND it is NOT excluded because it is a member of an on_top_of group
    whose anchor is a surface object (not Floor).

    basis_string is one of:
      "attached_to=wall"
      "class=<class_name>"
      "attached_to=wall; class=<class_name>"
    """
    wall_objs = {}

    # Collect attached_to=wall candidates
    attached_wall = set()
    for obj_entry in object_state.get("objects", []):
        obj_id = obj_entry.get("obj_id", "")
        if not obj_id:
            continue
        if "wall" in obj_entry.get("attached_to", []):
            attached_wall.add(obj_id)

    # Collect class-fallback candidates
    class_wall = set()
    for obj_id, cls in obj_to_class.items():
        if _class_matches(cls, WALL_CLASSES):
            class_wall.add(obj_id)

    # Union, then apply on_top_of surface exclusion
    candidates = attached_wall | class_wall
    for obj_id in sorted(candidates):
        if obj_id in on_top_of_surface_members:
            continue  # excluded: rests on a surface object

        # Build basis string
        parts = []
        if obj_id in attached_wall:
            parts.append("attached_to=wall")
        if obj_id in class_wall:
            cls = obj_to_class.get(obj_id, "?")
            parts.append(f"class={cls!r}")
        basis = "; ".join(parts) if parts else "unknown"
        wall_objs[obj_id] = basis

    return wall_objs


# ---------------------------------------------------------------------------
# Core planning logic
# ---------------------------------------------------------------------------

def build_ops(
    relation_graph: dict,
    polygon_v2: dict,
    blend_info: dict,
    object_state: dict,
    object_class: dict,
    scene_dir: Path,
) -> dict:
    """
    Translate relation_graph groups + per-object wall detection into
    deterministic tool ops.

    Returns the full graph_ops dict ready for JSON serialisation.
    Also writes json/wall_ambiguous.json as a side-effect.
    """
    groups: list = relation_graph.get("groups", [])

    # ------------------------------------------------------------------
    # A0. Defense-in-depth: refresh stale wall_mount_evidence in polygon_v2.
    #     Scenes whose Stage 2 ran BEFORE the classify_edges patch may have
    #     incomplete evidence lists (single-vertex criterion).  Recompute
    #     in-memory so nearest_wall() gets a correct whitelist.
    # ------------------------------------------------------------------
    _refresh_wall_mount_evidence(polygon_v2, object_state, scene_dir)

    # ------------------------------------------------------------------
    # A. Build per-object wall_set (independent of relation_graph wall groups)
    # ------------------------------------------------------------------
    # 1. Build on_top_of surface exclusions from relation_graph
    on_top_of_surface_members = build_on_top_of_surface_members(relation_graph)

    # 2. Build obj_id -> class_name lookup
    obj_to_class = build_class_lookup(object_state or {}, object_class or {})

    # 3. Determine wall_set: {obj_id: basis}
    wall_set = build_wall_set(object_state or {}, obj_to_class, on_top_of_surface_members)

    # ------------------------------------------------------------------
    # B. Build blend_info object index (for location lookup)
    # ------------------------------------------------------------------
    categories = blend_info.get("categories", {}) if blend_info else {}
    all_objects = {}
    for category in ["objects", "world", "geometry_meshes"]:
        for obj in categories.get(category, []):
            all_objects[obj.get("name")] = obj

    # ------------------------------------------------------------------
    # C. Emit attach_to_wall ops for wall_set using nearest_wall()
    # ------------------------------------------------------------------
    operation_list: list = []
    wall_attach_count = 0
    surface_attach_count = 0
    skipped_seated_around = 0
    skipped_other = 0
    n_skipped_distance = 0
    skipped_distance_objs: list = []

    ambiguous_records = []  # for wall_ambiguous.json

    for obj_id in sorted(wall_set.keys()):
        basis = wall_set[obj_id]

        # Get object (x,y) from blend_info; fall back to [0,0] with a warning
        obj_info = all_objects.get(obj_id)
        if obj_info:
            loc = obj_info.get("location", [0.0, 0.0, 0.0])
            obj_xy = [loc[0], loc[1]]
        else:
            print(
                f"graph_tool_planner: WARNING — {obj_id} not found in blend_info; "
                f"using [0,0] for wall distance (wall assignment may be inaccurate).",
                file=sys.stderr,
            )
            obj_xy = [0.0, 0.0]

        wall_obj, nearest_dist, wall_ambiguous, candidates, all_dists = nearest_wall(
            obj_xy, polygon_v2, obj_id=obj_id   # NEW: pass obj_id for evidence lookup
        )

        if wall_obj is None:
            print(
                f"graph_tool_planner: WARNING — could not resolve wall for {obj_id} "
                f"(no wall edges in polygon_v2); skipping.",
                file=sys.stderr,
            )
            skipped_other += 1
            continue

        # Distance sanity check: skip objects whose nearest wall exceeds the
        # threshold — the upstream attached_to=wall label is likely mis-classified.
        class_or_basis = obj_to_class.get(obj_id, basis)
        if nearest_dist > WALL_ATTACH_MAX_DIST_M:
            print(
                f"[graph_tool] skipping wall_attach for {obj_id} ({class_or_basis}) "
                f"— nearest wall {wall_obj} is {nearest_dist:.3f}m away "
                f"(> {WALL_ATTACH_MAX_DIST_M}m threshold); treating as free-standing."
            )
            n_skipped_distance += 1
            skipped_distance_objs.append(obj_id)
            continue

        # preserve_rotation omitted → dispatcher default (False) forces rz alignment to wall tangent.
        # Build op
        op = {
            "action": "attach_to_wall",
            "wall_obj": wall_obj,
            "moving_obj": obj_id,
            "wall_ambiguous": wall_ambiguous,
            "reason": (
                f"per-object wall-mounted ({basis}); nearest wall {wall_obj} "
                f"({nearest_dist:.3f}m)"
            ),
            "priority": 5,
            "source": "graph_tool",
        }
        if wall_ambiguous:
            op["wall_candidates"] = candidates

        # Compute t_along_m from the object's projected position on the wall
        wall_frame = compute_wall_frame(wall_obj, polygon_v2)
        if wall_frame and obj_info:
            loc = obj_info.get("location", [0.0, 0.0, 0.0])
            v_from, _v_to, L, t_unit = wall_frame
            dx_from = loc[0] - v_from[0]
            dy_from = loc[1] - v_from[1]
            t_proj = dx_from * t_unit[0] + dy_from * t_unit[1]
            t_clamped = max(0.0, min(L, t_proj))
            op["t_along_m"] = round(t_clamped, 4)

        operation_list.append(op)
        wall_attach_count += 1

        # Track ambiguous objects for the summary file
        if wall_ambiguous:
            ambiguous_records.append({
                "obj": obj_id,
                "candidates": candidates,
                "dists": [round(d, 4) for d, _ in all_dists[:2]],
            })

    # ------------------------------------------------------------------
    # D. Write wall_ambiguous.json (hook for future LLM tie-break)
    # ------------------------------------------------------------------
    wall_ambiguous_path = scene_dir / "json" / "wall_ambiguous.json"
    try:
        wall_ambiguous_path.parent.mkdir(parents=True, exist_ok=True)
        with wall_ambiguous_path.open("w", encoding="utf-8") as fh:
            json.dump({"ambiguous": ambiguous_records}, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(
            f"graph_tool_planner: WARNING — could not write wall_ambiguous.json: {exc}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # E. Process remaining relation_graph groups
    #    - mounted_on_same_wall: REMOVED from wall-attach path (wall_set above replaces it)
    #    - on_top_of: KEPT unchanged
    #    - all others: KEPT unchanged (skip)
    # ------------------------------------------------------------------
    for group in groups:
        group_id = group.get("group_id", group.get("id", "?"))
        edge_type = group.get("edge_type", "")
        members: list = group.get("members", [])
        anchor: str = group.get("anchor", "")

        # --- mounted_on_same_wall: no longer drives wall attachment ---
        # (wall_set above handles all wall objects via per-object detection)
        if edge_type == "mounted_on_same_wall":
            # intentionally skipped — superseded by wall_set logic
            pass

        # --- on_top_of ---
        elif edge_type == "on_top_of":
            if not anchor:
                print(
                    f"graph_tool_planner: WARNING — group {group_id} has edge_type on_top_of "
                    f"but no anchor; skipping.",
                    file=sys.stderr,
                )
                skipped_other += len(members)
                continue

            for obj in members:
                if obj == anchor:
                    continue
                operation_list.append(
                    {
                        "action": "attach",
                        "anchor_obj": anchor,
                        "moving_obj": obj,
                        "relation": "on",
                        "reason": f"on_top_of {group_id}",
                        "priority": 3,
                        "source": "graph_tool",
                    }
                )
                surface_attach_count += 1

        # --- seated_around: needs island refinement ---
        elif edge_type == "seated_around":
            skipped_seated_around += len(members)

        # --- all other edge types: skip ---
        else:
            skipped_other += len(members)

    summary = {
        "wall_attach_count": wall_attach_count,
        "surface_attach_count": surface_attach_count,
        "skipped_seated_around": skipped_seated_around,
        "skipped_other": skipped_other,
        "wall_set": {obj: basis for obj, basis in wall_set.items()},
        "on_top_of_surface_excluded": sorted(on_top_of_surface_members),
        "wall_ambiguous_count": len(ambiguous_records),
        "wall_attach_skipped_distance_count": n_skipped_distance,
        "wall_attach_skipped_distance_objs": skipped_distance_objs,
    }

    return {
        "operation_list": operation_list,
        "_graph_tool_summary": summary,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path, label: str, required: bool = True) -> dict | None:
    """Load and return a JSON file.

    If required=True (default): exit on missing/invalid.
    If required=False: return None on missing/invalid (with a warning).
    """
    if not path.exists():
        if required:
            print(f"graph_tool_planner: ERROR — {label} not found at {path}", file=sys.stderr)
            sys.exit(1)
        else:
            print(
                f"graph_tool_planner: WARNING — optional {label} not found at {path}; skipping.",
                file=sys.stderr,
            )
            return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        if required:
            print(
                f"graph_tool_planner: ERROR — invalid JSON in {label} ({path}): {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                f"graph_tool_planner: WARNING — invalid JSON in optional {label} ({path}): {exc}; skipping.",
                file=sys.stderr,
            )
            return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit deterministic tool ops from relation_graph.json + blend_info.json "
            "without invoking an LLM."
        )
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=Path,
        help="Root scene directory (must contain inputs/ and json/ sub-dirs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for graph_ops.json (default: <scene_dir>/json/graph_ops.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir: Path = args.scene_dir.resolve()

    relation_graph_path = scene_dir / "inputs" / "relation_graph.json"
    polygon_v2_path = scene_dir / "json" / "polygon_v2.json"
    blend_info_path = _resolve_blend_info_path(scene_dir)
    object_state_path = scene_dir / "json" / "object_state.json"
    object_class_path = scene_dir / "inputs" / "object_class.json"
    output_path: Path = args.output or scene_dir / "json" / "graph_ops.json"

    relation_graph = load_json(relation_graph_path, "relation_graph")
    polygon_v2 = load_json(polygon_v2_path, "polygon_v2")
    blend_info = load_json(blend_info_path, "blend_info")
    object_state = load_json(object_state_path, "object_state", required=False)
    object_class = load_json(object_class_path, "object_class", required=False)

    result = build_ops(
        relation_graph,
        polygon_v2,
        blend_info,
        object_state,
        object_class,
        scene_dir,
    )
    summary = result["_graph_tool_summary"]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(
        f"graph_tool_planner: wall_attach={summary['wall_attach_count']} "
        f"surface_attach={summary['surface_attach_count']} "
        f"skipped_seated_around={summary['skipped_seated_around']} "
        f"wall_ambiguous={summary['wall_ambiguous_count']}"
    )
    print(f"graph_tool_planner: wall_set={sorted(summary['wall_set'].keys())}")
    print(
        f"graph_tool_planner: on_top_of_excluded={summary['on_top_of_surface_excluded']}"
    )
    print(f"graph_tool_planner: output written to {output_path}")


if __name__ == "__main__":
    main()
