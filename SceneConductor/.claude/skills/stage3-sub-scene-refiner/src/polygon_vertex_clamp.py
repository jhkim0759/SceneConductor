"""
polygon_vertex_clamp.py — Precise polygon containment check using real mesh vertices.

Replaces the OBB-based _polygon_clamp_ops in heuristic_planner.py.
Runs inside Blender's bundled Python (bpy available).

CLI:
    blender --background <blend_in> --python polygon_vertex_clamp.py -- \\
        --scene-dir <scene_dir> --output-ops <output_path.json>

Reads:
    <scene_dir>/json/polygon_v2.json       — room polygon
    <scene_dir>/inputs/object_class.json   — label_id → class string
    <scene_dir>/json/object_state.json     — attached_to (soft-load)

Algorithm per obj_* object:
    1. Collect every mesh-descendant vertex in world-space XY (recursive walk).
    2. For each polygon edge compute the most-outward vertex (min signed dist).
    3. Wall-mounted objects: skip their closest-edge (the attached wall).
    4. For each non-skipped violated edge accumulate an inward push.
    5. Emit update_layout op if total push is nonzero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# WALL_CLASSES taxonomy — kept in sync with heuristic_planner.py manually.
# ---------------------------------------------------------------------------
WALL_CLASSES = [
    "chalkboard", "blackboard", "whiteboard", "poster", "picture frame",
    "painting", "mirror", "clock", "board", "tv", "television",
]


def _class_matches(class_name: str, taxonomy: list[str]) -> bool:
    if not class_name:
        return False
    cl = class_name.lower()
    return any(kw in cl for kw in taxonomy)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str, required: bool = True) -> dict | None:
    if not path.exists():
        if required:
            print(f"[polygon_vertex_clamp] ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)
        print(f"[polygon_vertex_clamp] INFO: {label} not found at {path} — skipping.", file=sys.stderr)
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[polygon_vertex_clamp] ERROR: invalid JSON in {label} ({path}): {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Polygon geometry helpers
# ---------------------------------------------------------------------------

def _build_edge_normals(
    verts: list[list[float]],
    cx: float,
    cy: float,
) -> list[tuple[float, float, float, float]]:
    """Return list of (nx, ny, ax, ay) inward unit normals for each edge."""
    n = len(verts)
    result: list[tuple[float, float, float, float]] = []
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9:
            result.append((0.0, 0.0, ax, ay))
            continue
        nx, ny = -dy / length, dx / length
        # Ensure normal faces toward centroid
        mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
        if (cx - mx) * nx + (cy - my) * ny < 0:
            nx, ny = -nx, -ny
        result.append((nx, ny, ax, ay))
    return result


def _signed_dist(px: float, py: float, normal: tuple[float, float, float, float]) -> float:
    """Signed distance of point from the edge (positive = inside room)."""
    nx, ny, ax, ay = normal
    return (px - ax) * nx + (py - ay) * ny


# ---------------------------------------------------------------------------
# Blender vertex collection
# ---------------------------------------------------------------------------

def _collect_mesh_vertices_xy(root_obj) -> list[tuple[float, float]]:
    """Recursively collect all MESH-descendant world-space (x, y) vertex coords.

    Walks root_obj and all its children recursively.  For each MESH object,
    transforms every vertex by the object's matrix_world and keeps only xy.
    """
    import bpy  # noqa: F401 — available inside Blender Python
    from mathutils import Vector

    xy_verts: list[tuple[float, float]] = []

    def _walk(obj) -> None:
        if obj.type == "MESH" and obj.data is not None:
            mw = obj.matrix_world
            for v in obj.data.vertices:
                world_v = mw @ Vector(v.co)
                xy_verts.append((world_v.x, world_v.y))
        for child in obj.children:
            _walk(child)

    _walk(root_obj)
    return xy_verts


# ---------------------------------------------------------------------------
# Core per-object clamp
# ---------------------------------------------------------------------------

def _process_object(
    bpy_obj,
    obj_id: str,
    cls: str,
    is_wall_mounted: bool,
    edge_normals: list[tuple[float, float, float, float]],
    edge_names: dict[int, str],
    margin: float,
) -> dict | None:
    """Compute the required push for one obj_* Blender object.

    Returns an update_layout op dict or None if no violation.
    """
    xy_verts = _collect_mesh_vertices_xy(bpy_obj)
    if not xy_verts:
        print(
            f"[polygon_vertex_clamp] WARNING: {obj_id} has no mesh vertices — skipping.",
            file=sys.stderr,
        )
        return None

    n_edges = len(edge_normals)
    loc = bpy_obj.location
    lx, ly, lz = loc.x, loc.y, loc.z

    # Identify attached edge (closest edge to object origin) for wall-mounted objects.
    skip_edges: set[int] = set()
    attached_edge_idx: int = -1
    attached_edge_name: str = ""
    if is_wall_mounted:
        min_sd = math.inf
        for ei, normal in enumerate(edge_normals):
            sd = _signed_dist(lx, ly, normal)
            if sd < min_sd:
                min_sd = sd
                attached_edge_idx = ei
        if attached_edge_idx >= 0:
            skip_edges.add(attached_edge_idx)
            attached_edge_name = edge_names.get(attached_edge_idx, f"Edge_{attached_edge_idx}")

    # For each edge: find the most-outward vertex (minimum signed dist).
    min_d: list[float] = []
    worst_vx: list[tuple[float, float]] = []
    for ei, normal in enumerate(edge_normals):
        cur_min = math.inf
        cur_vx = (lx, ly)
        for vx, vy in xy_verts:
            sd = _signed_dist(vx, vy, normal)
            if sd < cur_min:
                cur_min = sd
                cur_vx = (vx, vy)
        min_d.append(cur_min)
        worst_vx.append(cur_vx)

    # Accumulate push vectors from violated non-skipped edges.
    total_dx = 0.0
    total_dy = 0.0
    violated_walls: list[str] = []
    worst_violation_depth = 0.0
    worst_violation_edge_name = ""
    worst_vertex_xy: tuple[float, float] = (lx, ly)
    worst_vertex_depth: float = 0.0

    for ei in range(n_edges):
        if ei in skip_edges:
            continue
        md = min_d[ei]
        if md >= margin:
            continue
        depth = margin - md
        nx, ny, _, _ = edge_normals[ei]
        total_dx += depth * nx
        total_dy += depth * ny
        wname = edge_names.get(ei, f"Edge_{ei}")
        violated_walls.append(wname)
        if depth > worst_violation_depth:
            worst_violation_depth = depth
            worst_violation_edge_name = wname
            worst_vertex_xy = worst_vx[ei]
            worst_vertex_depth = -min_d[ei]  # negative = outside room

    if not violated_walls:
        return None

    new_x = lx + total_dx
    new_y = ly + total_dy
    push_dist = math.hypot(total_dx, total_dy)
    dx_sign = "+" if total_dx >= 0 else ""
    dy_sign = "+" if total_dy >= 0 else ""
    walls_str = ", ".join(violated_walls)

    if is_wall_mounted and attached_edge_name:
        reason = (
            f"polygon clamp (vertex): wall-mounted, attached_edge={attached_edge_name} (skipped); "
            f"pushed {dx_sign}{total_dx:.3f}m /x {dy_sign}{total_dy:.3f}m /y "
            f"(total {push_dist:.3f}m) to clear {worst_violation_edge_name} "
            f"(most-outward vertex at depth {worst_vertex_depth:.3f}m, "
            f"vx=({worst_vertex_xy[0]:.3f},{worst_vertex_xy[1]:.3f}), "
            f"attached_edge={attached_edge_name} skipped)"
        )
    else:
        reason = (
            f"polygon clamp (vertex): pushed {dx_sign}{total_dx:.3f}m /x "
            f"{dy_sign}{total_dy:.3f}m /y "
            f"(total {push_dist:.3f}m) to clear {walls_str} "
            f"(most-outward vertex at depth {worst_vertex_depth:.3f}m, "
            f"vx=({worst_vertex_xy[0]:.3f},{worst_vertex_xy[1]:.3f}))"
        )

    return {
        "action": "update_layout",
        "obj_name": obj_id,
        "location": [new_x, new_y, lz],
        "reason": reason,
        "criteria_used": ["polygon_clamp", "vertex"],
        "priority": 6,
        "confidence": 1.0,
        "requires_planner_review": False,
        "source": "polygon_clamp",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Parse arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Vertex-based polygon containment clamp (runs inside Blender)."
    )
    parser.add_argument("--scene-dir", required=True, type=Path, help="Scene root directory.")
    parser.add_argument(
        "--output-ops",
        required=True,
        type=Path,
        help="Output JSON path for emitted ops.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Minimum inward margin in metres (default: 0.05).",
    )
    args = parser.parse_args(argv)

    scene_dir: Path = args.scene_dir.resolve()
    output_path: Path = args.output_ops.resolve()
    margin: float = args.margin

    # -- Load input JSONs --
    polygon_data = _load_json(scene_dir / "json" / "polygon_v2.json", "polygon_v2.json", required=True)
    object_class = _load_json(scene_dir / "inputs" / "object_class.json", "object_class.json", required=True)
    object_state = _load_json(scene_dir / "json" / "object_state.json", "object_state.json", required=False)

    # -- Build polygon geometry --
    verts: list[list[float]] = polygon_data.get("polygon_vertices", [])
    if len(verts) < 3:
        print(
            "[polygon_vertex_clamp] ERROR: polygon_v2.json has fewer than 3 vertices.",
            file=sys.stderr,
        )
        sys.exit(1)

    centroid_xy: list[float] = polygon_data.get("polygon_centroid_xy", [0.0, 0.0])
    cx, cy = centroid_xy[0], centroid_xy[1]

    edge_normals = _build_edge_normals(verts, cx, cy)

    # Build edge index → wall name from polygon_v2 wall_edges.
    edge_names: dict[int, str] = {}
    for we in polygon_data.get("wall_edges", []):
        fi = we.get("from")
        if fi is not None:
            edge_names[fi] = we.get("object", f"Edge_{fi}")

    # -- Build wall-mount set from object_state (attached_to includes 'wall') --
    wall_attached_ids: set[str] = set()
    if object_state is not None:
        for entry in object_state.get("objects", []):
            eid = entry.get("obj_id", "")
            if "wall" in entry.get("attached_to", []):
                wall_attached_ids.add(eid)

    # -- Build obj_id → class string from object_class (key = str label_id) --
    def _obj_class(obj_id: str) -> str:
        suffix = obj_id[len("obj_"):]  # "obj_3" → "3"
        return object_class.get(suffix, "")  # type: ignore[return-value]

    def _is_wall_mounted(obj_id: str) -> bool:
        if obj_id in wall_attached_ids:
            return True
        cls = _obj_class(obj_id)
        return bool(cls and _class_matches(cls, WALL_CLASSES))

    # -- Import bpy (only available inside Blender Python) --
    try:
        import bpy  # type: ignore[import]
    except ImportError:
        print(
            "[polygon_vertex_clamp] ERROR: bpy not available — run this script inside Blender.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Main loop over obj_* top-level objects --
    ops: list[dict] = []
    n_checked = 0
    n_walls_processed = 0
    n_attached_edges_skipped = 0
    total_verts_checked = 0
    max_violation_depth = 0.0

    for bpy_obj in bpy.data.objects:
        obj_id: str = bpy_obj.name
        if not obj_id.startswith("obj_"):
            continue
        # Only process top-level obj_* (no obj_* parent)
        if bpy_obj.parent is not None and bpy_obj.parent.name.startswith("obj_"):
            continue

        n_checked += 1
        is_wall = _is_wall_mounted(obj_id)
        if is_wall:
            n_walls_processed += 1
            n_attached_edges_skipped += 1

        cls = _obj_class(obj_id)

        # Count vertices for summary (pre-process)
        xy_verts = _collect_mesh_vertices_xy(bpy_obj)
        total_verts_checked += len(xy_verts)

        # Find max violation depth for summary
        for ei, normal in enumerate(edge_normals):
            for vx, vy in xy_verts:
                sd = _signed_dist(vx, vy, normal)
                if sd < -max_violation_depth:
                    max_violation_depth = -sd

        op = _process_object(
            bpy_obj,
            obj_id,
            cls,
            is_wall,
            edge_normals,
            edge_names,
            margin,
        )
        if op is not None:
            ops.append(op)
            print(
                f"[polygon_vertex_clamp] {obj_id} ({cls or '?'}): "
                f"dx={op['location'][0] - bpy_obj.location.x:+.4f}m "
                f"dy={op['location'][1] - bpy_obj.location.y:+.4f}m — "
                f"{op['reason'][:80]}",
                file=sys.stderr,
            )

    # -- Emit output --
    output = {
        "operation_list": ops,
        "polygon_clamp_summary": {
            "objects_checked": n_checked,
            "walls_processed": n_walls_processed,
            "attached_edges_skipped": n_attached_edges_skipped,
            "ops_emitted": len(ops),
            "total_vertices_checked": total_verts_checked,
            "max_violation_depth_m": round(max_violation_depth, 6),
            "margin_m": margin,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(
        f"[polygon_vertex_clamp] Done. "
        f"checked={n_checked} "
        f"walls_processed={n_walls_processed} "
        f"attached_edges_skipped={n_attached_edges_skipped} "
        f"ops_emitted={len(ops)} "
        f"total_verts={total_verts_checked} "
        f"max_violation_depth={max_violation_depth:.4f}m — "
        f"output → {output_path}"
    )


if __name__ == "__main__":
    main()
