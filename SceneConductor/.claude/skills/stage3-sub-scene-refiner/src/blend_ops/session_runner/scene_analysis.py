"""scene_analysis.py — 4-type structural scene analysis for SceneWeaver sessions.

CLI:
    python scene_analysis.py <session_dir> [--iter <N>]
    python scene_analysis.py --selftest

If --iter is omitted, resolves N from <session_dir>/current symlink.
Writes <session_dir>/iter_<N>/analysis.json and prints a one-line summary.

Design note on EMPTY-parent objects:
  Blender exports obj_N as EMPTY parents whose dimensions=[0,0,0]. The actual
  geometry lives in children. We therefore use:
    - location[2] = the object's canonical world Z (the empty's origin)
    - scale[0] as a proxy for overall object size
  For floor/ceiling detection we use the metrics room_bbox Z bounds as
  authoritative anchors (floor = room_bbox[0][2], ceiling = room_bbox[1][2]).
  For z-consistency checks (Type 2) we compare location[2] directly across
  instances of the same mesh_group.
  For on-surface pairs (Type 3) we skip height-gap checks when dimensions are
  zero — instead we only report xy-overlap candidates and flag based on z
  relative to the surface's location (using a fixed 0.70 m table-height prior).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Anchor class tables  (lowercased substring matching)
# ---------------------------------------------------------------------------
WALL_CLASSES = [
    "chalkboard", "blackboard", "whiteboard", "poster", "picture frame",
    "painting", "mirror", "clock", "board", "tv", "television",
]
CEILING_CLASSES = [
    "fluorescent light", "ceiling light", "chandelier", "fan", "projector",
]
FLOOR_CLASSES = [
    "chair", "table", "counter", "desk", "sofa", "bed", "cabinet", "shelf",
    "bookshelf", "rug", "drawer", "toy bin", "person", "plant pot",
]
ON_SURFACE_CANDIDATE = [
    "book", "cup", "lamp", "pen", "computer", "laptop", "vase", "bottle",
]

# Tolerance constants
FLOOR_TOL   = 0.15  # m — acceptable float above computed floor_z
CEILING_TOL = 0.15  # m — acceptable gap below ceiling for ceiling-class objs
WALL_TOL    = 0.20  # m — acceptable distance from nearest wall
Z_CONSIST_TOL    = 0.15  # m — z consistency within mesh_group
ROT_CONSIST_TOL  = math.pi / 4   # 45° — rotation consistency threshold
ROT_EXCEPTION    = math.pi / 2   # 90° — deliberately opposite chairs
ROT_EXCEPTION_EPS = math.pi / 12 # 15° — margin around 90° exception
SURFACE_GAP_TOL  = 0.10  # m — max allowed gap between small obj bottom and surface top

# Wall expected rotations (rz): which direction the object "faces" toward room centre
#   north wall (high Y): object faces south → rz = -π/2
#   south wall (low Y): object faces north → rz = +π/2
#   east wall (high X): object faces west → rz = π
#   west wall (low X): object faces east → rz = 0
WALL_FACE_RZ = {
    "north": -math.pi / 2,
    "south":  math.pi / 2,
    "east":   math.pi,
    "west":   0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_anchor(class_label: str) -> str:
    """Return 'wall' | 'ceiling' | 'floor' | 'on_surface' | 'unknown'."""
    cl = class_label.lower()
    for kw in WALL_CLASSES:
        if kw in cl:
            return "wall"
    for kw in CEILING_CLASSES:
        if kw in cl:
            return "ceiling"
    for kw in FLOOR_CLASSES:
        if kw in cl:
            return "floor"
    for kw in ON_SURFACE_CANDIDATE:
        if kw in cl:
            return "on_surface"
    return "unknown"


def _is_on_surface_class(class_label: str) -> bool:
    cl = class_label.lower()
    return any(kw in cl for kw in ON_SURFACE_CANDIDATE)


def _is_floor_surface(class_label: str) -> bool:
    """True if the object is a large surface (table/shelf/counter/desk) things sit ON."""
    cl = class_label.lower()
    surface_kws = ["table", "counter", "desk", "shelf", "bookshelf"]
    return any(kw in cl for kw in surface_kws)


def _normalize_angle(a: float) -> float:
    """Bring angle into [0, 2π)."""
    return a % (2 * math.pi)


def _delta_rz(rz_a: float, rz_b: float) -> float:
    """Smallest angular distance between two rz values, in [0, π]."""
    diff = abs(_normalize_angle(rz_a) - _normalize_angle(rz_b))
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return diff


def _is_90deg_exception(delta: float) -> bool:
    """True if delta is close to π/2 (the 'opposite chairs' exception)."""
    return abs(delta - ROT_EXCEPTION) <= ROT_EXCEPTION_EPS


def _resolve_iter(session_dir: Path, iter_n: int | None) -> int:
    if iter_n is not None:
        return iter_n
    current_link = session_dir / "current"
    if not (current_link.exists() or current_link.is_symlink()):
        raise FileNotFoundError(f"No 'current' symlink in {session_dir}")
    target = os.readlink(str(current_link))
    m = re.fullmatch(r"iter_(\d+)", Path(target).name)
    if not m:
        raise ValueError(f"'current' points to unexpected: {target!r}")
    return int(m.group(1))


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(session_dir: Path, iter_n: int) -> dict:
    iter_dir = session_dir / f"iter_{iter_n}"

    # --- Load inputs --------------------------------------------------------
    classes_path    = session_dir / "object_class.json"
    mask_attr_path  = session_dir / "mask_attribute.json"
    list_obj_path   = iter_dir / "list_objects.json"
    metrics_path    = iter_dir / "metrics.json"

    for p in (classes_path, list_obj_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    object_classes: dict[str, str] = _load_json(classes_path)  # {"obj_N": "<class>"}

    list_obj_raw = _load_json(list_obj_path)
    objects_list: list[dict] = list_obj_raw.get("objects", list_obj_raw if isinstance(list_obj_raw, list) else [])

    metrics: dict = {}
    if metrics_path.exists():
        metrics = _load_json(metrics_path)

    mask_attr: dict = {}
    if mask_attr_path.exists():
        mask_attr = _load_json(mask_attr_path)

    mesh_groups: dict[str, dict] = mask_attr.get("mesh_groups", {})

    # Build keyed dict of objects
    objs: dict[str, dict] = {o["name"]: o for o in objects_list}

    # --- Derived geometry ---------------------------------------------------
    # Room bbox: use metrics.json room_bbox if present; fall back to object span.
    room_bbox = metrics.get("room_bbox")
    if room_bbox and len(room_bbox) == 2:
        floor_room_z  = room_bbox[0][2]
        ceiling_room_z = room_bbox[1][2]
        x_min_room = room_bbox[0][0]
        x_max_room = room_bbox[1][0]
        y_min_room = room_bbox[0][1]
        y_max_room = room_bbox[1][1]
    else:
        # Fall back: derive from object locations
        all_locs = [o["location"] for o in objs.values()]
        x_min_room = min(l[0] for l in all_locs)
        x_max_room = max(l[0] for l in all_locs)
        y_min_room = min(l[1] for l in all_locs)
        y_max_room = max(l[1] for l in all_locs)
        floor_room_z  = min(l[2] for l in all_locs) - 0.5
        ceiling_room_z = max(l[2] for l in all_locs) + 0.5

    # Compute floor_z and ceiling_z from floor/ceiling-class objects
    # since all objects are EMPTY parents with dimensions=[0,0,0], we use
    # location[2] as the canonical z reference for each object.
    floor_zs   = [objs[n]["location"][2] for n, cl in object_classes.items()
                  if n in objs and _match_anchor(cl) == "floor"]
    ceiling_zs = [objs[n]["location"][2] for n, cl in object_classes.items()
                  if n in objs and _match_anchor(cl) == "ceiling"]

    # Floor reference: minimum z among floor-class objects; fall back to room bbox floor
    floor_ref_z    = min(floor_zs)    if floor_zs    else floor_room_z
    ceiling_ref_z  = max(ceiling_zs)  if ceiling_zs  else ceiling_room_z

    # --- Type 1: Anchor violations -----------------------------------------
    type1: list[dict] = []

    for obj_name, cl in object_classes.items():
        if obj_name not in objs:
            continue
        obj = objs[obj_name]
        anchor = _match_anchor(cl)
        loc = obj["location"]
        rot = obj["rotation_euler"]
        z = loc[2]

        if anchor == "floor":
            # Object z should be close to floor_ref_z
            deviation = z - floor_ref_z
            if deviation > FLOOR_TOL:
                issue = f"floating {deviation:.2f}m above floor reference (z={floor_ref_z:.2f})"
                type1.append({
                    "obj_name": obj_name,
                    "class": cl,
                    "expected_anchor": "floor",
                    "issue": issue,
                    "current": {"location": loc, "rotation_euler": rot},
                    "suggested_op": {
                        "action": "update_layout",
                        "obj_name": obj_name,
                        "location": [loc[0], loc[1], floor_ref_z],
                    },
                })
            elif deviation < -FLOOR_TOL:
                issue = f"sunk {-deviation:.2f}m below floor reference (z={floor_ref_z:.2f})"
                type1.append({
                    "obj_name": obj_name,
                    "class": cl,
                    "expected_anchor": "floor",
                    "issue": issue,
                    "current": {"location": loc, "rotation_euler": rot},
                    "suggested_op": {
                        "action": "update_layout",
                        "obj_name": obj_name,
                        "location": [loc[0], loc[1], floor_ref_z],
                    },
                })

        elif anchor == "ceiling":
            # Object z should be close to ceiling_ref_z
            deviation = ceiling_ref_z - z
            if deviation > CEILING_TOL:
                issue = f"{deviation:.2f}m below ceiling reference (z={ceiling_ref_z:.2f})"
                type1.append({
                    "obj_name": obj_name,
                    "class": cl,
                    "expected_anchor": "ceiling",
                    "issue": issue,
                    "current": {"location": loc, "rotation_euler": rot},
                    "suggested_op": {
                        "action": "update_layout",
                        "obj_name": obj_name,
                        "location": [loc[0], loc[1], ceiling_ref_z],
                    },
                })

        elif anchor == "wall":
            # Find nearest wall and check distance
            x, y = loc[0], loc[1]
            dist_west  = abs(x - x_min_room)
            dist_east  = abs(x - x_max_room)
            dist_south = abs(y - y_min_room)
            dist_north = abs(y - y_max_room)

            nearest_dist = min(dist_west, dist_east, dist_south, dist_north)
            # inset away from the exact wall plane so the snapped object
            # doesn't trigger OOB (mesh AABB extends past origin even for
            # zero-dim Empty parents because children carry the geometry).
            WALL_INSET = 0.10
            if dist_west == nearest_dist:
                wall_name = "west"
                snap_loc  = [x_min_room + WALL_INSET, y, z]
            elif dist_east == nearest_dist:
                wall_name = "east"
                snap_loc  = [x_max_room - WALL_INSET, y, z]
            elif dist_south == nearest_dist:
                wall_name = "south"
                snap_loc  = [x, y_min_room + WALL_INSET, z]
            else:
                wall_name = "north"
                snap_loc  = [x, y_max_room - WALL_INSET, z]

            expected_rz = WALL_FACE_RZ[wall_name]

            if nearest_dist > WALL_TOL:
                issue = (
                    f"{nearest_dist:.2f}m from nearest wall ({wall_name}); "
                    f"should be ≤{WALL_TOL:.2f}m"
                )
                type1.append({
                    "obj_name": obj_name,
                    "class": cl,
                    "expected_anchor": "wall",
                    "issue": issue,
                    "current": {"location": loc, "rotation_euler": rot},
                    "suggested_op": {
                        "action": "update_layout",
                        "obj_name": obj_name,
                        "location": snap_loc,
                    },
                    "secondary_op_for_walls": {
                        "action": "update_rotation",
                        "obj_name": obj_name,
                        "rotation_euler": [0.0, 0.0, expected_rz],
                    },
                })

    # Sort Type 1 by distance/deviation magnitude (largest first)
    type1.sort(key=lambda e: _t1_deviation(e, objs, floor_ref_z, ceiling_ref_z,
                                            x_min_room, x_max_room, y_min_room, y_max_room),
               reverse=True)

    # --- Type 2: Class rotation/z consistency ------------------------------
    type2: list[dict] = []

    for group_name, group in mesh_groups.items():
        canonical_id = group.get("canonical_id")
        instance_ids = group.get("instance_ids", [])
        group_class  = group.get("class", "")
        canonical_name = f"obj_{canonical_id}"
        if canonical_name not in objs:
            continue
        canonical_obj = objs[canonical_name]
        canon_rz = canonical_obj["rotation_euler"][2]
        canon_z  = canonical_obj["location"][2]
        canon_loc = canonical_obj["location"]

        # Find the nearest table-class obj for chair face-center calculation
        table_centers = [
            objs[n]["location"]
            for n, cl in object_classes.items()
            if n in objs and _is_floor_surface(cl)
        ]

        for inst_id in instance_ids:
            inst_name = f"obj_{inst_id}"
            if inst_name == canonical_name or inst_name not in objs:
                continue
            inst_obj = objs[inst_name]
            inst_rz  = inst_obj["rotation_euler"][2]
            inst_z   = inst_obj["location"][2]
            inst_loc = inst_obj["location"]

            # --- Rotation check ---
            delta_rz = _delta_rz(inst_rz, canon_rz)
            rot_issue = (delta_rz > ROT_CONSIST_TOL and not _is_90deg_exception(delta_rz))

            # For chairs: compute expected face-center rotation if near a table
            face_center_rz = None
            if "chair" in group_class.lower() and table_centers:
                nearest_table = min(
                    table_centers,
                    key=lambda t: math.hypot(t[0] - inst_loc[0], t[1] - inst_loc[1])
                )
                face_center_rz = math.atan2(
                    nearest_table[1] - inst_loc[1],
                    nearest_table[0] - inst_loc[0],
                )
                # Re-evaluate: if instance is close to face-center expected, it's not an issue
                if face_center_rz is not None:
                    delta_vs_fc = _delta_rz(inst_rz, face_center_rz)
                    if delta_vs_fc <= ROT_CONSIST_TOL:
                        rot_issue = False  # chair is facing center — that's correct

            if rot_issue:
                suggested_rz = face_center_rz if face_center_rz is not None else canon_rz
                type2.append({
                    "mesh_group": group_name,
                    "class": group_class,
                    "canonical_obj": canonical_name,
                    "outlier_obj": inst_name,
                    "field": "rotation_z",
                    "delta": round(delta_rz, 4),
                    "rationale": (
                        f"rz={inst_rz:.3f} differs from canonical ({canon_rz:.3f}) "
                        f"by {math.degrees(delta_rz):.1f}° and is not a ±90° exception"
                    ),
                    "suggested_op": {
                        "action": "update_rotation",
                        "obj_name": inst_name,
                        "rotation_euler": [
                            inst_obj["rotation_euler"][0],
                            inst_obj["rotation_euler"][1],
                            round(suggested_rz, 5),
                        ],
                    },
                })

            # --- Z check ---
            dz = abs(inst_z - canon_z)
            if dz > Z_CONSIST_TOL:
                type2.append({
                    "mesh_group": group_name,
                    "class": group_class,
                    "canonical_obj": canonical_name,
                    "outlier_obj": inst_name,
                    "field": "z",
                    "delta": round(dz, 4),
                    "rationale": (
                        f"z={inst_z:.3f} vs canonical z={canon_z:.3f}, "
                        f"Δ={dz:.3f}m > {Z_CONSIST_TOL}m threshold"
                    ),
                    "suggested_op": {
                        "action": "update_layout",
                        "obj_name": inst_name,
                        "location": [inst_loc[0], inst_loc[1], canon_z],
                    },
                })

    # Sort Type 2: prefer rotation issues; within same type by largest delta
    type2.sort(key=lambda e: (e["field"] != "rotation_z", -e["delta"]))

    # --- Type 3: On-surface pairs ------------------------------------------
    type3: list[dict] = []

    # Build list of (small, large) candidates
    # small: ON_SURFACE_CANDIDATE class or dimensions < 0.4m all axes
    # large: FLOOR_CLASSES that are also surface types (table/shelf/desk/counter)
    small_objs = [
        (n, cl, objs[n])
        for n, cl in object_classes.items()
        if n in objs and (_is_on_surface_class(cl))
    ]
    large_surface_objs = [
        (n, cl, objs[n])
        for n, cl in object_classes.items()
        if n in objs and _is_floor_surface(cl)
    ]

    if small_objs and large_surface_objs:
        # Use location as proxy bbox center; for overlap use a small fixed half-extent
        # since dimensions are 0.  We'll use xy proximity instead: if within 1.5m xy.
        for s_name, s_cl, s_obj in small_objs:
            s_loc = s_obj["location"]
            for l_name, l_cl, l_obj in large_surface_objs:
                if s_name == l_name:
                    continue
                l_loc = l_obj["location"]
                xy_dist = math.hypot(s_loc[0] - l_loc[0], s_loc[1] - l_loc[1])
                # Only consider pairs that are xy-close (within 1.5m)
                if xy_dist > 1.5:
                    continue
                # Since dimensions are 0, we estimate surface top z using a prior:
                # typical table/counter top ≈ object z + 0.35 m (half table height ~0.7m)
                # We check if the small object is close in z to that expected surface top.
                # Surface top estimate: l_loc[2] + 0.35 (half of ~0.70m table height)
                surface_top_z_est = l_loc[2] + 0.35
                gap = s_loc[2] - surface_top_z_est
                if abs(gap) > SURFACE_GAP_TOL:
                    type3.append({
                        "small_obj": s_name,
                        "small_class": s_cl,
                        "surface_obj": l_name,
                        "surface_class": l_cl,
                        "xy_dist_m": round(xy_dist, 3),
                        "estimated_surface_top_z": round(surface_top_z_est, 3),
                        "small_z": round(s_loc[2], 3),
                        "gap_m": round(gap, 3),
                        "issue": (
                            f"{s_name} ({s_cl}) is {gap:+.2f}m "
                            f"relative to estimated surface top of {l_name} ({l_cl})"
                        ),
                        "suggested_op": {
                            "action": "update_layout",
                            "obj_name": s_name,
                            "location": [s_loc[0], s_loc[1], surface_top_z_est],
                        },
                    })

    # Sort by largest |gap|
    type3.sort(key=lambda e: abs(e["gap_m"]), reverse=True)

    # --- Type 4: Residual collisions ----------------------------------------
    # Objects already flagged in Types 1-3 will likely be fixed; exclude their pairs
    flagged_objs: set[str] = set()
    for e in type1:
        flagged_objs.add(e["obj_name"])
    for e in type2:
        flagged_objs.add(e["outlier_obj"])
    for e in type3:
        flagged_objs.add(e["small_obj"])

    raw_collisions = metrics.get("collisions", [])
    type4 = [
        c for c in raw_collisions
        if c["a"] not in flagged_objs and c["b"] not in flagged_objs
    ]
    # Sort by volume descending (already sorted in metrics, but re-sort to be safe)
    type4.sort(key=lambda c: -c["volume_m3"])

    # --- Priority action ---------------------------------------------------
    priority_action = None

    def _pick_priority(violations: list[dict], typ: int) -> dict | None:
        if not violations:
            return None
        v = violations[0]
        if typ == 1:
            return {
                "type": 1,
                "obj_name": v["obj_name"],
                "rationale": v["issue"],
                "suggested_op": v["suggested_op"],
            }
        elif typ == 2:
            return {
                "type": 2,
                "obj_name": v["outlier_obj"],
                "rationale": v["rationale"],
                "suggested_op": v["suggested_op"],
            }
        elif typ == 3:
            return {
                "type": 3,
                "obj_name": v["small_obj"],
                "rationale": v["issue"],
                "suggested_op": v["suggested_op"],
            }
        elif typ == 4:
            return {
                "type": 4,
                "obj_name": v["a"],
                "rationale": f"collision with {v['b']}, volume={v['volume_m3']}m³",
                "suggested_op": {
                    "action": "update_layout",
                    "obj_name": v["a"],
                    "location": None,  # agent must compute translation direction
                },
            }
        return None

    for typ, violations in [(1, type1), (2, type2), (3, type3), (4, type4)]:
        pa = _pick_priority(violations, typ)
        if pa is not None:
            priority_action = pa
            break

    result = {
        "iter": iter_n,
        "type1_anchor_violations": type1,
        "type2_class_inconsistency": type2,
        "type3_on_surface_pairs": type3,
        "type4_residual_collisions": type4,
        "priority_action": priority_action,
    }

    return result


def _t1_deviation(entry: dict, objs: dict, floor_ref_z: float, ceiling_ref_z: float,
                  x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """Return a numeric deviation for sorting Type 1 entries (largest = most urgent)."""
    n = entry["obj_name"]
    if n not in objs:
        return 0.0
    loc = objs[n]["location"]
    anchor = entry["expected_anchor"]
    if anchor == "floor":
        return abs(loc[2] - floor_ref_z)
    elif anchor == "ceiling":
        return abs(loc[2] - ceiling_ref_z)
    elif anchor == "wall":
        x, y = loc[0], loc[1]
        return min(abs(x - x_min), abs(x - x_max), abs(y - y_min), abs(y - y_max))
    return 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _selftest():
    """Verify imports, check that WALL_CLASSES etc are defined, exit 0."""
    errors = []
    for name, lst in [
        ("WALL_CLASSES", WALL_CLASSES),
        ("CEILING_CLASSES", CEILING_CLASSES),
        ("FLOOR_CLASSES", FLOOR_CLASSES),
        ("ON_SURFACE_CANDIDATE", ON_SURFACE_CANDIDATE),
    ]:
        if not isinstance(lst, list) or not lst:
            errors.append(f"{name} is empty or not a list")
    # Check helper functions
    assert _match_anchor("board") == "wall",    "board should match wall"
    assert _match_anchor("chair") == "floor",   "chair should match floor"
    assert _match_anchor("fluorescent light") == "ceiling", "fl light should match ceiling"
    assert _match_anchor("book") == "on_surface", "book should match on_surface"
    assert _match_anchor("xenomorph") == "unknown", "unknown class should be unknown"
    assert abs(_delta_rz(0.0, math.pi * 2)) < 1e-9, "full circle = 0"
    assert abs(_delta_rz(0.0, math.pi)) - math.pi < 1e-9, "half circle = pi"
    assert _is_90deg_exception(math.pi / 2), "pi/2 is the 90-deg exception"
    assert not _is_90deg_exception(math.pi / 4), "pi/4 is NOT the 90-deg exception"
    if errors:
        for e in errors:
            print(f"SELFTEST FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    scripts_dir = Path(__file__).parent
    print("selftest OK")
    print(f"  WALL_CLASSES     = {WALL_CLASSES[:3]}...")
    print(f"  scripts_dir      = {scripts_dir}")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="4-type scene analysis for SceneWeaver.")
    parser.add_argument("session_dir", nargs="?", help="path to session directory")
    parser.add_argument("--iter", type=int, default=None, help="iter number (default: from current symlink)")
    parser.add_argument("--selftest", action="store_true", help="run selftest and exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()

    if not args.session_dir:
        parser.print_help(sys.stderr)
        sys.exit(1)

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.is_dir():
        print(f"ERROR: session_dir is not a directory: {session_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        iter_n = _resolve_iter(session_dir, args.iter)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        result = analyze(session_dir, iter_n)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Write output
    out_path = session_dir / f"iter_{iter_n}" / "analysis.json"
    out_path.write_text(json.dumps(result, indent=2))

    # One-line summary to stdout
    t1 = len(result["type1_anchor_violations"])
    t2 = len(result["type2_class_inconsistency"])
    t3 = len(result["type3_on_surface_pairs"])
    t4 = len(result["type4_residual_collisions"])
    pa = result["priority_action"]
    pa_str = f"T{pa['type']}:{pa['obj_name']}" if pa else "none"
    print(
        f"[scene_analysis] iter={iter_n} "
        f"T1={t1} T2={t2} T3={t3} T4={t4} "
        f"priority={pa_str} → {out_path}"
    )


if __name__ == "__main__":
    main()
