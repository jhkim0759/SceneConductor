#!/usr/bin/env python3
"""compute_member_geometry.py — Deterministic per-member geometry computation.

Forward axes are determined per-asset: if same-mesh siblings have a consistent
cardinal axis, ambiguous instances inherit it (handles baked-rotation noise in
identical chair models).

Produces two outputs:
  1. <group_dir>/forward_axes.json      — cached; written ONLY on the first call
     (when the file is absent). Contains the local-frame forward axis for each
     member derived from mesh vertex distribution (upper-half centroid vs full
     centroid).  No rendering needed; runs once per dispatch.

  2. <group_dir>/simple_refiner/iter_N/geometry.json  — per-iter; always written.
     Combines the cached forward axes with the live absolute poses from
     iter_N/island.blend to produce:
       - current_facing_world_xy  (rotate local forward by current yaw)
       - ideal_facing_world_xy    (unit vector from member toward anchor centroid)
       - facing_alignment_dot, facing_status
       - current_clearance_m_to_anchor, clearance_excess_m, clearance_status
       - per-island summary

CLI:
    python compute_member_geometry.py \\
        --group-dir <DIR> \\
        --iter N \\
        [--blender-bin PATH]

Performance target: < 5 s per call (single Blender headless invocation).

All stdlib + bpy (Blender headless). No new pip dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum confidence to snap to a cardinal axis ({+X,-X,+Y,-Y}).
# Below this the axis is reported as "ambiguous" (may be overridden later by
# same-asset inference).  0.55 instead of 0.60 to capture borderline cases
# where one axis dominates by ~40% (e.g. |y|=0.020, |x|=0.014 → confidence 0.588).
CARDINAL_CONFIDENCE_THRESHOLD: float = 0.55

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_SKILL_SRC = _THIS_FILE.parent
# parents: [0]=src [1]=stage3-sub-island-refiner [2]=skills [3]=.claude [4]=repo-root
_REPO_ROOT = _THIS_FILE.parents[4]
_DIRS_YAML = _REPO_ROOT / "DIRECTORYS.yaml"

# ---------------------------------------------------------------------------
# Blender-bin resolver (same priority order as render_island.py)
# ---------------------------------------------------------------------------

def _resolve_blender(cli_arg: str | None) -> str:
    if cli_arg:
        candidate = Path(cli_arg)
        if not candidate.is_absolute():
            candidate = (_REPO_ROOT / candidate).resolve()
        return str(candidate)

    if _DIRS_YAML.is_file():
        try:
            import yaml  # type: ignore
            d = yaml.safe_load(_DIRS_YAML.read_text()) or {}
            raw = (d.get("blender_bin_linux")
                   or d.get("blender_bin"))
            if raw:
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = (_REPO_ROOT / candidate).resolve()
                return str(candidate)
        except Exception:
            pass
        # naive line scan fallback
        for line in _DIRS_YAML.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("blender_bin:") and not stripped.startswith("blender_bin_"):
                val = stripped.split(":", 1)[1].strip()
                if val:
                    return val

    env_val = os.environ.get("BLENDER", "").strip()
    if env_val:
        return env_val

    return "blender"

# ---------------------------------------------------------------------------
# Blender inner script (runs inside Blender via -P)
# ---------------------------------------------------------------------------

# This script is written to a temp file and executed by Blender headless.
# It writes a JSON to a path supplied via environment variable GEOM_OUT_JSON.
_BLENDER_INNER = r"""
import bpy, json, math, os, sys

# Blender args: everything after '--' in the command line.
argv = sys.argv
try:
    sep = argv.index("--")
    extra = argv[sep + 1:]
except ValueError:
    extra = []

out_path = os.environ.get("GEOM_OUT_JSON", "")
if not out_path:
    for i, a in enumerate(extra):
        if a == "--out" and i + 1 < len(extra):
            out_path = extra[i + 1]
            break

if not out_path:
    print("BLENDER_INNER ERROR: no output path", file=sys.stderr)
    sys.exit(1)

_CARDINAL_THRESHOLD = float(os.environ.get("GEOM_CARDINAL_THRESHOLD", "0.55"))

# GEOM_ANCHOR_ID lets stage-geometry anchors (Floor / Wall_NN / Ceiling) be
# included in the result alongside the obj_* members. Without this, the
# obj_*-only filter below silently drops MESH anchors used by synthetic islands.
_ANCHOR_ID = os.environ.get("GEOM_ANCHOR_ID", "")

def bbox_and_verts(obj, use_world=False):
    # Return (verts, min_xyz, max_xyz) for obj, including all mesh children.
    # use_world=False (default): verts are in the ROOT obj's LOCAL frame
    #   (parent transform of obj is NOT applied — stable across iters; used for
    #   forward-axis detection).
    # use_world=True: verts are in WORLD frame (used for AABB clearance).
    verts = []

    if use_world:
        # World frame: apply matrix_world for each mesh node.
        def _collect_world(o):
            if o.type == "MESH" and o.data:
                mat = o.matrix_world
                for v in o.data.vertices:
                    co = mat @ v.co
                    verts.append((co.x, co.y, co.z))
            for child in o.children:
                _collect_world(child)
        _collect_world(obj)
    else:
        # Root-local frame: start with the inverse of obj's world transform
        # so the root obj sits at identity. Children's relative offsets to obj
        # are preserved.
        import mathutils
        try:
            root_inv = obj.matrix_world.inverted()
        except Exception:
            root_inv = mathutils.Matrix.Identity(4)

        def _collect_local(o):
            if o.type == "MESH" and o.data:
                mat = root_inv @ o.matrix_world
                for v in o.data.vertices:
                    co = mat @ v.co
                    verts.append((co.x, co.y, co.z))
            for child in o.children:
                _collect_local(child)
        _collect_local(obj)

    if not verts:
        return [], (0, 0, 0), (0, 0, 0)

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return verts, (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def forward_axis_from_verts(verts, mn, mx):
    # Compute local forward axis via upper-half centroid vs full centroid.
    # Returns dict with keys: axis, confidence, backrest_offset_xy_local,
    # forward_local_xy, method.
    if not verts:
        return {"axis": "ambiguous", "confidence": 0.0,
                "backrest_offset_xy_local": [0.0, 0.0],
                "forward_local_xy": [0.0, 0.0],
                "method": "upper_half_centroid_vs_full_centroid"}

    mid_z = (mn[2] + mx[2]) / 2.0
    upper = [v for v in verts if v[2] > mid_z]
    if not upper:
        return {"axis": "ambiguous", "confidence": 0.0,
                "backrest_offset_xy_local": [0.0, 0.0],
                "forward_local_xy": [0.0, 0.0],
                "method": "upper_half_centroid_vs_full_centroid"}

    full_cx = sum(v[0] for v in verts) / len(verts)
    full_cy = sum(v[1] for v in verts) / len(verts)
    upper_cx = sum(v[0] for v in upper) / len(upper)
    upper_cy = sum(v[1] for v in upper) / len(upper)

    # Backrest is in the upper half; offset points toward backrest.
    bx = upper_cx - full_cx
    by = upper_cy - full_cy

    # Chair forward is OPPOSITE of backrest direction.
    fx = -bx
    fy = -by

    mag = math.sqrt(fx * fx + fy * fy)
    if mag < 1e-6:
        return {"axis": "ambiguous", "confidence": 0.0,
                "backrest_offset_xy_local": [round(bx, 6), round(by, 6)],
                "forward_local_xy": [0.0, 0.0],
                "method": "upper_half_centroid_vs_full_centroid"}

    # Snap to nearest cardinal axis.
    abs_x = abs(fx)
    abs_y = abs(fy)
    total = abs_x + abs_y
    confidence = max(abs_x, abs_y) / total if total > 1e-9 else 0.5

    if abs_x >= abs_y:
        axis = "+X" if fx > 0 else "-X"
    else:
        axis = "+Y" if fy > 0 else "-Y"

    if confidence < _CARDINAL_THRESHOLD:
        axis = "ambiguous"

    return {
        "axis": axis,
        "confidence": round(confidence, 4),
        "backrest_offset_xy_local": [round(bx, 6), round(by, 6)],
        "forward_local_xy": [round(fx / mag, 6), round(fy / mag, 6)],
        "method": "upper_half_centroid_vs_full_centroid",
    }


result = {}

for obj in bpy.data.objects:
    is_obj_empty = (obj.type == "EMPTY" and obj.name.startswith("obj_") and obj.parent is None)
    is_named_anchor = (_ANCHOR_ID and obj.name == _ANCHOR_ID)
    if not (is_obj_empty or is_named_anchor):
        continue

    name = obj.name
    # World-space position and yaw.
    pos = obj.location
    yaw_rad = obj.rotation_euler.z

    # Pass A — root-local vertices for forward-axis detection (stable, scale-free).
    verts_local, mn_l, mx_l = bbox_and_verts(obj, use_world=False)
    half_x = (mx_l[0] - mn_l[0]) / 2.0
    half_y = (mx_l[1] - mn_l[1]) / 2.0
    half_z = (mx_l[2] - mn_l[2]) / 2.0
    fwd_info = forward_axis_from_verts(verts_local, mn_l, mx_l)

    # Pass B — world-frame vertices for the true world AABB (includes scale).
    verts_world, mn_w, mx_w = bbox_and_verts(obj, use_world=True)
    if verts_world:
        aabb_min = [round(mn_w[i], 6) for i in range(3)]
        aabb_max = [round(mx_w[i], 6) for i in range(3)]
    else:
        # No mesh children — degenerate to position-only point.
        aabb_min = [float(pos.x), float(pos.y), float(pos.z)]
        aabb_max = [float(pos.x), float(pos.y), float(pos.z)]

    # Mesh signature: sorted, comma-joined names of direct mesh-child datablocks.
    mesh_child_names = sorted(
        c.data.name
        for c in obj.children
        if c.type == "MESH" and c.data is not None
    )
    mesh_sig = ",".join(mesh_child_names)

    cx, cy, cz = float(pos.x), float(pos.y), float(pos.z)

    result[name] = {
        "position": [round(cx, 6), round(cy, 6), round(cz, 6)],
        "yaw_deg": round(math.degrees(yaw_rad), 4),
        "bbox_local_half_extents": [round(half_x, 6), round(half_y, 6), round(half_z, 6)],
        "bbox_world_aabb_min": [round(v, 6) for v in aabb_min],
        "bbox_world_aabb_max": [round(v, 6) for v in aabb_max],
        "forward_raw": fwd_info,
        "mesh_signature": mesh_sig,
    }

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"GEOM_INNER_OK: wrote {len(result)} objects to {out_path}")
"""

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _unit_xy(vx: float, vy: float) -> tuple[float, float]:
    mag = math.sqrt(vx * vx + vy * vy)
    if mag < 1e-9:
        return (0.0, 0.0)
    return (vx / mag, vy / mag)


def _dot2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _rotate_xy(vx: float, vy: float, yaw_deg: float) -> tuple[float, float]:
    """Rotate a 2-D vector by yaw_deg degrees (counter-clockwise, Blender convention)."""
    rad = math.radians(yaw_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return (cos_r * vx - sin_r * vy, sin_r * vx + cos_r * vy)


_AXIS_LOCAL: dict[str, tuple[float, float]] = {
    "+X": (1.0, 0.0),
    "-X": (-1.0, 0.0),
    "+Y": (0.0, 1.0),
    "-Y": (0.0, -1.0),
    "ambiguous": (0.0, 0.0),  # handled specially
}

# ---------------------------------------------------------------------------
# Clearance computation
# ---------------------------------------------------------------------------

def _compute_clearance(
    member_pos: list[float],
    member_half: list[float],
    anchor_pos: list[float],
    anchor_half: list[float],
    member_yaw_deg: float,
    anchor_yaw_deg: float,
) -> float:
    """Compute true 2D AABB-to-AABB clearance in metres.

    Positive = surface-to-surface gap.
    Negative = penetration depth (axis with smallest |overlap| — the depth needed
    to push apart along that axis).

    NOTE on inputs: `bbox_local_half_extents` is misnamed in the inner script — it
    is actually the WORLD AABB half-extents (mesh vertices are extracted via
    `obj.matrix_local @ v.co`, which is world-coord for root objects). So we do
    NOT rotate by yaw here; the half-extents already reflect the current rotation.
    """
    m_hx, m_hy = member_half[0], member_half[1]
    a_hx, a_hy = anchor_half[0], anchor_half[1]

    # Per-axis gap = |centre_distance| − (h_A + h_B). Positive = separated; negative = overlap.
    gap_x = abs(member_pos[0] - anchor_pos[0]) - (m_hx + a_hx)
    gap_y = abs(member_pos[1] - anchor_pos[1]) - (m_hy + a_hy)

    if gap_x > 0 and gap_y > 0:
        # Separated on both axes — Euclidean corner-to-corner distance.
        return math.sqrt(gap_x * gap_x + gap_y * gap_y)
    if gap_x > 0:
        return gap_x  # separated only along X
    if gap_y > 0:
        return gap_y  # separated only along Y
    # Overlap on both axes — penetration depth is the smaller |overlap|
    # (axis along which the boxes are easiest to separate).
    return max(gap_x, gap_y)


# ---------------------------------------------------------------------------
# Status classifiers
# ---------------------------------------------------------------------------

def _facing_status(dot: float) -> str:
    if dot >= 0.95:
        return "OK"
    if dot >= 0.0:
        return "DRIFT"
    if dot >= -0.5:
        return "OFF_90"
    return "OFF_180"


def _clearance_status(excess: float, target: float) -> str:
    half = target * 0.5
    if abs(excess) <= half:
        return "OK_within_tolerance"
    if excess > half:
        return "TOO_FAR"
    if excess > -target:
        return "TOO_CLOSE"
    return "PENETRATING"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _run_blender_inner(blend_path: Path, blender_bin: str, anchor_id: str = "") -> dict:
    """Run the Blender inner script on blend_path, return parsed dict.

    anchor_id is propagated via GEOM_ANCHOR_ID so the inner script can include
    stage-geometry anchors (Floor / Wall_NN / Ceiling) alongside obj_* members.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(_BLENDER_INNER)
        inner_script = tf.name

    out_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    out_tmp.close()

    env = dict(os.environ)
    env["GEOM_OUT_JSON"] = out_tmp.name
    env["GEOM_CARDINAL_THRESHOLD"] = str(CARDINAL_CONFIDENCE_THRESHOLD)
    if anchor_id:
        env["GEOM_ANCHOR_ID"] = anchor_id

    try:
        cmd = [
            blender_bin,
            "-b", str(blend_path),
            "-P", inner_script,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Check for our success marker in stdout.
        if "GEOM_INNER_OK" not in result.stdout:
            print(f"[compute_member_geometry] Blender inner script failed.",
                  file=sys.stderr)
            print(result.stdout[-2000:], file=sys.stderr)
            print(result.stderr[-1000:], file=sys.stderr)
            sys.exit(1)

        with open(out_tmp.name) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(inner_script)
        except OSError:
            pass
        try:
            os.unlink(out_tmp.name)
        except OSError:
            pass


def _compute_and_write(
    group_dir: Path,
    iter_n: int,
    blender_bin: str,
) -> None:
    """Main computation entry point."""
    metadata_path = group_dir / "metadata.json"
    target_spec_path = group_dir / "target_spec.json"
    forward_axes_path = group_dir / "forward_axes.json"

    iter_dir = group_dir / "simple_refiner" / f"iter_{iter_n}"
    blend_path = iter_dir / "island.blend"
    geom_out = iter_dir / "geometry.json"

    # --- Load inputs ---
    if not metadata_path.is_file():
        print(f"[compute_member_geometry] ERROR: missing {metadata_path}", file=sys.stderr)
        sys.exit(1)
    if not blend_path.is_file():
        print(f"[compute_member_geometry] ERROR: missing {blend_path}", file=sys.stderr)
        sys.exit(1)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    anchor_id: str = metadata["anchor_id"]

    target_clearance = 0.05  # fallback
    if target_spec_path.is_file():
        try:
            spec = json.loads(target_spec_path.read_text(encoding="utf-8"))
            target_clearance = float(spec.get("clearance_m", target_clearance))
        except Exception:
            pass

    # --- Run Blender headless to extract mesh data ---
    print(f"[compute_member_geometry] running Blender inner script on {blend_path} ...")
    raw_data = _run_blender_inner(blend_path, blender_bin, anchor_id=anchor_id)
    print(f"[compute_member_geometry] Blender inner returned {len(raw_data)} objects")

    # --- Same-asset inference: resolve ambiguous members from cardinal siblings ---
    # Group members by mesh_signature.
    sig_to_members: dict[str, list[str]] = {}
    for obj_name, obj_data in raw_data.items():
        sig = obj_data.get("mesh_signature", "")
        sig_to_members.setdefault(sig, []).append(obj_name)

    # For each ambiguous member, look at same-signature siblings.
    # If 2+ siblings have cardinal axes AND agree (strict majority), inherit it.
    # Tie or all-ambiguous siblings → do NOT inherit (leave as ambiguous).
    inferred_count = 0
    from collections import Counter
    for obj_name, obj_data in raw_data.items():
        fwd = obj_data["forward_raw"]
        if fwd["axis"] != "ambiguous":
            continue
        sig = obj_data.get("mesh_signature", "")
        siblings = [n for n in sig_to_members.get(sig, []) if n != obj_name]
        sibling_axes = [
            raw_data[n]["forward_raw"]["axis"]
            for n in siblings
            if raw_data[n]["forward_raw"]["axis"] != "ambiguous"
        ]
        if len(sibling_axes) < 2:
            # Edge case: fewer than 2 cardinal siblings → no reliable consensus.
            continue
        most_common = Counter(sibling_axes).most_common()
        # Require a strict majority (top count > second count, or only one distinct value).
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            winner_axis = most_common[0][0]
            fwd["axis"] = winner_axis
            fwd["confidence"] = round(CARDINAL_CONFIDENCE_THRESHOLD, 4)
            fwd["method"] = "same_asset_inference"
            fwd["inferred_from"] = [
                n for n in siblings
                if raw_data[n]["forward_raw"]["axis"] == winner_axis
            ]
            inferred_count += 1

    if inferred_count > 0:
        print(
            f"[compute_member_geometry] same-asset inference: "
            f"resolved {inferred_count} ambiguous members"
        )

    # --- Compute / load forward_axes.json ---
    need_forward_axes_write = not forward_axes_path.is_file()

    # If the cached file exists but we just resolved new inferences, overwrite it
    # so that the agent reads up-to-date axes (not stale ambiguous entries).
    if not need_forward_axes_write and inferred_count > 0:
        existing_fa = json.loads(forward_axes_path.read_text(encoding="utf-8"))
        existing_axes = existing_fa.get("forward_axes_local", {})
        stale_ambiguous = any(
            existing_axes.get(n, {}).get("axis") == "ambiguous"
            and raw_data[n]["forward_raw"]["axis"] != "ambiguous"
            for n in raw_data
        )
        if stale_ambiguous:
            need_forward_axes_write = True
            print(
                f"[compute_member_geometry] re-writing forward_axes.json "
                f"with {inferred_count} inferred axes (stale ambiguous entries detected)"
            )

    if need_forward_axes_write:
        # Compute from raw mesh data (post-inference).
        forward_axes_local: dict[str, dict] = {}
        for obj_name, obj_data in raw_data.items():
            fwd = obj_data["forward_raw"]
            entry: dict = {
                "axis": fwd["axis"],
                "confidence": fwd["confidence"],
                "backrest_offset_xy_local": fwd["backrest_offset_xy_local"],
                "method": fwd["method"],
            }
            if "inferred_from" in fwd:
                entry["inferred_from"] = fwd["inferred_from"]
            forward_axes_local[obj_name] = entry

        fa_doc = {
            "scene_dir": str(group_dir.parent.parent),  # scene_dir is 2 levels up
            "group_id": group_dir.name,
            "forward_axes_local": forward_axes_local,
        }
        forward_axes_path.write_text(json.dumps(fa_doc, indent=2), encoding="utf-8")
        print(f"[compute_member_geometry] wrote forward_axes.json ({len(forward_axes_local)} members)")
    else:
        fa_doc = json.loads(forward_axes_path.read_text(encoding="utf-8"))
        forward_axes_local = fa_doc.get("forward_axes_local", {})
        print(f"[compute_member_geometry] loaded cached forward_axes.json")

    # --- Build per-member geometry ---
    anchor_data = raw_data.get(anchor_id)
    if anchor_data is None:
        print(f"[compute_member_geometry] ERROR: anchor '{anchor_id}' not found in blend data",
              file=sys.stderr)
        sys.exit(1)

    anchor_pos = anchor_data["position"]
    anchor_half = anchor_data["bbox_local_half_extents"]
    anchor_yaw_deg = anchor_data["yaw_deg"]
    anchor_aabb_min = anchor_data["bbox_world_aabb_min"]
    anchor_aabb_max = anchor_data["bbox_world_aabb_max"]

    members_out: dict[str, dict] = {}
    n_facing_ok = 0
    n_clearance_ok = 0
    worst_facing_member = None
    worst_facing_dot = 2.0
    worst_clearance_member = None
    worst_clearance_excess = 0.0

    member_ids = [k for k in raw_data if k != anchor_id]

    for obj_name in member_ids:
        obj_data = raw_data[obj_name]
        pos = obj_data["position"]
        yaw_deg = obj_data["yaw_deg"]
        half = obj_data["bbox_local_half_extents"]
        aabb_min = obj_data["bbox_world_aabb_min"]
        aabb_max = obj_data["bbox_world_aabb_max"]

        # Forward axis (from cache or freshly computed).
        fa_entry = forward_axes_local.get(obj_name, {})
        asset_forward_local: str = fa_entry.get("axis", "ambiguous")

        # Current facing in world XY.
        if asset_forward_local in _AXIS_LOCAL and asset_forward_local != "ambiguous":
            lx, ly = _AXIS_LOCAL[asset_forward_local]
            cur_world_fwd = _rotate_xy(lx, ly, yaw_deg)
            cur_world_fwd = _unit_xy(*cur_world_fwd)
        else:
            # Ambiguous — use +Y local rotated as a best-effort guess.
            cur_world_fwd = _rotate_xy(0.0, 1.0, yaw_deg)
            cur_world_fwd = _unit_xy(*cur_world_fwd)

        # Ideal facing: from member toward anchor centroid.
        ideal_fwd = _unit_xy(anchor_pos[0] - pos[0], anchor_pos[1] - pos[1])

        # Facing alignment.
        if asset_forward_local == "ambiguous":
            # Ambiguous forward: we can't reliably compute dot; report neutral.
            alignment_dot = 0.0
            status_facing = "DRIFT"
        else:
            alignment_dot = round(_dot2(cur_world_fwd, ideal_fwd), 4)
            status_facing = _facing_status(alignment_dot)

        # Clearance — use world AABB half-extents directly (already axis-aligned
        # in world frame), bypassing yaw rotation in _compute_clearance.
        member_world_half = [
            (aabb_max[0] - aabb_min[0]) / 2.0,
            (aabb_max[1] - aabb_min[1]) / 2.0,
            (aabb_max[2] - aabb_min[2]) / 2.0,
        ]
        anchor_world_half = [
            (anchor_aabb_max[0] - anchor_aabb_min[0]) / 2.0,
            (anchor_aabb_max[1] - anchor_aabb_min[1]) / 2.0,
            (anchor_aabb_max[2] - anchor_aabb_min[2]) / 2.0,
        ]
        clearance_m = _compute_clearance(
            pos, member_world_half,
            anchor_pos, anchor_world_half,
            0.0, 0.0,  # yaw args unused after axis-aligned half-extents
        )
        clearance_excess = round(clearance_m - target_clearance, 4)
        clearance_st = _clearance_status(clearance_excess, target_clearance)

        # Summary tracking.
        if status_facing == "OK":
            n_facing_ok += 1
        if clearance_st == "OK_within_tolerance":
            n_clearance_ok += 1

        if alignment_dot < worst_facing_dot:
            worst_facing_dot = alignment_dot
            worst_facing_member = obj_name

        if abs(clearance_excess) > abs(worst_clearance_excess):
            worst_clearance_excess = clearance_excess
            worst_clearance_member = obj_name

        members_out[obj_name] = {
            "position": pos,
            "yaw_deg": yaw_deg,
            "bbox_local_half_extents": half,
            "asset_forward_local": asset_forward_local,
            "current_facing_world_xy": [round(cur_world_fwd[0], 6), round(cur_world_fwd[1], 6)],
            "ideal_facing_world_xy": [round(ideal_fwd[0], 6), round(ideal_fwd[1], 6)],
            "facing_alignment_dot": alignment_dot,
            "facing_status": status_facing,
            "current_clearance_m_to_anchor": round(clearance_m, 4),
            "clearance_excess_m": clearance_excess,
            "clearance_status": clearance_st,
        }

    n_members = len(member_ids)

    geom_doc = {
        "iter": iter_n,
        "frame": "canonical",
        "anchor": {
            "id": anchor_id,
            "position": anchor_pos,
            "yaw_deg": anchor_yaw_deg,
            "bbox_local_half_extents": anchor_half,
            "bbox_world_aabb_min": anchor_aabb_min,
            "bbox_world_aabb_max": anchor_aabb_max,
        },
        "target_clearance_m": target_clearance,
        "members": members_out,
        "summary": {
            "n_members": n_members,
            "n_facing_ok": n_facing_ok,
            "n_clearance_ok": n_clearance_ok,
            "worst_facing_member": worst_facing_member,
            "worst_clearance_member": worst_clearance_member,
        },
    }

    # ── Pairwise member-member spacing (world AABB-AABB distance). ────────
    member_pairs: dict[str, dict] = {}
    pair_min_gap = float("inf")
    pair_max_gap = float("-inf")
    pair_min_key = None
    pair_max_key = None
    sorted_ids = sorted(member_ids)
    for i in range(len(sorted_ids)):
        for j in range(i + 1, len(sorted_ids)):
            a_name = sorted_ids[i]
            b_name = sorted_ids[j]
            a = raw_data[a_name]
            b = raw_data[b_name]
            a_amin, a_amax = a["bbox_world_aabb_min"], a["bbox_world_aabb_max"]
            b_amin, b_amax = b["bbox_world_aabb_min"], b["bbox_world_aabb_max"]
            a_half = [(a_amax[k] - a_amin[k]) / 2.0 for k in range(3)]
            b_half = [(b_amax[k] - b_amin[k]) / 2.0 for k in range(3)]
            gap = _compute_clearance(
                a["position"], a_half,
                b["position"], b_half,
                0.0, 0.0,
            )
            cdist = math.sqrt(
                (a["position"][0] - b["position"][0]) ** 2
                + (a["position"][1] - b["position"][1]) ** 2
            )
            key = f"{a_name}__{b_name}"
            member_pairs[key] = {
                "gap_m": round(gap, 4),
                "center_distance_m": round(cdist, 4),
            }
            if gap < pair_min_gap:
                pair_min_gap = gap
                pair_min_key = key
            if gap > pair_max_gap:
                pair_max_gap = gap
                pair_max_key = key

    geom_doc["member_spacing"] = member_pairs
    geom_doc["member_spacing_summary"] = {
        "n_pairs": len(member_pairs),
        "min_gap_m": round(pair_min_gap, 4) if pair_min_key else None,
        "min_gap_pair": pair_min_key,
        "max_gap_m": round(pair_max_gap, 4) if pair_max_key else None,
        "max_gap_pair": pair_max_key,
    }

    geom_out.write_text(json.dumps(geom_doc, indent=2), encoding="utf-8")
    print(
        f"[compute_member_geometry] wrote geometry.json "
        f"(iter={iter_n}, {n_members} members, "
        f"facing_ok={n_facing_ok}/{n_members}, clearance_ok={n_clearance_ok}/{n_members}, "
        f"pairs={len(member_pairs)} min_gap={round(pair_min_gap,3) if pair_min_key else 'NA'}m)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deterministic per-member geometry computation for the island refiner. "
            "Writes forward_axes.json (cached) and iter_N/geometry.json."
        )
    )
    p.add_argument("--group-dir", type=Path, required=True,
                   help="Absolute path to relation_groups/<G>/")
    p.add_argument("--iter", dest="iter_n", type=int, required=True,
                   help="Iteration number N (>= 0).")
    p.add_argument("--blender-bin", type=str, default=None,
                   help="Override Blender executable path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    group_dir = args.group_dir.resolve()

    if not group_dir.is_dir():
        print(f"[compute_member_geometry] ERROR: group-dir not found: {group_dir}",
              file=sys.stderr)
        sys.exit(1)

    blender_bin = _resolve_blender(args.blender_bin)
    print(f"[compute_member_geometry] group_dir={group_dir}  iter={args.iter_n}  blender={blender_bin}")

    _compute_and_write(group_dir, args.iter_n, blender_bin)


if __name__ == "__main__":
    main()
