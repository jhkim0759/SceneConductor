#!/usr/bin/env python3
"""
deterministic_gate.py — Geometric pre-gate for Stage 3 validation.

Checks each relation-group's members against their anchor's world AABB using
deterministic geometry. Runs BEFORE the vision call so that obviously broken
groups (e.g., 9 desk items floating 1 m off the desk) are never missed.

Public API:
    run_pregate(scene_dir: Path) -> dict

Writes <scene_dir>/json/deterministic_gate.json as a side-effect.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tunable thresholds (override with environment variables)
# ---------------------------------------------------------------------------

Z_TOL_M: float = float(os.environ.get("PREGATE_Z_TOL_M", "0.05"))
XY_MARGIN_M: float = float(os.environ.get("PREGATE_XY_MARGIN_M", "0.10"))
SEATED_MAX_DIST_FACTOR: float = float(os.environ.get("PREGATE_SEATED_MAX_DIST_FACTOR", "3.0"))
SEATED_MAX_YAW_DEG: float = float(os.environ.get("PREGATE_SEATED_MAX_YAW_DEG", "75.0"))

_VERSION = "1"

# Edge types handled by this gate (v1 scope)
_HANDLED_EDGE_TYPES = {"on_top_of", "seated_around"}


# ---------------------------------------------------------------------------
# Internal: resolve blender_bin via DIRECTORYS.yaml
# ---------------------------------------------------------------------------

def _repo_root_from_script() -> Path:
    """Walk up from this script's location to find DIRECTORYS.yaml."""
    candidate = Path(__file__).resolve()
    for _ in range(12):
        candidate = candidate.parent
        if (candidate / "DIRECTORYS.yaml").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate DIRECTORYS.yaml within 12 parent directories of {__file__}"
    )


def _resolve_blender_bin() -> str:
    """
    Resolution order:
      1. BLENDER environment variable
      2. SCENE_EVAL_BLENDER environment variable
      3. DIRECTORYS.yaml — platform-specific key first, then blender_bin
    """
    for env_var in ("SCENE_EVAL_BLENDER", "BLENDER"):
        val = os.environ.get(env_var)
        if val:
            return val

    try:
        import yaml  # type: ignore
        repo_root = _repo_root_from_script()
        dirs = yaml.safe_load((repo_root / "DIRECTORYS.yaml").read_text(encoding="utf-8"))
        import platform
        plat = platform.system().lower()
        platform_key = f"blender_bin_{plat}"
        if platform_key in dirs:
            return str(repo_root / dirs[platform_key])
        bin_val = dirs.get("blender_bin", "blender")
        # If relative, resolve from repo root
        if bin_val.startswith("./"):
            return str(repo_root / bin_val)
        return bin_val
    except Exception as exc:
        print(
            f"[deterministic_gate] WARNING: could not read blender_bin from DIRECTORYS.yaml: {exc}",
            file=sys.stderr,
        )
        return "blender"


# ---------------------------------------------------------------------------
# Internal: Blender subprocess helper that dumps world-space AABBs
# ---------------------------------------------------------------------------

# This script is written to a temp file and run inside Blender (--background).
# It dumps a JSON result file at the path passed as its first argument after --.
_BBOX_BLENDER_SCRIPT = r"""
import json
import sys
import bpy

def _find_arg():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("missing -- separator")
    after = argv[argv.index("--") + 1:]
    if not after:
        raise SystemExit("expected: <result_json_path>")
    return after[0], after[1] if len(after) > 1 else "obj_"

result_path, prefix = _find_arg()

def world_bbox(obj):
    # Return ((xmin, ymin, zmin), (xmax, ymax, zmax)) in world space.
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
    try:
        from mathutils import Vector
    except ImportError:
        import mathutils
        Vector = mathutils.Vector
    for m in mesh_descendants:
        for corner in m.bound_box:
            wc = m.matrix_world @ Vector(corner)
            xs.append(wc.x); ys.append(wc.y); zs.append(wc.z)
    return [(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))]

targets = [o for o in bpy.data.objects if o.name.startswith(prefix)]

bboxes = {}
locations = {}
for obj in targets:
    bb = world_bbox(obj)
    bboxes[obj.name] = bb
    locations[obj.name] = list(obj.location)

# Also derive room bbox from Floor / Wall_* / Ceiling if present
room_bbox = None
rx, ry, rz_vals = [], [], []
try:
    from mathutils import Vector
except ImportError:
    import mathutils
    Vector = mathutils.Vector
for o in bpy.data.objects:
    if o.name == "Floor" or o.name.startswith("Wall_") or o.name == "Ceiling":
        for corner in o.bound_box:
            wc = o.matrix_world @ Vector(corner)
            rx.append(wc.x); ry.append(wc.y); rz_vals.append(wc.z)
if rx:
    room_bbox = [[min(rx), min(ry), min(rz_vals)], [max(rx), max(ry), max(rz_vals)]]

result = {
    "success": True,
    "bboxes": {name: [list(lo), list(hi)] for name, (lo, hi) in bboxes.items()},
    "locations": locations,
    "room_bbox": room_bbox,
}
with open(result_path, "w") as f:
    json.dump(result, f)
"""


def _run_blender_bbox(blend_path: str, blender_bin: str, name_prefix: str = "obj_",
                      timeout: int = 120) -> dict:
    """
    Spawn Blender headlessly to collect per-object world-space AABBs.

    Returns the parsed result dict on success.  Raises RuntimeError on failure.
    """
    fd, script_path = tempfile.mkstemp(prefix="deterministic_gate_", suffix=".py")
    os.close(fd)
    fd2, result_path = tempfile.mkstemp(prefix="deterministic_gate_result_", suffix=".json")
    os.close(fd2)

    try:
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_BBOX_BLENDER_SCRIPT)

        cmd = [
            blender_bin,
            "--background", blend_path,
            "--python", script_path,
            "--",
            result_path,
            name_prefix,
        ]

        print(f"[deterministic_gate] Running: {' '.join(cmd)}", file=sys.stderr)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode != 0:
            tail = proc.stderr[-1000:] if proc.stderr else ""
            raise RuntimeError(
                f"Blender exited with code {proc.returncode}.\n"
                f"stderr tail:\n{tail}"
            )

        if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
            tail = proc.stderr[-1000:] if proc.stderr else ""
            raise RuntimeError(
                f"Blender ran but produced no result file.\n"
                f"stderr tail:\n{tail}"
            )

        with open(result_path, encoding="utf-8") as fh:
            data = json.load(fh)

        if not data.get("success"):
            raise RuntimeError(f"Blender script reported failure: {data}")

        return data

    finally:
        for p in (script_path, result_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _compute_world_aabbs(blend_path: Path) -> dict[str, list]:
    """
    Returns {obj_name: [[xmin,ymin,zmin],[xmax,ymax,zmax]]}.

    Spawns a Blender subprocess to read world-space AABBs.
    Raises on any failure — does NOT silently fall back.
    """
    blender_bin = _resolve_blender_bin()
    data = _run_blender_bbox(str(blend_path), blender_bin, name_prefix="obj_")

    raw_bboxes = data.get("bboxes", {})
    # Normalise to list-of-lists [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    result: dict[str, list] = {}
    for name, bbox in raw_bboxes.items():
        lo, hi = bbox
        result[name] = [list(lo), list(hi)]
    return result


# ---------------------------------------------------------------------------
# Per-edge-type geometric checks
# ---------------------------------------------------------------------------

def _check_on_top_of(
    member_aabb: list,
    anchor_aabb: list,
    z_tol: float,
    xy_margin: float,
) -> Optional[dict]:
    """
    Anomaly if:
      - |member.zmin - anchor.zmax| > z_tol   (floating OR submerged)
      - member XY centroid is outside anchor XY bbox expanded by xy_margin

    Returns a reason dict or None if all checks pass.
    """
    m_lo, m_hi = member_aabb
    a_lo, a_hi = anchor_aabb

    member_zmin = m_lo[2]
    anchor_zmax = a_hi[2]

    # Z gap check: member base should rest on anchor top
    z_gap = member_zmin - anchor_zmax
    floating_or_sinking = abs(z_gap) > z_tol

    # XY containment check: member centroid within anchor XY footprint + margin
    member_cx = (m_lo[0] + m_hi[0]) / 2.0
    member_cy = (m_lo[1] + m_hi[1]) / 2.0

    ax_min = a_lo[0] - xy_margin
    ax_max = a_hi[0] + xy_margin
    ay_min = a_lo[1] - xy_margin
    ay_max = a_hi[1] + xy_margin

    xy_off_anchor = not (ax_min <= member_cx <= ax_max and ay_min <= member_cy <= ay_max)

    if not floating_or_sinking and not xy_off_anchor:
        return None

    details_parts = []
    if floating_or_sinking:
        if z_gap > 0:
            details_parts.append(
                f"member_base_z={member_zmin:.3f} anchor_top_z={anchor_zmax:.3f} "
                f"gap={z_gap:.3f}m exceeds {z_tol}m tolerance (floating)"
            )
        else:
            details_parts.append(
                f"member_base_z={member_zmin:.3f} anchor_top_z={anchor_zmax:.3f} "
                f"gap={abs(z_gap):.3f}m exceeds {z_tol}m tolerance (sunk below surface)"
            )
    if xy_off_anchor:
        details_parts.append(
            f"member_xy_centroid=({member_cx:.3f},{member_cy:.3f}) outside anchor "
            f"XY bbox [{a_lo[0]:.3f},{a_lo[1]:.3f}]→[{a_hi[0]:.3f},{a_hi[1]:.3f}] "
            f"(+{xy_margin}m margin)"
        )

    return {
        "floating_or_sinking": floating_or_sinking,
        "xy_off_anchor": xy_off_anchor,
        "details": "; ".join(details_parts),
    }


def _check_seated_around(
    member_aabb: list,
    member_yaw_rad: float,
    anchor_aabb: list,
    max_dist_factor: float,
    max_yaw_deg: float,
) -> Optional[dict]:
    """
    Anomaly if:
      - 2D distance from member XY centroid to anchor XY centroid >
        max_dist_factor * anchor_half_diagonal
      - Member yaw not aligned toward the anchor direction (within max_yaw_deg)

    Returns a reason dict or None if all checks pass.
    """
    m_lo, m_hi = member_aabb
    a_lo, a_hi = anchor_aabb

    member_cx = (m_lo[0] + m_hi[0]) / 2.0
    member_cy = (m_lo[1] + m_hi[1]) / 2.0
    anchor_cx = (a_lo[0] + a_hi[0]) / 2.0
    anchor_cy = (a_lo[1] + a_hi[1]) / 2.0

    dx = anchor_cx - member_cx
    dy = anchor_cy - member_cy
    dist = math.sqrt(dx * dx + dy * dy)

    # Anchor half-diagonal in XY
    anchor_w = a_hi[0] - a_lo[0]
    anchor_d = a_hi[1] - a_lo[1]
    anchor_half_diag = math.sqrt(anchor_w * anchor_w + anchor_d * anchor_d) / 2.0

    # Avoid division by zero for degenerate anchor
    if anchor_half_diag < 1e-6:
        anchor_half_diag = 1e-6

    dist_threshold = max_dist_factor * anchor_half_diag
    too_far = dist > dist_threshold

    # Yaw alignment: forward axis (cos(yaw), sin(yaw)) vs. direction to anchor
    forward_x = math.cos(member_yaw_rad)
    forward_y = math.sin(member_yaw_rad)

    if dist > 1e-6:
        dir_to_anchor_x = dx / dist
        dir_to_anchor_y = dy / dist
        dot = forward_x * dir_to_anchor_x + forward_y * dir_to_anchor_y
        dot = max(-1.0, min(1.0, dot))  # clamp for acos safety
        angle_misalignment_deg = math.degrees(math.acos(dot))
    else:
        # Member is at anchor centroid — yaw check not meaningful
        angle_misalignment_deg = 0.0

    yaw_misaligned = angle_misalignment_deg > max_yaw_deg

    if not too_far and not yaw_misaligned:
        return None

    details_parts = []
    if too_far:
        details_parts.append(
            f"dist_to_anchor={dist:.3f}m exceeds "
            f"{max_dist_factor}x anchor_half_diag({anchor_half_diag:.3f}m)="
            f"{dist_threshold:.3f}m"
        )
    if yaw_misaligned:
        details_parts.append(
            f"yaw_misalignment={angle_misalignment_deg:.1f}deg exceeds {max_yaw_deg}deg "
            f"(member not facing anchor)"
        )

    return {
        "too_far": too_far,
        "yaw_misaligned": yaw_misaligned,
        "details": "; ".join(details_parts),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pregate(scene_dir: Path) -> dict:
    """
    Read blend_info / relation_graph / object_state from scene_dir,
    compute world AABBs (via _run_blender_bbox on blend/stage3-sub-planned.blend),
    and check each group's members against its edge_type geometric contract.

    Returns:
      {
        "flagged_groups": ["G1", ...],
        "rationale_by_group": {"G1": "obj_X is 1.23m above anchor obj_4 top surface; ..."},
        "per_member_checks": {
          "G1": [
            {"member":"obj_3","edge_type":"on_top_of","ok":false,
             "reason":"member_base_z=4.21 anchor_top_z=2.98 gap=1.23m exceeds 0.05m tolerance"},
            ...
          ]
        },
        "skipped_groups": {"G3": "anchor 'left' is non-object (wall) — wall checks deferred"},
        "thresholds": {"z_tol_m": 0.05, "xy_margin_m": 0.10,
                       "seated_max_dist_factor": 3.0, "seated_max_yaw_deg": 75},
        "blend_used": "<abs path to blend>",
        "meta": {"generator": "deterministic_gate.py", "version": "1", "timestamp_utc": "..."}
      }

    Also writes the same dict to <scene_dir>/json/deterministic_gate.json.

    Graceful degradation:
      - If blend_info.json or blend/stage3-sub-planned.blend is missing,
        return {"flagged_groups": [], "skipped_groups": {"_all": "blend_info or blend missing"}, ...}
        and log a clear warning to stdout. The caller will then fall back to vision-only.
    """
    scene_dir = Path(scene_dir).resolve()

    thresholds = {
        "z_tol_m": Z_TOL_M,
        "xy_margin_m": XY_MARGIN_M,
        "seated_max_dist_factor": SEATED_MAX_DIST_FACTOR,
        "seated_max_yaw_deg": SEATED_MAX_YAW_DEG,
    }

    meta = {
        "generator": "deterministic_gate.py",
        "version": _VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    def _empty_result(skip_reason: str) -> dict:
        result = {
            "flagged_groups": [],
            "rationale_by_group": {},
            "per_member_checks": {},
            "skipped_groups": {"_all": skip_reason},
            "thresholds": thresholds,
            "blend_used": None,
            "meta": meta,
        }
        _write_gate_json(scene_dir, result)
        return result

    # --- Load prerequisites ---------------------------------------------------

    blend_path = scene_dir / "blend" / "stage3-sub-planned.blend"
    blend_info_path = scene_dir / "json" / "blend_info.json"
    relation_graph_path = scene_dir / "inputs" / "relation_graph.json"
    object_class_path = scene_dir / "inputs" / "object_class.json"

    if not blend_path.exists():
        print(
            f"[deterministic_gate] WARNING: blend not found: {blend_path} — skipping pre-gate",
            file=sys.stderr,
        )
        return _empty_result(f"blend not found: {blend_path}")

    if not blend_info_path.exists():
        print(
            f"[deterministic_gate] WARNING: blend_info.json not found: {blend_info_path} — skipping pre-gate",
            file=sys.stderr,
        )
        return _empty_result(f"blend_info.json not found: {blend_info_path}")

    if not relation_graph_path.exists():
        print(
            f"[deterministic_gate] WARNING: relation_graph.json not found: {relation_graph_path} — skipping pre-gate",
            file=sys.stderr,
        )
        return _empty_result(f"relation_graph.json not found: {relation_graph_path}")

    # object_class.json is optional for skipping — if missing, we can't check anchors
    object_class: dict = {}
    if object_class_path.exists():
        try:
            object_class = json.loads(object_class_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"[deterministic_gate] WARNING: could not parse object_class.json: {exc}",
                file=sys.stderr,
            )

    try:
        blend_info = json.loads(blend_info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[deterministic_gate] WARNING: could not parse blend_info.json: {exc} — skipping pre-gate",
            file=sys.stderr,
        )
        return _empty_result(f"blend_info.json parse error: {exc}")

    try:
        relation_graph = json.loads(relation_graph_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[deterministic_gate] WARNING: could not parse relation_graph.json: {exc} — skipping pre-gate",
            file=sys.stderr,
        )
        return _empty_result(f"relation_graph.json parse error: {exc}")

    # --- Build yaw lookup from blend_info.categories.objects ------------------
    # blend_info.categories.objects[].rotation_euler[2] = Z rotation (yaw)
    yaw_by_obj: dict[str, float] = {}
    cats = blend_info.get("categories", {})
    for obj_entry in cats.get("objects", []):
        name = obj_entry.get("name", "")
        rot = obj_entry.get("rotation_euler", [0.0, 0.0, 0.0])
        if len(rot) >= 3:
            yaw_by_obj[name] = float(rot[2])

    # --- Compute world AABBs from Blender -------------------------------------
    print(
        f"[deterministic_gate] Computing world AABBs from: {blend_path}",
        file=sys.stderr,
    )
    try:
        aabbs = _compute_world_aabbs(blend_path)
    except Exception as exc:
        print(
            f"[deterministic_gate] WARNING: AABB computation failed: {exc} — skipping pre-gate",
            file=sys.stderr,
        )
        return _empty_result(f"AABB computation failed: {exc}")

    print(
        f"[deterministic_gate] AABBs computed for {len(aabbs)} objects.",
        file=sys.stderr,
    )

    # --- Process each group ---------------------------------------------------
    groups = relation_graph.get("groups", [])

    flagged_groups: list[str] = []
    rationale_by_group: dict[str, str] = {}
    per_member_checks: dict[str, list] = {}
    skipped_groups: dict[str, str] = {}

    for group in groups:
        group_id = group.get("group_id", "")
        edge_type = group.get("edge_type", "")
        anchor = group.get("anchor", "")
        members: list[str] = group.get("members", [])

        if not group_id:
            continue

        # Skip if edge_type is not in v1 scope
        if edge_type not in _HANDLED_EDGE_TYPES:
            skipped_groups[group_id] = (
                f"edge_type '{edge_type}' not in v1 scope "
                f"(handled: {sorted(_HANDLED_EDGE_TYPES)})"
            )
            continue

        # Skip if anchor is not a key in object_class (wall / abstract anchors).
        # object_class.json uses bare numeric keys ("4") while relation_graph uses "obj_4".
        # Try both forms.
        if object_class:
            anchor_numeric = anchor.removeprefix("obj_") if anchor.startswith("obj_") else anchor
            if anchor not in object_class and anchor_numeric not in object_class:
                skipped_groups[group_id] = (
                    f"anchor '{anchor}' is non-object (not in object_class.json) — "
                    "wall/abstract anchor checks deferred to v2"
                )
                continue

        # Skip if anchor has no AABB
        if anchor not in aabbs:
            skipped_groups[group_id] = (
                f"anchor '{anchor}' has no AABB in computed blend data"
            )
            continue

        anchor_aabb = aabbs[anchor]
        member_results: list[dict] = []
        group_has_anomaly = False
        group_anomaly_parts: list[str] = []

        for member in members:
            if member not in aabbs:
                member_results.append({
                    "member": member,
                    "edge_type": edge_type,
                    "ok": None,
                    "reason": "no AABB — object may be missing from blend",
                })
                continue

            member_aabb = aabbs[member]

            if edge_type == "on_top_of":
                anomaly = _check_on_top_of(
                    member_aabb, anchor_aabb, Z_TOL_M, XY_MARGIN_M
                )
            elif edge_type == "seated_around":
                member_yaw = yaw_by_obj.get(member, 0.0)
                anomaly = _check_seated_around(
                    member_aabb, member_yaw, anchor_aabb,
                    SEATED_MAX_DIST_FACTOR, SEATED_MAX_YAW_DEG,
                )
            else:
                # Should not reach here given the earlier edge_type check
                anomaly = None

            if anomaly is None:
                member_results.append({
                    "member": member,
                    "edge_type": edge_type,
                    "ok": True,
                    "reason": "pass",
                })
            else:
                group_has_anomaly = True
                reason_str = f"{member} ({edge_type}): {anomaly['details']}"
                group_anomaly_parts.append(reason_str)
                member_results.append({
                    "member": member,
                    "edge_type": edge_type,
                    "ok": False,
                    "reason": anomaly["details"],
                })

        per_member_checks[group_id] = member_results

        if group_has_anomaly:
            flagged_groups.append(group_id)
            rationale_by_group[group_id] = "; ".join(group_anomaly_parts)
            print(
                f"[deterministic_gate] Group {group_id} FLAGGED: "
                + "; ".join(group_anomaly_parts[:2]),
                file=sys.stderr,
            )
        else:
            print(
                f"[deterministic_gate] Group {group_id} ({edge_type}): all {len(members)} member(s) pass.",
                file=sys.stderr,
            )

    result = {
        "flagged_groups": sorted(flagged_groups),
        "rationale_by_group": rationale_by_group,
        "per_member_checks": per_member_checks,
        "skipped_groups": skipped_groups,
        "thresholds": thresholds,
        "blend_used": str(blend_path),
        "meta": meta,
    }

    _write_gate_json(scene_dir, result)
    print(
        f"[deterministic_gate] Pre-gate complete. "
        f"Flagged: {result['flagged_groups']} | "
        f"Skipped: {list(skipped_groups.keys())}",
        file=sys.stderr,
    )
    return result


def _write_gate_json(scene_dir: Path, result: dict) -> None:
    """Write the gate result to <scene_dir>/json/deterministic_gate.json."""
    out_path = scene_dir / "json" / "deterministic_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[deterministic_gate] Written: {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point (for manual testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run deterministic pre-gate on a scene_dir")
    parser.add_argument("--scene_dir", type=Path, required=True)
    args = parser.parse_args()

    res = run_pregate(args.scene_dir)
    print(json.dumps(
        {k: v for k, v in res.items() if k != "per_member_checks"},
        indent=2,
        ensure_ascii=False,
    ))
    print("--- per_member_checks ---")
    print(json.dumps(res.get("per_member_checks", {}), indent=2, ensure_ascii=False))
