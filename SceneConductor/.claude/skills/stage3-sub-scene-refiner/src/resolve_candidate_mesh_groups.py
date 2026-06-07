"""
resolve_candidate_mesh_groups.py — Stage-3 resolution of Stage-1 candidate mesh groups.

Stage 1 emits `candidate_mesh_groups` in inputs/merge_plan.json: uncertain
same-model hypotheses where each instance got its OWN SAM3D mesh (no sharing).
This module RESOLVES those candidates using the 3D bounding-box dimensions
recorded in json/blend_info.json:

  * For each candidate group, take the canonical instance's effective bbox
    dimensions (recursive child-mesh lookup — the obj_<id> parent EMPTY has
    dimensions [0,0,0]; the real bbox lives on a descendant MESH).
  * For every non-canonical instance, compare its effective dimensions to the
    canonical's using a rotation-robust SORTED-TRIPLE per-axis ratio test.
  * Matching instances are PROMOTED into a CONFIRMED mesh group whose shape is
    identical to the `mesh_groups` dict that heuristic_planner._mesh_group_scale_ops
    already consumes (so confirmed groups flow straight into scale-normalization).

CRITICAL: this module NEVER mutates inputs/merge_plan.json. It writes a new
artifact (json/resolved_mesh_groups.json) and exposes a pure merge helper
`merge_confirmed_into()` for the planner to augment its in-memory view.

Usage (CLI):
    python resolve_candidate_mesh_groups.py --scene_dir <dir> [--tolerance 0.20]

Output:
    <scene_dir>/json/resolved_mesh_groups.json
"""

from __future__ import annotations

import argparse
import copy
import json
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
# Effective bbox dimensions (mirror of graph_tool_planner's recursive walk)
# ---------------------------------------------------------------------------
#
# graph_tool_planner.get_object_width() collapses descendant meshes to a single
# scalar (0.5 * max(dim_x, dim_y)). For a rotation-robust three-axis comparison
# we need the full [x, y, z] triple, so we mirror the same recursive descendant
# walk but keep all three components, selecting the MESH descendant with the
# largest extent (sum of dims) as the object's effective bounding box.

def _build_all_objects(blend_info: dict) -> dict:
    """Build {obj_name: object_dict} across the dimension-bearing categories.

    Mirrors graph_tool_planner (categories: objects / world / geometry_meshes).
    """
    categories = blend_info.get("categories", {}) if blend_info else {}
    all_objects: dict = {}
    for category in ["objects", "world", "geometry_meshes"]:
        for obj in categories.get(category, []):
            name = obj.get("name")
            if name is not None:
                all_objects[name] = obj
    return all_objects


def effective_dimensions(obj_name: str, all_objects: dict) -> list[float] | None:
    """Return the [x, y, z] dimensions of the largest descendant MESH, or None.

    The obj_<id> parent is an EMPTY with dimensions [0,0,0]; the real bbox lives
    on a descendant MESH (e.g. obj_11 -> world.010 -> geometry_0.010). We walk
    the children recursively and pick the MESH whose extent-sum is largest.
    """
    if not obj_name:
        return None

    best_dims: list[float] | None = None
    best_extent = -1.0

    def collect(node_name: str) -> None:
        nonlocal best_dims, best_extent
        node = all_objects.get(node_name)
        if not node:
            return
        if node.get("type") == "MESH":
            dims = node.get("dimensions", [0, 0, 0])
            if isinstance(dims, (list, tuple)) and len(dims) >= 3:
                extent = abs(dims[0]) + abs(dims[1]) + abs(dims[2])
                if extent > best_extent:
                    best_extent = extent
                    best_dims = [float(dims[0]), float(dims[1]), float(dims[2])]
        for child_name in node.get("children", []):
            collect(child_name)

    collect(obj_name)
    return best_dims


# ---------------------------------------------------------------------------
# Matching rule
# ---------------------------------------------------------------------------

def _dims_present(dims: list[float] | None) -> bool:
    """True iff dims is a 3-vector with all components meaningfully non-zero."""
    if not dims or len(dims) < 3:
        return False
    return all(abs(v) > 1e-6 for v in dims[:3])


def _sorted_ratios(canon: list[float], inst: list[float]) -> list[float]:
    """Per-axis max(a/b, b/a) over the ascending-sorted extent triples.

    Sorting both triples makes the comparison robust to rotation: a model rotated
    90 degrees swaps which world axis carries which extent, but the sorted triple
    is invariant.
    """
    c = sorted(abs(v) for v in canon[:3])
    i = sorted(abs(v) for v in inst[:3])
    ratios: list[float] = []
    for a, b in zip(c, i):
        # both guaranteed > 0 by _dims_present guard before this is called
        ratios.append(max(a / b, b / a))
    return ratios


def _matches(canon: list[float], inst: list[float], tolerance: float) -> tuple[bool, list[float]]:
    """Return (is_match, sorted_ratio_per_axis)."""
    ratios = _sorted_ratios(canon, inst)
    ok = all(r <= 1.0 + tolerance for r in ratios)
    return ok, ratios


# ---------------------------------------------------------------------------
# Candidate normalization
# ---------------------------------------------------------------------------

def _iter_candidates(candidates) -> list[dict]:
    """Normalize candidate_mesh_groups (list OR dict shape) into a list of dicts.

    Stage-1 has shipped both shapes:
      * list:  [{"canonical_id":..,"instance_ids":[..],"class":..}, ...]
      * dict:  {group_name: {"canonical_id":..,"instance_ids":[..],"class":..}}
    """
    out: list[dict] = []
    if isinstance(candidates, dict):
        for gname, gdata in candidates.items():
            if isinstance(gdata, dict):
                entry = dict(gdata)
                entry.setdefault("group_name", gname)
                out.append(entry)
    elif isinstance(candidates, list):
        for gdata in candidates:
            if isinstance(gdata, dict):
                out.append(dict(gdata))
    return out


# ---------------------------------------------------------------------------
# Core resolve
# ---------------------------------------------------------------------------

def resolve(scene_dir, tolerance: float = 0.20) -> dict:
    """Resolve candidate_mesh_groups into confirmed_mesh_groups using bbox dims.

    Returns the resolved dict and writes <scene_dir>/json/resolved_mesh_groups.json.
    Safe no-op (empty confirmed/report) if merge_plan or blend_info is missing or
    has no candidate_mesh_groups.
    """
    scene_dir = Path(scene_dir)
    merge_plan_path = scene_dir / "inputs" / "merge_plan.json"
    blend_info_path = _resolve_blend_info_path(scene_dir)

    result: dict = {"confirmed_mesh_groups": {}, "report": []}

    merge_plan = _read_json(merge_plan_path)
    blend_info = _read_json(blend_info_path)

    candidates = _iter_candidates((merge_plan or {}).get("candidate_mesh_groups"))
    if not candidates or not blend_info:
        # Safe no-op: still write the (empty) artifact for downstream consistency.
        _write_resolved(scene_dir, result)
        return result

    all_objects = _build_all_objects(blend_info)

    for cand in candidates:
        canonical_id = cand.get("canonical_id")
        instance_ids = cand.get("instance_ids", []) or []
        cls = cand.get("class")

        canon_dims = effective_dimensions(f"obj_{canonical_id}", all_objects)
        promoted: list[int] = []

        for iid in instance_ids:
            if iid == canonical_id:
                continue
            inst_dims = effective_dimensions(f"obj_{iid}", all_objects)

            if not _dims_present(canon_dims) or not _dims_present(inst_dims):
                decision = "unresolved"
                ratios = None
            else:
                is_match, ratios = _matches(canon_dims, inst_dims, tolerance)
                decision = "promoted" if is_match else "rejected"
                if is_match:
                    promoted.append(iid)

            result["report"].append({
                "canonical_id": canonical_id,
                "instance_id": iid,
                "canonical_dims": canon_dims,
                "instance_dims": inst_dims,
                "sorted_ratio_per_axis": ratios,
                "decision": decision,
            })

        # Only confirm a group that ended with >= 2 members (canonical + >=1 promoted).
        if promoted:
            group_name = f"candidate_confirmed_{canonical_id}"
            result["confirmed_mesh_groups"][group_name] = {
                "canonical_id": canonical_id,
                "instance_ids": [canonical_id] + promoted,
                "class": cls,
            }

    _write_resolved(scene_dir, result)
    return result


def merge_confirmed_into(merge_plan: dict, resolved: dict) -> dict:
    """Return a shallow-copied merge_plan with confirmed groups merged into mesh_groups.

    Pure function — no disk writes, does NOT mutate the input merge_plan. Confirmed
    groups are added under distinct names (candidate_confirmed_<id>) so they never
    clobber pre-existing mesh_groups entries.
    """
    out = dict(merge_plan) if isinstance(merge_plan, dict) else {}
    existing = out.get("mesh_groups", {})
    # Copy the mesh_groups dict so we don't mutate the caller's nested object.
    merged = dict(existing) if isinstance(existing, dict) else {}
    confirmed = (resolved or {}).get("confirmed_mesh_groups", {}) or {}
    for gname, gdata in confirmed.items():
        if gname in merged:
            # Defensive: never clobber an existing group of the same name.
            gname = f"{gname}_resolved"
        merged[gname] = copy.deepcopy(gdata)
    out["mesh_groups"] = merged
    return out


# ---------------------------------------------------------------------------
# Local IO helpers (kept self-contained — module is importable standalone)
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[resolve_candidate_mesh_groups] WARNING: cannot read {path}: {exc}", file=sys.stderr)
        return None


def _write_resolved(scene_dir: Path, result: dict) -> None:
    out_path = scene_dir / "json" / "resolved_mesh_groups.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resolve Stage-1 candidate_mesh_groups via 3D bbox dims.")
    ap.add_argument("--scene_dir", required=True, help="Scene directory (absolute path).")
    ap.add_argument("--tolerance", type=float, default=0.20, help="Per-axis sorted-ratio tolerance (default 0.20).")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    resolved = resolve(args.scene_dir, tolerance=args.tolerance)
    confirmed = resolved.get("confirmed_mesh_groups", {})
    report = resolved.get("report", [])
    promoted = sum(1 for r in report if r.get("decision") == "promoted")
    rejected = sum(1 for r in report if r.get("decision") == "rejected")
    unresolved = sum(1 for r in report if r.get("decision") == "unresolved")
    print(
        f"[resolve_candidate_mesh_groups] confirmed_groups={len(confirmed)} "
        f"promoted={promoted} rejected={rejected} unresolved={unresolved} "
        f"(tolerance={args.tolerance})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
