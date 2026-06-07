"""
heuristic_planner.py — Deterministic pre-pass for the 3D scene planner.

Reads structured JSON files and emits operations that do not require LLM
reasoning, leaving vision-dependent decisions to the downstream LLM planner.

Usage:
    python3 heuristic_planner.py --scene-dir /path/to/scene_dir
    python3 heuristic_planner.py --scene-dir /path/to/scene_dir --output /path/to/out.json

Output:
    <scene_dir>/json/heuristic_ops.json  (or --output path)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
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

# Stage-3 candidate-mesh-group resolution (in-memory, non-destructive).
# Imported from the same skill src/ dir; degrade gracefully if unavailable.
try:
    from resolve_candidate_mesh_groups import (
        resolve as _resolve_candidate_groups,
        merge_confirmed_into as _merge_confirmed_into,
    )
except ImportError:  # pragma: no cover - import path fallback
    _resolve_candidate_groups = None
    _merge_confirmed_into = None


# ---------------------------------------------------------------------------
# Class taxonomy (copied verbatim from
#   src/blend_ops/session_runner/scene_analysis.py
# Copied — NOT imported — for cross-skill decoupling. Keep in sync manually.
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


def _class_matches(class_name: str, taxonomy: list[str]) -> bool:
    """Case-insensitive substring/word match.

    Returns True if any taxonomy keyword appears as a substring of the
    (lower-cased) class name. This lets multi-word class names match a
    single taxonomy keyword: "table counter" matches "table" or "counter";
    "bookshelf cabinet" matches "bookshelf" or "cabinet".
    """
    if not class_name:
        return False
    cl = class_name.lower()
    for kw in taxonomy:
        if kw in cl:
            return True
    return False


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> dict | None:
    """Return parsed JSON or None (with a warning) if the file is missing/invalid."""
    if not path.exists():
        print(f"[heuristic_planner] WARNING: {label} not found at {path} — skipping.", file=sys.stderr)
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[heuristic_planner] WARNING: {label} at {path} is invalid JSON ({exc}) — skipping.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------

def _scale_magnitude(scale: list[float]) -> float:
    """Mean of absolute values of the three scale components."""
    return sum(abs(v) for v in scale) / len(scale)


# ---------------------------------------------------------------------------
# Priority-1: mesh-group same-scale ops
# ---------------------------------------------------------------------------

def _build_blender_index(blender_scene: dict) -> dict[str, dict]:
    """Return {obj_id: object_dict} from blender_scene.json."""
    index: dict[str, dict] = {}
    for obj in blender_scene.get("objects", []):
        obj_id = obj.get("id")
        if obj_id:
            index[obj_id] = obj
    return index


def _mesh_group_scale_ops(
    merge_plan: dict,
    blender_index: dict[str, dict],
) -> list[dict]:
    """Emit update_size ops for members whose scale deviates from the group median."""
    ops: list[dict] = []
    mesh_groups: dict = merge_plan.get("mesh_groups", {})

    for group_name, group_data in mesh_groups.items():
        # mesh_groups entries come in two shapes across pipeline versions:
        #   {group: {"instance_ids": [...]}}  (dict)  and  {group: [...]}  (bare list).
        if isinstance(group_data, dict):
            instance_ids: list[int] = group_data.get("instance_ids", [])
        elif isinstance(group_data, list):
            instance_ids = group_data
        else:
            print(
                f"[heuristic_planner] WARNING: mesh_group '{group_name}' has unexpected "
                f"type {type(group_data).__name__} — skipping group.",
                file=sys.stderr,
            )
            continue
        obj_names = [f"obj_{iid}" for iid in instance_ids]

        # Collect magnitudes for members present in blender_scene.json
        magnitudes: list[float] = []
        member_data: list[tuple[str, float, list[float]]] = []  # (name, mag, raw_scale)

        for name in obj_names:
            obj = blender_index.get(name)
            if obj is None:
                print(
                    f"[heuristic_planner] WARNING: mesh_group '{group_name}' member "
                    f"'{name}' not found in blender_scene.json — skipping member.",
                    file=sys.stderr,
                )
                continue
            raw_scale = obj.get("scale", [1.0, 1.0, 1.0])
            mag = _scale_magnitude(raw_scale)
            magnitudes.append(mag)
            member_data.append((name, mag, raw_scale))

        if not magnitudes:
            continue

        median_mag = statistics.median(magnitudes)
        instance_ids_str = str(instance_ids)

        for name, mag, _raw_scale in member_data:
            if abs(mag - median_mag) <= 1e-9:
                continue  # already at median — nothing to do
            ops.append({
                "action": "update_size",
                "obj_name": name,
                "scale": [median_mag, median_mag, median_mag],
                "reason": (
                    f"mesh_group '{group_name}' median scale is {median_mag} "
                    f"(computed from instance_ids {instance_ids_str} in json/blender_scene.json); "
                    f"{name} current scale {mag} differs."
                ),
                "criteria_used": ["mesh_group_same_scale"],
                "priority": 1,
                "confidence": 0.95,
                "requires_planner_review": False,
                "source": "heuristic",
            })

    return ops


# ---------------------------------------------------------------------------
# Priority-3: environment attachment (floor / ceiling)
# ---------------------------------------------------------------------------

def _build_class_maps(
    object_state: dict,
    object_class: dict | None,
) -> tuple[dict[int, str], dict[str, str]]:
    """Build label_to_obj (label_id -> obj_id) and obj_to_class (obj_id -> class).

    object_class.json keys are string label_ids; object_state.json maps
    label_id <-> obj_id. obj_to_class is empty if object_class is unavailable.
    """
    label_to_obj: dict[int, str] = {}
    obj_to_class: dict[str, str] = {}
    class_lookup = object_class or {}

    for obj_entry in object_state.get("objects", []):
        obj_id: str = obj_entry.get("obj_id", "")
        label_id = obj_entry.get("label_id")
        if not obj_id or label_id is None:
            continue
        label_to_obj[label_id] = obj_id
        cls = class_lookup.get(str(label_id))
        if cls:
            obj_to_class[obj_id] = cls

    return label_to_obj, obj_to_class


def _load_ground_selection(scene_dir: Path) -> tuple[set[str], set[str]]:
    """Read optional LLM visual ground selection from json/ground_objects.json.

    Schema:
        {"ground_objects": ["obj_3"], "not_ground_objects": ["obj_1"],
         "reason": {"obj_1": "on shelf"}}

    Returns (llm_ground, llm_not_ground). On missing/malformed file: logs a
    one-line warning and returns two empty sets (class-based fallback).
    """
    path = scene_dir / "json" / "ground_objects.json"
    if not path.exists():
        print(
            "[heuristic_planner] INFO: no json/ground_objects.json — "
            "falling back to class-based ground selection.",
            file=sys.stderr,
        )
        return set(), set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
        llm_ground = {x for x in data.get("ground_objects", []) if isinstance(x, str)}
        llm_not_ground = {x for x in data.get("not_ground_objects", []) if isinstance(x, str)}
        print(
            f"[heuristic_planner] INFO: loaded ground_objects.json "
            f"(ground={len(llm_ground)}, not_ground={len(llm_not_ground)}).",
            file=sys.stderr,
        )
        return llm_ground, llm_not_ground
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(
            f"[heuristic_planner] WARNING: json/ground_objects.json is malformed "
            f"({exc}) — falling back to class-based ground selection.",
            file=sys.stderr,
        )
        return set(), set()


def _relation_exclusions(relation_graph: dict | None) -> set[str]:
    """Objects that rest on a surface / wall / ceiling per relation_graph.

    Excludes members of:
      - on_top_of groups whose anchor is NOT Floor (rest on a surface)
      - mounted_on_same_wall groups
      - co_illuminates groups (ceiling lighting arrays)
    """
    excluded: set[str] = set()
    if not relation_graph:
        return excluded
    for group in relation_graph.get("groups", []):
        edge_type = group.get("edge_type", "")
        anchor = str(group.get("anchor", "")).lower()
        members = group.get("members", [])
        if edge_type == "on_top_of":
            if anchor in ("floor",):
                continue  # rests on the floor — not an exclusion
            excluded.update(members)
        elif edge_type in ("mounted_on_same_wall", "co_illuminates"):
            excluded.update(members)
    return excluded


def _env_attach_ops(
    object_state: dict,
    support_covered: set[str],  # objects already handled by relation_graph
    object_class: dict | None,
    relation_graph: dict | None,
    scene_dir: Path,
) -> list[dict]:
    """Emit floor-grounding (+ ceiling) attach ops.

    Ground selection (force floor-grounding):
        final_ground = (class_candidates ∪ attached_to_floor ∪ llm.ground_objects)
                       − (exclusions ∪ llm.not_ground_objects)
    LLM (json/ground_objects.json) holds exclusion authority: an object in
    llm.not_ground_objects is dropped even if it matched a FLOOR class.
    A floor-attach op is emitted for EVERY object in final_ground.
    """
    ops: list[dict] = []

    _label_to_obj, obj_to_class = _build_class_maps(object_state, object_class)

    # --- union sources ---
    class_candidates: set[str] = set()
    attached_to_floor: set[str] = set()
    has_ceiling_set: set[str] = set()
    all_obj_ids: list[str] = []

    for obj_entry in object_state.get("objects", []):
        obj_id: str = obj_entry.get("obj_id", "")
        if not obj_id:
            continue
        all_obj_ids.append(obj_id)
        attached_to: list[str] = obj_entry.get("attached_to", [])
        if "floor" in attached_to:
            attached_to_floor.add(obj_id)
        if "ceiling" in attached_to:
            has_ceiling_set.add(obj_id)
        cls = obj_to_class.get(obj_id, "")
        if cls and _class_matches(cls, FLOOR_CLASSES):
            class_candidates.add(obj_id)

    llm_ground, llm_not_ground = _load_ground_selection(scene_dir)

    # --- exclusions ---
    exclusions: set[str] = set()
    exclusions |= _relation_exclusions(relation_graph)
    exclusions |= set(support_covered)  # non-floor support anchor already handled
    for obj_id in all_obj_ids:
        cls = obj_to_class.get(obj_id, "")
        if cls and (_class_matches(cls, CEILING_CLASSES) or _class_matches(cls, WALL_CLASSES)):
            exclusions.add(obj_id)

    # --- combine ---
    final_ground = (class_candidates | attached_to_floor | llm_ground) - (exclusions | llm_not_ground)

    # --- emit floor-attach ops (de-duplicated by moving_obj) ---
    emitted: set[str] = set()
    for obj_id in sorted(final_ground):
        if obj_id in emitted:
            continue
        emitted.add(obj_id)

        # Build a reason stating the selection basis.
        bases: list[str] = []
        if obj_id in llm_ground:
            bases.append("LLM ground_objects")
        if obj_id in attached_to_floor:
            bases.append("object_state attached_to=floor")
        if obj_id in class_candidates:
            cls = obj_to_class.get(obj_id, "?")
            bases.append(f"class='{cls}' matched FLOOR_CLASSES")
        basis = "; ".join(bases) if bases else "ground selection"

        ops.append({
            "action": "attach",
            "anchor_obj": "Floor",
            "moving_obj": obj_id,
            "relation": "on",
            "reason": f"force floor-grounding {obj_id} ({basis})",
            "criteria_used": ["environment_attachment"],
            "priority": 3,
            "confidence": 0.9,
            "requires_planner_review": False,
            "source": "heuristic",
        })

    # --- ceiling attachments (preserve existing behavior) ---
    for obj_id in sorted(has_ceiling_set):
        if obj_id in support_covered:
            continue
        ops.append({
            "action": "attach",
            "anchor_obj": "Ceiling",
            "moving_obj": obj_id,
            "relation": "-z",
            "reason": f"object_state.json reports {obj_id} attached_to includes 'ceiling'",
            "criteria_used": ["environment_attachment"],
            "priority": 3,
            "confidence": 0.85,
            "requires_planner_review": False,
            "source": "heuristic",
        })

    return ops


# ---------------------------------------------------------------------------
# Priority-4: support attachment (relation_graph on_top_of)
# ---------------------------------------------------------------------------

def _support_attach_ops(
    relation_graph: dict,
    floor_covered: set[str],  # objects with a floor-attach from object_state
) -> tuple[list[dict], set[str]]:
    """
    Emit attach ops for on_top_of relationships.

    Returns (ops, support_covered) where support_covered is the set of
    moving_obj IDs that were handled here (so the caller can suppress
    conflicting floor-attach ops).
    """
    ops: list[dict] = []
    support_covered: set[str] = set()

    for group in relation_graph.get("groups", []):
        if group.get("edge_type") != "on_top_of":
            continue

        group_id: str = group.get("group_id", "")
        group_name: str = group.get("name", "")
        anchor: str = group.get("anchor", "")
        members: list[str] = group.get("members", [])

        for member in members:
            # Skip if the support anchor IS Floor (priority-3 already covers it)
            if anchor.lower() == "floor":
                continue

            support_covered.add(member)
            ops.append({
                "action": "attach",
                "anchor_obj": anchor,
                "moving_obj": member,
                "relation": "on",
                "reason": (
                    f"relation_graph.json {group_id} '{group_name}' "
                    f"edge_type=on_top_of: {member} rests on anchor {anchor}"
                ),
                "criteria_used": ["support_relationship"],
                "priority": 4,
                "confidence": 0.8,
                "requires_planner_review": False,
                "source": "heuristic",
            })

    return ops, support_covered


# ---------------------------------------------------------------------------
# Priority-1: yaw rotation snap to nearest 45° grid
# ---------------------------------------------------------------------------

def _yaw_snap_ops(
    blender_scene: dict,
    object_class: dict | None,
    merge_plan: dict | None = None,
    relation_graph: dict | None = None,
) -> tuple[list[dict], dict]:
    """For each obj_* with a non-trivial yaw misalignment, emit an update_rotation
    op that snaps rotation_euler[2] to the nearest 45° grid step (8 directions:
    0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°). Pitch (index 0) and roll
    (index 1) are preserved unchanged.

    Snap math (floor objects):
        yaw_norm  = atan2(sin(yaw), cos(yaw))   # normalises to [-π, π]
        step      = π / 4                        # 45°
        snapped   = round(yaw_norm / step) * step
        delta     = normalise(snapped − yaw_norm)
        emit if |delta| > 1e-4 rad

    Ceiling objects (CEILING_CLASSES) — Option E: mesh_group median + 90° snap:
        - Group ceiling instances by their mesh_group (from merge_plan).
        - Compute median of normalised yaws within each group.
        - Snap median to nearest 90° grid (π/2 multiples).
        - Emit update_rotation for every group member with that snapped yaw.
        - Single-instance groups (or objects with no mesh_group entry) snap their
          own yaw to the 90° grid directly.

    Object filter:
        - Only obj_* prefixed names (skip Floor / Wall_NN / Ceiling).
        - Wall-mounted objects (WALL_CLASSES) are skipped — rotated by attach_to_wall.
        - Members of `seated_around` or `on_top_of` relation groups are skipped — a
          global 45° grid is wrong for them; their target yaw is anchor-relative
          (face toward anchor) and is set later by the planner-review or island
          refiner. Snapping them here can WORSEN alignment (e.g. a chair at 119°
          toward a 90°-anchor would be snapped to 135°, further from the target).

    Returns (ops, summary_counts).
    """
    ops: list[dict] = []
    n_emitted = 0
    n_aligned = 0
    n_skipped_class = 0
    n_skipped_relation = 0
    n_ceil_emitted = 0
    n_ceil_meshgroups = 0

    # Build set of obj_ids that are members of seated_around / on_top_of groups.
    # Their orientation is anchor-relative (toward_anchor), not aligned to a
    # global 45° grid. Anchors are NOT excluded — anchor (e.g., a dining table)
    # is structural and snapping it is appropriate.
    RELATION_EXCLUDED_EDGE_TYPES = {"seated_around", "on_top_of"}
    relation_excluded: set[str] = set()
    if relation_graph is not None:
        for _grp in relation_graph.get("groups", []):
            if _grp.get("edge_type") in RELATION_EXCLUDED_EDGE_TYPES:
                for _m in _grp.get("members", []):
                    if isinstance(_m, str) and _m.startswith("obj_"):
                        relation_excluded.add(_m)

    step_45 = math.pi / 4   # 45° in radians
    step_90 = math.pi / 2   # 90° in radians

    # Build obj_id → class string from object_class (keyed by string label_id).
    # object_class is None when the file is unavailable — we still process all
    # obj_* objects, just without class-based filtering (ceiling/wall checks skip).
    # A more cautious alternative would be to bail early, but the spec says
    # "skip ceiling/wall" — if we can't determine class we err on the safe side
    # and let the snap run (no attach op conflicts for floor objects).
    label_to_class: dict[str, str] = object_class if isinstance(object_class, dict) else {}

    # We need obj_id → label_id to look up classes. Build a reverse map from
    # blender_scene itself: use the object id suffix as the instance id, then
    # cross-reference object_class whose keys are string label_ids.
    # object_class keys are label_ids which equal the numeric suffix of obj_NN.
    def _obj_class(obj_id: str) -> str:
        """Look up class name for obj_NN using the label_id = NN convention."""
        suffix = obj_id[len("obj_"):]  # e.g. "3" from "obj_3"
        return label_to_class.get(suffix, "")

    # -------------------------------------------------------------------
    # Build ceiling-object → mesh_group map from merge_plan.
    # mesh_groups entries come in two shapes (dict with instance_ids, or bare list).
    # Only ceiling-class members are tracked here.
    # -------------------------------------------------------------------
    # ceil_obj_to_group: obj_id -> group_name
    # ceil_group_to_members: group_name -> [obj_id, ...]
    ceil_obj_to_group: dict[str, str] = {}
    ceil_group_to_members: dict[str, list[str]] = {}

    if merge_plan is not None:
        mesh_groups: dict = merge_plan.get("mesh_groups", {})
        for group_name, group_data in mesh_groups.items():
            if isinstance(group_data, dict):
                instance_ids: list[int] = group_data.get("instance_ids", [])
            elif isinstance(group_data, list):
                instance_ids = group_data
            else:
                instance_ids = []
            for iid in instance_ids:
                oid = f"obj_{iid}"
                cls = _obj_class(oid)
                if cls and _class_matches(cls, CEILING_CLASSES):
                    ceil_obj_to_group[oid] = group_name
                    ceil_group_to_members.setdefault(group_name, []).append(oid)

    # Build a quick lookup of blender objects by id for ceiling group processing.
    blender_obj_index: dict[str, dict] = {
        obj.get("id", ""): obj
        for obj in blender_scene.get("objects", [])
        if obj.get("id", "").startswith("obj_")
    }

    # -------------------------------------------------------------------
    # Ceiling pass: mesh_group median + 90° snap.
    # Process each ceiling mesh_group exactly once.
    # -------------------------------------------------------------------
    processed_ceiling: set[str] = set()  # obj_ids handled in the ceiling pass

    def _snap_90(yaw_rad: float) -> float:
        """Snap yaw_rad to nearest 90° grid, result normalised to [-π, π]."""
        snapped_raw = round(yaw_rad / step_90) * step_90
        return math.atan2(math.sin(snapped_raw), math.cos(snapped_raw))

    # Collect all ceiling objects (with and without a mesh_group assignment).
    # First handle grouped ones, then orphan ceiling objects.
    for group_name, members in ceil_group_to_members.items():
        n_ceil_meshgroups += 1

        # Gather normalised yaws for members present in blender_scene.
        group_yaws: list[tuple[str, float, float, float]] = []  # (obj_id, pitch, roll, yaw_norm)
        for oid in members:
            obj = blender_obj_index.get(oid)
            if obj is None:
                continue
            rot = obj.get("rotation_euler")
            if not rot or len(rot) < 3:
                continue
            pitch, roll, raw_yaw = rot[0], rot[1], rot[2]
            yaw_norm = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))
            group_yaws.append((oid, pitch, roll, yaw_norm))
            processed_ceiling.add(oid)

        if not group_yaws:
            continue

        # Compute median yaw and snap to 90° grid.
        median_yaw = statistics.median(y for _, _, _, y in group_yaws)
        snapped = _snap_90(median_yaw)
        median_deg = math.degrees(median_yaw)
        snapped_deg = math.degrees(snapped)

        for oid, pitch, roll, yaw_norm in group_yaws:
            delta_raw = snapped - yaw_norm
            delta = math.atan2(math.sin(delta_raw), math.cos(delta_raw))
            if abs(delta) <= 1e-4:
                n_aligned += 1
                continue

            orig_deg = math.degrees(yaw_norm)
            delta_deg = math.degrees(delta)
            sign = "+" if delta_deg >= 0 else ""

            ops.append({
                "action": "update_rotation",
                "obj_name": oid,
                "rotation_euler": [pitch, roll, snapped],
                "reason": (
                    f"ceiling mesh_group '{group_name}' median yaw "
                    f"{median_deg:+.2f}° → snapped to 90° grid: {snapped_deg:.0f}° "
                    f"(this instance original yaw {orig_deg:.2f}°, Δ={sign}{delta_deg:.2f}°)"
                ),
                "criteria_used": ["ceiling_meshgroup_median_90deg_snap"],
                "priority": 1,
                "confidence": 0.9,
                "requires_planner_review": False,
                "source": "heuristic",
            })
            n_ceil_emitted += 1

    # Orphan ceiling objects (class matches CEILING_CLASSES but not in any mesh_group):
    # snap their own yaw to 90° grid as a single-instance group.
    for obj in blender_scene.get("objects", []):
        oid: str = obj.get("id", "")
        if not oid.startswith("obj_"):
            continue
        if oid in processed_ceiling:
            continue
        cls = _obj_class(oid)
        if not cls or not _class_matches(cls, CEILING_CLASSES):
            continue
        # Ceiling orphan — single-instance 90° snap.
        processed_ceiling.add(oid)
        n_ceil_meshgroups += 1
        rot = obj.get("rotation_euler")
        if not rot or len(rot) < 3:
            continue
        pitch, roll, raw_yaw = rot[0], rot[1], rot[2]
        yaw_norm = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))
        snapped = _snap_90(yaw_norm)
        delta_raw = snapped - yaw_norm
        delta = math.atan2(math.sin(delta_raw), math.cos(delta_raw))
        if abs(delta) <= 1e-4:
            n_aligned += 1
            continue
        orig_deg = math.degrees(yaw_norm)
        snapped_deg = math.degrees(snapped)
        delta_deg = math.degrees(delta)
        sign = "+" if delta_deg >= 0 else ""
        ops.append({
            "action": "update_rotation",
            "obj_name": oid,
            "rotation_euler": [pitch, roll, snapped],
            "reason": (
                f"ceiling mesh_group '(singleton)' median yaw "
                f"{orig_deg:+.2f}° → snapped to 90° grid: {snapped_deg:.0f}° "
                f"(this instance original yaw {orig_deg:.2f}°, Δ={sign}{delta_deg:.2f}°)"
            ),
            "criteria_used": ["ceiling_meshgroup_median_90deg_snap"],
            "priority": 1,
            "confidence": 0.9,
            "requires_planner_review": False,
            "source": "heuristic",
        })
        n_ceil_emitted += 1

    # -------------------------------------------------------------------
    # Floor pass: per-instance 45° snap (unchanged).
    # Skip ceiling and wall objects.
    # -------------------------------------------------------------------
    for obj in blender_scene.get("objects", []):
        obj_id: str = obj.get("id", "")
        if not obj_id.startswith("obj_"):
            continue  # skip Floor / Wall_NN / Ceiling

        # Class-based exclusions
        cls = _obj_class(obj_id)
        if cls:
            if _class_matches(cls, CEILING_CLASSES):
                n_skipped_class += 1
                continue
            if _class_matches(cls, WALL_CLASSES):
                n_skipped_class += 1
                continue

        # Relation-graph exclusion: members of seated_around / on_top_of groups
        # must NOT be snapped to a global 45° grid — their target yaw is
        # anchor-relative (face toward anchor). See module docstring.
        if obj_id in relation_excluded:
            n_skipped_relation += 1
            continue

        rotation = obj.get("rotation_euler")
        if not rotation or len(rotation) < 3:
            continue

        pitch: float = rotation[0]
        roll: float  = rotation[1]
        yaw: float   = rotation[2]

        # Normalise yaw to [-π, π]
        yaw_norm = math.atan2(math.sin(yaw), math.cos(yaw))

        # Snap to nearest 45° grid
        snapped_raw = round(yaw_norm / step_45) * step_45
        # Re-normalise snapped value into [-π, π]
        snapped = math.atan2(math.sin(snapped_raw), math.cos(snapped_raw))

        # Delta (also normalised to [-π, π] for a stable sign)
        delta_raw = snapped - yaw_norm
        delta = math.atan2(math.sin(delta_raw), math.cos(delta_raw))

        if abs(delta) <= 1e-4:
            n_aligned += 1
            continue

        orig_deg   = math.degrees(yaw_norm)
        snapped_deg = math.degrees(snapped)
        delta_deg  = math.degrees(delta)
        sign       = "+" if delta_deg >= 0 else ""

        ops.append({
            "action": "update_rotation",
            "obj_name": obj_id,
            "rotation_euler": [pitch, roll, snapped],
            "reason": (
                f"yaw snap to 45° grid: {orig_deg:.2f}° → {snapped_deg:.2f}° "
                f"(Δ={sign}{delta_deg:.2f}°)"
            ),
            "criteria_used": ["yaw_45deg_snap"],
            "priority": 1,
            "confidence": 0.9,
            "requires_planner_review": False,
            "source": "heuristic",
        })
        n_emitted += 1

    summary = {
        "yaw_snap_emitted": n_emitted,
        "yaw_snap_skipped_already_aligned": n_aligned,
        "yaw_snap_skipped_wall_or_ceiling": n_skipped_class,
        "yaw_snap_skipped_relation_member": n_skipped_relation,
        "yaw_snap_ceiling_emitted": n_ceil_emitted,
        "yaw_snap_ceiling_meshgroups": n_ceil_meshgroups,
    }
    return ops, summary


# ---------------------------------------------------------------------------
# Priority-1: polygon BEV containment clamp
# ---------------------------------------------------------------------------

def _build_blend_info_dims(blend_info: dict | None) -> dict[str, list[float]]:
    """Build {obj_id: [dim_x, dim_y, dim_z]} from blend_info categories.

    Hierarchy in blend_info:
        obj_NN (EMPTY, categories['objects'])
            └─ world.NNN (EMPTY, categories['world'])
                   └─ geometry_0.NNN (MESH, categories['geometry_meshes'], has dimensions)

    The 'world' intermediate nodes have scale=1 and location=[0,0,0], so the
    geometry 'dimensions' (in local space of the geometry node) are equal to
    the geometry's axis-aligned extent in the obj_NN's local frame.  The
    actual world footprint is then geometry_dim * obj_scale.
    """
    if blend_info is None:
        return {}

    cats: dict = blend_info.get("categories", {})

    # Map world-node-name → obj_id  (e.g. "world.018" → "obj_19")
    world_to_obj: dict[str, str] = {}
    for w in cats.get("world", []):
        parent = w.get("parent", "")
        if parent.startswith("obj_"):
            world_to_obj[w["name"]] = parent

    # Map obj_id → geometry dimensions
    obj_dims: dict[str, list[float]] = {}
    for gm in cats.get("geometry_meshes", []):
        parent_world = gm.get("parent", "")
        obj_id = world_to_obj.get(parent_world)
        if obj_id:
            dims = gm.get("dimensions")
            if dims and len(dims) >= 2:
                obj_dims[obj_id] = dims

    return obj_dims


def _obb_xy_corners(
    loc_xy: tuple[float, float],
    yaw_rad: float,
    half_dx: float,
    half_dy: float,
) -> list[tuple[float, float]]:
    """Return the 4 corners of a yaw-rotated bbox in world xy plane (OBB).

    Args:
        loc_xy:  (x, y) world-space centre of the object.
        yaw_rad: Rotation around the Z axis (Blender rotation_euler[2]).
        half_dx: Half-extent along the object's local X axis (dim_x * scale / 2).
        half_dy: Half-extent along the object's local Y axis (dim_y * scale / 2).

    Returns:
        List of 4 (wx, wy) world-space corner points.
    """
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    local: list[tuple[float, float]] = [
        (-half_dx, -half_dy),
        ( half_dx, -half_dy),
        ( half_dx,  half_dy),
        (-half_dx,  half_dy),
    ]
    return [
        (loc_xy[0] + lx * cos_y - ly * sin_y,
         loc_xy[1] + lx * sin_y + ly * cos_y)
        for lx, ly in local
    ]


def _polygon_clamp_ops(
    blender_scene: dict,
    polygon: dict,
    blend_info: dict | None,
    object_class: dict | None,
    margin: float = 0.05,
    object_state: dict | None = None,
) -> tuple[list[dict], dict]:
    """Emit update_layout ops that push obj_* footprints inside the room polygon.

    Algorithm (per object):
        1. Derive xy OBB (oriented bounding box) from blend_info geometry
           dimensions × scale × yaw rotation.  Falls back to location ± 0.3 m
           if dims are unavailable.
        2. For each polygon edge compute the inward unit normal (toward centroid).
        3. Find the minimum signed distance (most-outward OBB corner) for that edge.
        4. If min_sd < margin, accumulate a push of (margin − min_sd) along the
           inward normal.
        5. Sum pushes from all non-skipped violated edges; if total is nonzero,
           emit one update_layout op translating the object by (Δx, Δy) while
           preserving z.

    Wall-mounted objects (WALL_CLASSES match OR object_state attached_to includes
    'wall') are now processed but their ATTACHED EDGE is skipped:
        - Compute perpendicular distance from object origin to each polygon edge.
        - The closest edge is the "attached edge" and is excluded from violation
          checks (to avoid pushing the object off the wall).
        - The remaining edges are checked normally.

    Non-wall objects: all edges are checked as before.

    Returns (ops, summary_counts).
    """
    ops: list[dict] = []
    n_emitted = 0
    n_walls_processed = 0
    n_attached_edges_skipped = 0

    # -- polygon geometry --
    verts: list[list[float]] = polygon.get("polygon_vertices", [])
    if len(verts) < 3:
        print(
            "[heuristic_planner] WARNING: polygon_v2.json has fewer than 3 vertices — "
            "skipping polygon clamp.",
            file=sys.stderr,
        )
        return ops, {
            "polygon_clamp_emitted": 0,
            "polygon_clamp_walls_processed": 0,
            "polygon_clamp_attached_edges_skipped": 0,
        }

    centroid_xy: list[float] = polygon.get("polygon_centroid_xy", [0.0, 0.0])
    cx, cy = centroid_xy[0], centroid_xy[1]
    n_edges = len(verts)

    # Wall edge index → wall name (for reason string), sourced from polygon_v2.json
    wall_edges_meta: list[dict] = polygon.get("wall_edges", [])
    edge_name: dict[int, str] = {}
    for we in wall_edges_meta:
        fi = we.get("from")
        if fi is not None:
            edge_name[fi] = we.get("object", f"Edge_{fi}")

    def _inward_normal(ei: int) -> tuple[float, float, float, float]:
        """Return (nx, ny, ax, ay) — unit inward normal + edge start point."""
        ax, ay = verts[ei]
        bx, by = verts[(ei + 1) % n_edges]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return 0.0, 0.0, ax, ay
        nx, ny = -dy / length, dx / length
        # Ensure the normal faces inward (toward centroid)
        mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
        if (cx - mx) * nx + (cy - my) * ny < 0:
            nx, ny = -nx, -ny
        return nx, ny, ax, ay

    # Pre-compute inward normals once.
    inward: list[tuple[float, float, float, float]] = [_inward_normal(ei) for ei in range(n_edges)]

    def _sd_fast(px: float, py: float, ei: int) -> float:
        """Signed distance of point (px,py) from edge ei (positive = inside room)."""
        nx, ny, ax, ay = inward[ei]
        return (px - ax) * nx + (py - ay) * ny

    # -- blend_info geometry dims --
    obj_dims: dict[str, list[float]] = _build_blend_info_dims(blend_info)

    # -- class lookup (label_id == numeric suffix of obj_NN) --
    label_to_class: dict[str, str] = object_class if isinstance(object_class, dict) else {}

    def _obj_class(obj_id: str) -> str:
        suffix = obj_id[len("obj_"):]
        return label_to_class.get(suffix, "")

    # -- Build wall-attachment set from object_state (attached_to includes 'wall') --
    wall_attached_obj_ids: set[str] = set()
    if object_state is not None:
        for obj_entry in object_state.get("objects", []):
            eid = obj_entry.get("obj_id", "")
            if "wall" in obj_entry.get("attached_to", []):
                wall_attached_obj_ids.add(eid)

    def _is_wall_mounted(obj_id: str, cls: str) -> bool:
        """Return True if object is wall-mounted (WALL_CLASSES match OR object_state says so)."""
        if obj_id in wall_attached_obj_ids:
            return True
        if cls and _class_matches(cls, WALL_CLASSES):
            return True
        return False

    def _closest_edge_idx(ox: float, oy: float) -> int:
        """Return the edge index with minimum signed distance to (ox, oy) — i.e. the nearest wall."""
        min_sd = math.inf
        closest = 0
        for ei in range(n_edges):
            sd = _sd_fast(ox, oy, ei)
            if sd < min_sd:
                min_sd = sd
                closest = ei
        return closest

    # -- main loop --
    for obj in blender_scene.get("objects", []):
        obj_id: str = obj.get("id", "")
        if not obj_id.startswith("obj_"):
            continue

        cls = _obj_class(obj_id) or obj.get("class", "")
        is_wall = _is_wall_mounted(obj_id, cls)

        loc: list[float] = obj.get("location", [0.0, 0.0, 0.0])
        lx, ly, lz = loc[0], loc[1], loc[2]

        # Compute half-extents in x and y (object local frame).
        scale_val = obj.get("scale", [1.0, 1.0, 1.0])
        s = abs(scale_val[0]) if isinstance(scale_val, (list, tuple)) and scale_val else 1.0

        # Read yaw from rotation_euler[2]; pitch/roll ignored for xy footprint.
        rot = obj.get("rotation_euler", [0.0, 0.0, 0.0])
        yaw = rot[2] if isinstance(rot, (list, tuple)) and len(rot) >= 3 else 0.0

        dims = obj_dims.get(obj_id)
        if dims and (dims[0] > 0 or dims[1] > 0):
            hx = dims[0] * s * 0.5
            hy = dims[1] * s * 0.5
            use_obb = True
        else:
            # Fallback: treat object as a 0.6 m × 0.6 m footprint square.
            hx = hy = 0.3
            use_obb = False
            print(
                f"[heuristic_planner] WARNING: no geometry dims for {obj_id} "
                f"({cls or '?'}) — using fallback radius 0.3 m for polygon clamp.",
                file=sys.stderr,
            )

        # Build 4 OBB corners (or axis-aligned fallback treated as OBB with yaw=0).
        if use_obb:
            corners: list[tuple[float, float]] = _obb_xy_corners((lx, ly), yaw, hx, hy)
        else:
            # Fallback: axis-aligned 0.6 m square (same as old AABB behavior).
            corners = _obb_xy_corners((lx, ly), 0.0, hx, hy)

        # Determine which edges to skip (attached edge for wall-mounted objects).
        skip_edges: set[int] = set()
        attached_edge_name: str = ""
        if is_wall:
            n_walls_processed += 1
            attached_ei = _closest_edge_idx(lx, ly)
            skip_edges.add(attached_ei)
            attached_edge_name = edge_name.get(attached_ei, f"Edge_{attached_ei}")
            n_attached_edges_skipped += 1

        # Accumulate push vectors from all non-skipped violated edges.
        total_dx = 0.0
        total_dy = 0.0
        violated_wall_names: list[str] = []
        worst_corner_info: tuple[float, float, float] | None = None  # (wx, wy, depth)

        for ei in range(n_edges):
            if ei in skip_edges:
                continue
            nx, ny, _, _ = inward[ei]
            corner_sds = [(_sd_fast(px, py, ei), px, py) for px, py in corners]
            min_sd, min_px, min_py = min(corner_sds, key=lambda t: t[0])
            if min_sd < margin:
                depth = margin - min_sd
                push = depth
                total_dx += push * nx
                total_dy += push * ny
                wname = edge_name.get(ei, f"Edge_{ei}")
                violated_wall_names.append(wname)
                # Track the worst violating corner for the reason string.
                if worst_corner_info is None or depth > worst_corner_info[2]:
                    worst_corner_info = (min_px, min_py, depth)

        if not violated_wall_names:
            continue  # fully inside with margin — nothing to do

        new_x = lx + total_dx
        new_y = ly + total_dy

        # Summarise the push for the reason string.
        push_dist = math.hypot(total_dx, total_dy)
        dx_sign = "+" if total_dx >= 0 else ""
        dy_sign = "+" if total_dy >= 0 else ""
        walls_str = ", ".join(violated_wall_names)

        if is_wall:
            wcx, wcy, wdepth = worst_corner_info or (lx, ly, 0.0)
            reason = (
                f"polygon clamp (OBB): wall-mounted, attached_edge={attached_edge_name} (skipped); "
                f"pushed {dx_sign}{total_dx:.3f}m /x {dy_sign}{total_dy:.3f}m /y "
                f"(total {push_dist:.3f}m) to clear {walls_str} "
                f"(violator OBB corner xy=({wcx:.3f},{wcy:.3f}), depth={wdepth:.3f}m)"
            )
        else:
            reason = (
                f"polygon clamp (OBB): pushed {dx_sign}{total_dx:.3f}m /x "
                f"{dy_sign}{total_dy:.3f}m /y "
                f"(total {push_dist:.3f}m) to stay margin>={margin:.2f}m inside "
                f"polygon edges ({walls_str})"
            )

        ops.append({
            "action": "update_layout",
            "obj_name": obj_id,
            "location": [new_x, new_y, lz],
            "reason": reason,
            "criteria_used": ["polygon_clamp", "obb"],
            "priority": 6,
            "confidence": 0.95,
            "requires_planner_review": False,
            "source": "polygon_clamp",
        })
        n_emitted += 1

    summary = {
        "polygon_clamp_emitted": n_emitted,
        "polygon_clamp_walls_processed": n_walls_processed,
        "polygon_clamp_attached_edges_skipped": n_attached_edges_skipped,
    }
    return ops, summary


# ---------------------------------------------------------------------------
# Sorting helpers
# ---------------------------------------------------------------------------

def _obj_sort_key(op: dict) -> str:
    """Sort key: prefer moving_obj for attach ops, obj_name for update_size."""
    return op.get("moving_obj") or op.get("obj_name") or ""


def _sort_ops(ops: list[dict]) -> list[dict]:
    """Sort by (priority, obj_name alphabetically)."""
    return sorted(ops, key=lambda op: (op["priority"], _obj_sort_key(op)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scene_dir: Path, output_path: Path) -> None:
    # ---- Load input files ----
    merge_plan = _load_json(scene_dir / "inputs" / "merge_plan.json", "merge_plan.json")
    blender_scene = _load_json(scene_dir / "json" / "blender_scene.json", "blender_scene.json")
    object_state = _load_json(scene_dir / "json" / "object_state.json", "object_state.json")
    relation_graph = _load_json(scene_dir / "inputs" / "relation_graph.json", "relation_graph.json")
    object_class = _load_json(scene_dir / "inputs" / "object_class.json", "object_class.json")
    polygon_v2 = _load_json(scene_dir / "json" / "polygon_v2.json", "polygon_v2.json")
    blend_info = _load_json(_resolve_blend_info_path(scene_dir), "blend_info.json")

    # ---- Resolve Stage-1 candidate mesh groups (in-memory, non-destructive) ----
    # Stage 1 emits uncertain `candidate_mesh_groups`; promote instances whose 3D
    # bbox dimensions (json/blend_info.json) match the canonical's, merging the
    # confirmed groups into the IN-MEMORY merge_plan so they flow into the existing
    # priority-1 scale-normalization. We never write back to inputs/merge_plan.json.
    if (
        merge_plan is not None
        and merge_plan.get("candidate_mesh_groups")
        and _resolve_candidate_groups is not None
        and (scene_dir / "json" / "blend_info.json").exists()
    ):
        try:
            resolved = _resolve_candidate_groups(scene_dir)
            confirmed = resolved.get("confirmed_mesh_groups", {})
            if confirmed:
                merge_plan = _merge_confirmed_into(merge_plan, resolved)
                print(
                    f"[heuristic_planner] INFO: promoted {len(confirmed)} confirmed "
                    f"candidate mesh group(s) into in-memory merge_plan "
                    f"(see json/resolved_mesh_groups.json).",
                    file=sys.stderr,
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            print(
                f"[heuristic_planner] WARNING: candidate-mesh-group resolution failed "
                f"({exc}) — proceeding with original merge_plan.",
                file=sys.stderr,
            )

    # ---- Build indices ----
    blender_index: dict[str, dict] = _build_blender_index(blender_scene) if blender_scene else {}

    # ---- Priority-4 pass first (to know which objects to suppress in priority-3) ----
    p4_ops: list[dict] = []
    support_covered: set[str] = set()
    if relation_graph is not None:
        if "groups" not in relation_graph:
            print(
                "[heuristic_planner] WARNING: relation_graph.json has no 'groups' key — skipping priority-4 pass.",
                file=sys.stderr,
            )
        else:
            p4_ops, support_covered = _support_attach_ops(relation_graph, floor_covered=set())

    # ---- Priority-1 pass ----
    p1_ops: list[dict] = []
    if merge_plan is not None:
        if "mesh_groups" not in merge_plan:
            print(
                "[heuristic_planner] WARNING: merge_plan.json has no 'mesh_groups' key — skipping priority-1 pass.",
                file=sys.stderr,
            )
        elif blender_scene is None:
            print(
                "[heuristic_planner] WARNING: blender_scene.json unavailable — cannot compute mesh-group scales.",
                file=sys.stderr,
            )
        else:
            p1_ops = _mesh_group_scale_ops(merge_plan, blender_index)

    # ---- Priority-1 yaw snap pass (DISABLED) ----
    # The yaw snap snapped each floor object's yaw to the WORLD 45° grid
    # (0/45/90/...). But rooms are reconstructed as ORIENTED rectangles
    # (floor_plan rect_angle is rarely 0), and GALP already places furniture
    # aligned to that rotated room axis. Snapping to the world grid therefore
    # rotated furniture ~rect_angle AWAY from the walls, while attach_to_wall
    # (which uses the true wall tangent) kept windows/frames at the room angle —
    # producing a within-stage contradiction (bed at 0° vs walls at ~20°).
    # Removed entirely: furniture keeps its GALP yaw, which is already wall-aligned.
    # _yaw_snap_ops() is retained above for reference but no longer invoked.
    yaw_ops: list[dict] = []
    yaw_summary: dict = {}

    # ---- Priority-3 pass ----
    p3_ops: list[dict] = []
    if object_state is not None:
        p3_ops = _env_attach_ops(
            object_state,
            support_covered=support_covered,
            object_class=object_class,
            relation_graph=relation_graph,
            scene_dir=scene_dir,
        )

    # ---- Polygon BEV containment clamp pass (DISABLED) ----
    # Superseded by polygon_vertex_clamp.py which uses real mesh vertices
    # (not OBB approximation) and is invoked separately via merge_ops
    # --polygon-clamp-ops.  The function definition is kept below for
    # reference; it produces 0 ops in the normal pipeline.
    if False:  # noqa: SIM210 — intentional permanent disable
        poly_clamp_ops, poly_clamp_summary = _polygon_clamp_ops(  # type: ignore[assignment]
            blender_scene, polygon_v2, blend_info, object_class,
            object_state=object_state,
        )
    poly_clamp_ops: list[dict] = []
    poly_clamp_summary: dict = {
        "polygon_clamp_emitted": 0,
        "polygon_clamp_walls_processed": 0,
        "polygon_clamp_attached_edges_skipped": 0,
    }

    # ---- Merge yaw snap ops into priority-1 bucket ----
    p1_ops = p1_ops + yaw_ops + poly_clamp_ops

    # ---- Sort each priority bucket alphabetically by object name ----
    p1_ops = sorted(p1_ops, key=lambda op: _obj_sort_key(op))
    p3_ops = sorted(p3_ops, key=lambda op: _obj_sort_key(op))
    p4_ops = sorted(p4_ops, key=lambda op: _obj_sort_key(op))

    all_ops: list[dict] = p1_ops + p3_ops + p4_ops

    # ---- Covered objects ----
    covered: set[str] = set()
    for op in all_ops:
        obj = op.get("moving_obj") or op.get("obj_name")
        if obj:
            covered.add(obj)

    # ---- Stats ----
    stats = {
        "mesh_group_scale_ops": sum(
            1 for op in p1_ops if "mesh_group_same_scale" in op.get("criteria_used", [])
        ),
        "yaw_snap_ops": yaw_summary.get("yaw_snap_emitted", 0),
        "yaw_snap_ceiling_ops": yaw_summary.get("yaw_snap_ceiling_emitted", 0),
        "yaw_snap_ceiling_meshgroups": yaw_summary.get("yaw_snap_ceiling_meshgroups", 0),
        "floor_attach_ops": sum(1 for op in p3_ops if op.get("anchor_obj") == "Floor"),
        "ceiling_attach_ops": sum(1 for op in p3_ops if op.get("anchor_obj") == "Ceiling"),
        "support_attach_ops": len(p4_ops),
        "polygon_clamp_ops": poly_clamp_summary.get("polygon_clamp_emitted", 0),
        "polygon_clamp_walls_processed": poly_clamp_summary.get("polygon_clamp_walls_processed", 0),
        "polygon_clamp_attached_edges_skipped": poly_clamp_summary.get("polygon_clamp_attached_edges_skipped", 0),
    }

    # ---- Build output ----
    result = {
        "operation_list": all_ops,
        "covered_objects": sorted(covered),
        "heuristic_stats": stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    total = len(all_ops)
    print(
        f"[heuristic_planner] Done. {total} ops emitted "
        f"(scale={stats['mesh_group_scale_ops']}, "
        f"yaw_snap={stats['yaw_snap_ops']}, "
        f"ceil_yaw_snap={stats['yaw_snap_ceiling_ops']}, "
        f"floor={stats['floor_attach_ops']}, "
        f"ceiling={stats['ceiling_attach_ops']}, "
        f"support={stats['support_attach_ops']}, "
        f"poly_clamp=0 (disabled — see polygon_vertex_clamp.py)). "
        f"Covered {len(covered)} objects. "
        f"Output → {output_path}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heuristic pre-pass: emit deterministic scene ops without an LLM."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=Path,
        help="Root scene directory (contains inputs/ and json/ sub-folders).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <scene_dir>/json/heuristic_ops.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scene_dir: Path = args.scene_dir.resolve()
    output_path: Path = (
        args.output.resolve() if args.output else scene_dir / "json" / "heuristic_ops.json"
    )
    run(scene_dir, output_path)
