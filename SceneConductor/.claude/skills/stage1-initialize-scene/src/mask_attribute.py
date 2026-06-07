#!/usr/bin/env python3
"""
mask_attribute.py — Maintains <scene_dir>/mask_attribute.json.

Public API (import as a library; no CLI):
    init_attributes(scene_dir)
    record_merge(scene_dir, merge_plan)
    record_remask(scene_dir, remask_plan, new_ids)
    apply_mesh_groups(scene_dir, mesh_groups)

Schema:
{
  "objects": {
    "1": {"class": "chair", "mesh_group": null, "canonical": true,
          "source": "grounded_sam", "bbox_xy": [x0,y0,x1,y1], "area_px": N}
  },
  "mesh_groups": {
    "chair_A": {"canonical_id": 1, "instance_ids": [1,2], "class": "chair"}
  },
  "candidate_mesh_groups": [
    {"canonical_id": 11, "instance_ids": [11,15], "class": "chair",
     "reason": "...", "risk": "low"}
  ],
  "history": [{"step": "grounded_sam", "timestamp": "..."}]
}

mesh_groups are HIGH-CONFIDENCE commits (run_sam3d dedups them → shared GLB).
candidate_mesh_groups are GENEROUS hypotheses only — run_sam3d ignores them, so
each instance keeps its own mesh; Stage 3 resolves them via 3D bbox dimensions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def _attr_path(scene_dir: Path) -> Path:
    return Path(scene_dir) / "mask_attribute.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(scene_dir: Path) -> dict[str, Any]:
    p = _attr_path(scene_dir)
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"objects": {}, "mesh_groups": {}, "candidate_mesh_groups": [], "history": []}


def _save(scene_dir: Path, data: dict[str, Any]) -> None:
    p = _attr_path(scene_dir)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _mask_bbox_and_area(mask_png: Path, img_w: int, img_h: int) -> tuple[list[float], int]:
    """Return normalised bbox_xy [x0,y0,x1,y1] and area in pixels."""
    arr = np.array(Image.open(mask_png).convert("L"))
    binary = arr > 0
    area = int(binary.sum())
    if area == 0:
        return [0.0, 0.0, 0.0, 0.0], 0
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    bbox = [cmin / img_w, rmin / img_h, cmax / img_w, rmax / img_h]
    return bbox, area


def _load_object_class(scene_dir: Path) -> dict[str, str]:
    """Return {mask_id_str: class_name} from object_class.json."""
    oc_path = Path(scene_dir) / "object_class.json"
    if not oc_path.exists():
        return {}
    with open(oc_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Support two formats: {"1": "chair", ...} or {"objects": {"1": "chair"}}
    if "objects" in data:
        return {str(k): v for k, v in data["objects"].items()}
    return {str(k): v for k, v in data.items()}


# ── public API ────────────────────────────────────────────────────────────────

def init_attributes(scene_dir: str | Path) -> None:
    """
    Called once after GroundedSAM. Populates 'objects' from:
      - masks/1.png .. N.png
      - object_class.json  (maps mask id -> class name)

    Computes bbox_xy (normalised) and area_px for each mask.
    Appends to history. Merge-safe: if mask_attribute.json already exists,
    updates only objects not yet present.
    """
    scene_dir = Path(scene_dir)
    masks_dir = scene_dir / "masks"

    # Collect PNG files named as integers
    mask_files: list[Path] = sorted(
        [f for f in masks_dir.glob("*.png") if f.stem.isdigit()],
        key=lambda f: int(f.stem),
    )
    if not mask_files:
        raise FileNotFoundError(f"No integer-named PNGs found in {masks_dir}")

    # Determine image dimensions from first mask
    first_arr = np.array(Image.open(mask_files[0]).convert("L"))
    img_h, img_w = first_arr.shape

    class_map = _load_object_class(scene_dir)
    data = _load(scene_dir)
    data.setdefault("candidate_mesh_groups", [])

    for mf in mask_files:
        mid = mf.stem  # e.g. "1"
        if mid in data["objects"]:
            continue  # already recorded
        bbox, area = _mask_bbox_and_area(mf, img_w, img_h)
        data["objects"][mid] = {
            "class": class_map.get(mid, "unknown"),
            "mesh_group": None,
            "canonical": True,
            "source": "grounded_sam",
            "bbox_xy": bbox,
            "area_px": area,
        }

    data["history"].append({"step": "grounded_sam", "timestamp": _now()})
    _save(scene_dir, data)


def record_merge(scene_dir: str | Path, merge_plan: dict[str, Any]) -> None:
    """
    Called from merge_masks.py after masks have been physically merged/deleted/renumbered.
    Updates objects dict to reflect the new id mapping embedded in merge_plan["id_remap"],
    and removes absorbed/deleted ids.

    merge_plan should contain:
      "merge_groups": [{"keep_id": N, "absorb_ids": [...], "reason": "..."}]
      "delete_ids": [...]
      "mesh_groups": {...}
      "id_remap": {"old_id": "new_id", ...}   (added by merge_masks.py)
    """
    scene_dir = Path(scene_dir)
    data = _load(scene_dir)

    absorbed_ids: set[str] = set()
    for mg in merge_plan.get("merge_groups", []):
        for aid in mg.get("absorb_ids", []):
            absorbed_ids.add(str(aid))

    deleted_ids: set[str] = {str(i) for i in merge_plan.get("delete_ids", [])}
    remove_ids = absorbed_ids | deleted_ids

    # Drop absorbed/deleted objects
    for rid in remove_ids:
        data["objects"].pop(rid, None)

    # Apply id renumbering
    id_remap: dict[str, str] = merge_plan.get("id_remap", {})
    if id_remap:
        new_objects: dict[str, Any] = {}
        for old_id, obj in data["objects"].items():
            new_id = id_remap.get(old_id, old_id)
            new_objects[new_id] = obj
        data["objects"] = new_objects

    # Apply mesh_groups if present — REMAP IDs through id_remap since the
    # Mask-Evaluator wrote the plan against pre-merge IDs but masks were renumbered.
    mesh_groups = merge_plan.get("mesh_groups", {})
    if mesh_groups:
        if id_remap:
            remapped: dict[str, Any] = {}
            for grp_name, ginfo in mesh_groups.items():
                old_canonical = str(ginfo["canonical_id"])
                old_instances = [str(i) for i in ginfo.get("instance_ids", [old_canonical])]
                # Drop any IDs that were absorbed/deleted (not in id_remap)
                new_canonical_str = id_remap.get(old_canonical)
                new_instances_str = [
                    id_remap[i] for i in old_instances if i in id_remap
                ]
                if new_canonical_str is None and new_instances_str:
                    # Canonical got absorbed — promote first surviving instance
                    new_canonical_str = new_instances_str[0]
                if not new_canonical_str:
                    continue  # whole group disappeared
                remapped[grp_name] = {
                    "canonical_id": int(new_canonical_str),
                    "instance_ids": [int(i) for i in new_instances_str],
                    "class": ginfo.get("class", ""),
                }
            mesh_groups = remapped
        _apply_mesh_groups_inplace(data, mesh_groups)

    # Persist candidate_mesh_groups — GENEROUS hypotheses that do NOT cause mesh
    # sharing (run_sam3d ignores them). Remap canonical_id + instance_ids through
    # the SAME id_remap used for mesh_groups so post-merge ids stay consistent;
    # drop ids that were absorbed/deleted. The evaluator's latest plan is
    # authoritative, so overwrite rather than append.
    candidate_mesh_groups = merge_plan.get("candidate_mesh_groups", [])
    remapped_candidates: list[dict[str, Any]] = []
    for cand in candidate_mesh_groups:
        old_canonical = str(cand["canonical_id"])
        old_instances = [str(i) for i in cand.get("instance_ids", [old_canonical])]
        if id_remap:
            # Drop any IDs that were absorbed/deleted (absent from id_remap)
            new_canonical_str = id_remap.get(old_canonical)
            new_instances_str = [id_remap[i] for i in old_instances if i in id_remap]
            if new_canonical_str is None and new_instances_str:
                # Canonical got absorbed — promote first surviving instance
                new_canonical_str = new_instances_str[0]
            if not new_canonical_str:
                continue  # whole candidate group disappeared
        else:
            new_canonical_str = old_canonical
            new_instances_str = old_instances
        entry = dict(cand)  # preserve all fields (class, reason, risk, ...)
        entry["canonical_id"] = int(new_canonical_str)
        entry["instance_ids"] = [int(i) for i in new_instances_str]
        remapped_candidates.append(entry)
    data["candidate_mesh_groups"] = remapped_candidates

    data["history"].append({
        "step": "merge",
        "plan": merge_plan,
        "timestamp": _now(),
    })
    _save(scene_dir, data)


def record_remask(
    scene_dir: str | Path,
    remask_plan: dict[str, Any],
    new_ids: list[int],
) -> None:
    """
    Called from remask_region.py after new masks have been appended.
    new_ids: list of new integer mask IDs assigned (e.g. [4, 5]).
    """
    scene_dir = Path(scene_dir)
    masks_dir = scene_dir / "masks"

    # Determine image dimensions
    any_mask = next(
        (f for f in masks_dir.glob("*.png") if f.stem.isdigit()), None
    )
    img_w, img_h = 1, 1
    if any_mask:
        arr = np.array(Image.open(any_mask).convert("L"))
        img_h, img_w = arr.shape

    data = _load(scene_dir)

    new_objects_in_plan = remask_plan.get("new_objects", [])
    for idx, obj_spec in zip(new_ids, new_objects_in_plan):
        mid = str(idx)
        mask_path = masks_dir / f"{idx}.png"
        bbox: list[float] = [0.0, 0.0, 0.0, 0.0]
        area: int = 0
        if mask_path.exists():
            bbox, area = _mask_bbox_and_area(mask_path, img_w, img_h)

        source = "remask_bbox" if "bbox_xy" in obj_spec else "remask_point"
        data["objects"][mid] = {
            "class": obj_spec.get("class", "unknown"),
            "mesh_group": None,
            "canonical": True,
            "source": source,
            "bbox_xy": bbox,
            "area_px": area,
        }

    data["history"].append({
        "step": "remask",
        "plan": remask_plan,
        "new_ids": new_ids,
        "timestamp": _now(),
    })
    _save(scene_dir, data)


def apply_mesh_groups(scene_dir: str | Path, mesh_groups: dict[str, Any]) -> None:
    """
    Sets mesh_group + canonical flags on objects according to mesh_groups dict.

    mesh_groups format:
      {"chair_A": {"canonical_id": 1, "instance_ids": [1, 2], "class": "chair"}}
    """
    scene_dir = Path(scene_dir)
    data = _load(scene_dir)
    _apply_mesh_groups_inplace(data, mesh_groups)
    data["history"].append({
        "step": "apply_mesh_groups",
        "mesh_groups": mesh_groups,
        "timestamp": _now(),
    })
    _save(scene_dir, data)


# ── private helpers ───────────────────────────────────────────────────────────

def _apply_mesh_groups_inplace(data: dict[str, Any], mesh_groups: dict[str, Any]) -> None:
    """Mutates data in place — updates objects and mesh_groups dicts."""
    for group_name, ginfo in mesh_groups.items():
        canonical_id = str(ginfo["canonical_id"])
        instance_ids = [str(i) for i in ginfo.get("instance_ids", [canonical_id])]
        grp_class = ginfo.get("class", "")

        for iid in instance_ids:
            if iid in data["objects"]:
                data["objects"][iid]["mesh_group"] = group_name
                data["objects"][iid]["canonical"] = (iid == canonical_id)
                if not grp_class:
                    grp_class = data["objects"][iid].get("class", "")

        data["mesh_groups"][group_name] = {
            "canonical_id": int(canonical_id),
            "instance_ids": [int(i) for i in instance_ids],
            "class": grp_class,
        }
