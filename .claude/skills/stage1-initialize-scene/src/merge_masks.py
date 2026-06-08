#!/usr/bin/env python3
"""
merge_masks.py — Apply a Mask-Evaluator merge plan to a scene's masks directory.

No LLM calls. Pure PIL/numpy/json.

Merge plan schema:
{
  "merge_groups": [
    {"keep_id": 3, "absorb_ids": [4, 5], "reason": "sofa cushions split too finely"}
  ],
  "delete_ids": [7],
  "mesh_groups": {
    "chair_A": {"canonical_id": 1, "instance_ids": [1, 2, 6]},
    "table":   {"canonical_id": 8, "instance_ids": [8]}
  }
}

Side-effects:
  - masks/<keep_id>.png  updated to union of keep + absorbed masks
  - masks/<absorbed>.png removed
  - masks/<deleted>.png  removed
  - Remaining masks renumbered 1..M (contiguous)
  - masks/mask.png (integer label map) rewritten
  - object_class.json   updated
  - mask_attribute.json updated (via mask_attribute module)

Usage:
    python merge_masks.py --scene_dir /path/to/scene --merge_plan plan.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Local sibling import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mask_attribute


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_mask(path: Path) -> np.ndarray:
    """Load PNG as binary boolean array (H, W)."""
    return np.array(Image.open(path).convert("L")) > 0


def _save_mask(path: Path, arr: np.ndarray) -> None:
    """Save boolean array as 0/255 grayscale PNG."""
    Image.fromarray((arr.astype(np.uint8) * 255)).save(path)


def _load_object_class(scene_dir: Path) -> dict[str, str]:
    oc_path = scene_dir / "object_class.json"
    if not oc_path.exists():
        return {}
    with open(oc_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if "objects" in raw:
        return {str(k): str(v) for k, v in raw["objects"].items()}
    return {str(k): str(v) for k, v in raw.items()}


def _save_object_class(scene_dir: Path, class_map: dict[str, str]) -> None:
    oc_path = scene_dir / "object_class.json"
    with open(oc_path, "w", encoding="utf-8") as fh:
        json.dump(class_map, fh, indent=2)


def _existing_mask_ids(masks_dir: Path) -> list[int]:
    """Return sorted list of integer mask IDs present as PNG files."""
    return sorted(
        int(f.stem) for f in masks_dir.glob("*.png")
        if f.stem.isdigit() and f.name != "mask.png"
    )


def _rebuild_label_map_from_pngs(masks_dir: Path) -> None:
    """
    Rebuild masks/mask.png entirely from the individual PNG files that are
    currently on disk.  Pixels belonging to mask N get value N; later masks
    overwrite earlier ones where they overlap (highest id wins).
    """
    mask_files = sorted(
        [f for f in masks_dir.glob("*.png") if f.stem.isdigit() and f.name != "mask.png"],
        key=lambda f: int(f.stem),
    )
    if not mask_files:
        return

    first = np.array(Image.open(mask_files[0]).convert("L"))
    label_arr = np.zeros(first.shape, dtype=np.int32)

    for mf in mask_files:
        mid = int(mf.stem)
        binary = np.array(Image.open(mf).convert("L")) > 0
        label_arr[binary] = mid

    Image.fromarray(label_arr.astype(np.uint8)).save(masks_dir / "mask.png")


def _pick_largest_canonical_per_group(scene_dir: Path) -> None:
    """
    Heuristic canonical selection for mesh_groups: pick the surviving instance
    whose binary mask has the largest pixel area (mask.sum()).

    Ties are broken by lowest id for determinism. No LLM, no judgement —
    purely a mechanical override of whatever canonical_id the Mask-Evaluator
    wrote, applied AFTER merge_masks has finalized the on-disk masks.
    """
    masks_dir = scene_dir / "masks"
    attr_path = scene_dir / "mask_attribute.json"
    if not attr_path.exists():
        return
    with open(attr_path, "r", encoding="utf-8") as fh:
        attr = json.load(fh)

    mesh_groups = attr.get("mesh_groups") or {}
    if not mesh_groups:
        return

    changes: list[tuple[str, int, int]] = []
    for gname, ginfo in mesh_groups.items():
        instance_ids = [int(i) for i in ginfo.get("instance_ids", [])]
        live_ids = [i for i in instance_ids if (masks_dir / f"{i}.png").exists()]
        if not live_ids:
            continue

        if len(live_ids) == 1:
            new_canonical = live_ids[0]
        else:
            scored: list[tuple[int, int]] = []
            for iid in live_ids:
                m = _load_mask(masks_dir / f"{iid}.png")
                scored.append((int(m.sum()), iid))
            scored.sort(key=lambda t: (-t[0], t[1]))
            new_canonical = scored[0][1]

        old_canonical = int(ginfo.get("canonical_id", new_canonical))
        if new_canonical != old_canonical:
            changes.append((gname, old_canonical, new_canonical))

        ginfo["canonical_id"] = int(new_canonical)
        ginfo["instance_ids"] = sorted(live_ids)

        objects = attr.get("objects") or {}
        for iid in instance_ids:
            obj = objects.get(str(iid))
            if obj is not None and obj.get("mesh_group") == gname:
                obj["canonical"] = (iid == new_canonical)

    if changes:
        attr.setdefault("history", []).append({
            "step": "pick_largest_canonical",
            "changes": [
                {"group": g, "old_canonical_id": o, "new_canonical_id": n}
                for g, o, n in changes
            ],
        })
        print(
            f"[merge_masks] Re-picked canonical for {len(changes)} mesh_group(s) "
            f"by largest mask area.",
            file=sys.stderr,
        )

    with open(attr_path, "w", encoding="utf-8") as fh:
        json.dump(attr, fh, indent=2)


# ── main logic ────────────────────────────────────────────────────────────────

def _normalize_mesh_groups(merge_plan: dict, class_map: dict[str, str]) -> dict:
    """
    Normalize mesh_groups to the canonical dict-of-dicts format expected by
    mask_attribute.record_merge.

    The Mask-Evaluator may write mesh_groups in a compact list format:
        {"2": [2, 3]}         # list of instance ids; first element is canonical
    or the full dict format:
        {"2": {"canonical_id": 2, "instance_ids": [2, 3], "class": "chair"}}

    This function converts the compact form to the full form in-place (on a copy).
    """
    raw = merge_plan.get("mesh_groups")
    if not raw:
        return merge_plan

    normalized: dict = {}
    for grp_name, ginfo in raw.items():
        if isinstance(ginfo, list):
            # Compact format: list of instance ids; treat first as canonical
            ids = [int(i) for i in ginfo]
            canonical = ids[0] if ids else int(grp_name)
            cls = class_map.get(str(canonical), "")
            normalized[grp_name] = {
                "canonical_id": canonical,
                "instance_ids": ids,
                "class": cls,
            }
        else:
            # Already in full dict format
            normalized[grp_name] = ginfo

    result = dict(merge_plan)
    result["mesh_groups"] = normalized
    return result


def _validate_consistency(masks_dir: Path, obj_class_path: Path) -> None:
    mask_stems = {int(p.stem) for p in masks_dir.glob("*.png") if p.stem.isdigit()}
    with open(obj_class_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if "objects" in raw:
        raw = raw["objects"]
    class_keys = {int(k) for k in raw.keys()}
    if mask_stems != class_keys:
        only_mask = sorted(mask_stems - class_keys)
        only_class = sorted(class_keys - mask_stems)
        raise SystemExit(
            f"[merge_masks] FATAL: mask/object_class drift. "
            f"only-in-masks={only_mask} only-in-object_class={only_class}"
        )


def apply_merge_plan(scene_dir: Path, merge_plan: dict) -> None:
    masks_dir = scene_dir / "masks"
    if not masks_dir.is_dir():
        raise NotADirectoryError(f"masks/ not found in {scene_dir}")

    raw_merge_groups = merge_plan.get("merge_groups", [])
    delete_ids = {int(i) for i in merge_plan.get("delete_ids", [])}

    # Normalize each merge_group: keep_id MUST be min(keep_id, *absorb_ids).
    # If the evaluator violated the lowest-id rule, swap and rebuild absorb_ids.
    # Also build absorb→keep mapping for id_remap propagation below.
    normalized_groups: list[dict] = []
    absorb_to_keep: dict[int, int] = {}
    absorbed_ids: set[int] = set()
    for mg in raw_merge_groups:
        raw_keep = int(mg["keep_id"])
        raw_absorbs = [int(i) for i in mg.get("absorb_ids", [])]
        all_ids = [raw_keep, *raw_absorbs]
        normalized_keep = min(all_ids)
        if normalized_keep != raw_keep:
            print(
                f"[merge_masks] WARN: normalized keep_id from {raw_keep} to "
                f"{normalized_keep} for merge_group ids={sorted(all_ids)} "
                f"(evaluator violated lowest-id rule)",
                file=sys.stderr,
            )
        new_absorbs = [i for i in all_ids if i != normalized_keep]
        ng = dict(mg)
        ng["keep_id"] = normalized_keep
        ng["absorb_ids"] = new_absorbs
        normalized_groups.append(ng)
        for aid in new_absorbs:
            absorb_to_keep[aid] = normalized_keep
            absorbed_ids.add(aid)

    class_map = _load_object_class(scene_dir)

    # Normalize mesh_groups from compact list format to full dict format
    merge_plan = _normalize_mesh_groups(merge_plan, class_map)
    merge_plan["merge_groups"] = normalized_groups

    # Step 1: Union all source masks (every id in the group, including the original
    # keep if it got demoted) into the normalized keep_id PNG.
    for mg in normalized_groups:
        keep_id = int(mg["keep_id"])
        absorb_ids = [int(i) for i in mg.get("absorb_ids", [])]
        source_ids = [keep_id, *absorb_ids]

        keep_path = masks_dir / f"{keep_id}.png"
        combined: np.ndarray | None = None
        for sid in source_ids:
            sp = masks_dir / f"{sid}.png"
            if sp.exists():
                m = _load_mask(sp)
                combined = m if combined is None else np.logical_or(combined, m)
            else:
                print(f"[merge_masks] WARNING: mask id {sid} not found", file=sys.stderr)
        if combined is None:
            print(f"[merge_masks] WARNING: no source masks found for keep_id {keep_id}, skipping", file=sys.stderr)
            continue
        _save_mask(keep_path, combined)

    # Step 2: Delete absorbed masks and explicitly deleted masks
    to_remove = absorbed_ids | delete_ids
    for rid in to_remove:
        p = masks_dir / f"{rid}.png"
        if p.exists():
            p.unlink()
        class_map.pop(str(rid), None)

    # Step 3: (DISABLED) Renumber remaining masks contiguously from 1.
    # Original IDs are now preserved so every downstream artifact can reference
    # objects by stable IDs across merges. Absorbed/deleted IDs leave intentional
    # gaps (e.g. after absorbing 18, masks become 1..17, 19..N with no 18.png).
    remaining_ids = _existing_mask_ids(masks_dir)

    # Step 4: Rebuild label map from the updated PNGs (handles merges correctly).
    # _rebuild_label_map_from_pngs uses each PNG's filename stem as the label id,
    # so gaps in the id sequence are honoured.
    _rebuild_label_map_from_pngs(masks_dir)

    # Step 5: Persist class map. class_map already had absorbed/deleted entries
    # popped during Step 2, so writing it as-is preserves the surviving original ids.
    _save_object_class(scene_dir, class_map)

    # Step 6: id_remap. Identity for surviving ids + absorb→keep entries so
    # mask_attribute.record_merge migrates attribute history correctly even
    # when keep_id was demoted (original-keep absorbed into lower id).
    id_remap_str = {str(i): str(i) for i in remaining_ids}
    for absorb_id, keep_id in absorb_to_keep.items():
        id_remap_str[str(absorb_id)] = str(keep_id)
    merge_plan_with_remap = dict(merge_plan)
    merge_plan_with_remap["id_remap"] = id_remap_str

    # Step 7: Update mask_attribute.json
    mask_attribute.record_merge(scene_dir, merge_plan_with_remap)

    # Step 8: Heuristic canonical pick — largest surviving mask area per
    # mesh_group wins, regardless of what the Mask-Evaluator wrote.
    _pick_largest_canonical_per_group(scene_dir)

    # Step 9: Validate mask/object_class consistency before returning.
    _validate_consistency(masks_dir, scene_dir / "object_class.json")

    sorted_ids = sorted(remaining_ids)
    if sorted_ids and sorted_ids != list(range(1, len(sorted_ids) + 1)):
        gap_summary = f"{sorted_ids[0]}..{sorted_ids[-1]} with gaps"
    else:
        gap_summary = f"1..{len(sorted_ids)}"
    print(
        f"[merge_masks] Done. {len(remaining_ids)} masks remain (ids preserved, {gap_summary}).",
        file=sys.stderr,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Apply Mask-Evaluator merge plan to a scene's masks")
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--merge_plan", type=Path, required=True, help="Path to merge plan JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    with open(args.merge_plan, "r", encoding="utf-8") as fh:
        merge_plan = json.load(fh)
    apply_merge_plan(scene_dir, merge_plan)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[merge_masks] ERROR: {exc}", file=sys.stderr)
        raise
