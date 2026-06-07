#!/usr/bin/env python3
"""For each group in inputs/relation_graph.json, write a masked reference image.

Per group:
  - Union the masks of all real obj_* members (and the JSON anchor if obj_*).
  - Apply that union mask to image.png — pixels inside the union keep their
    RGB; pixels outside go black.
  - Save to <scene_dir>/relation_groups/<group_id>/masked.png

Pairs with build_group_islands_from_graph.py — same member-set logic, so each
group's island.blend and masked.png describe the same set of objects.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def collect_member_ids(group: dict) -> list[str]:
    """Mirror build_group_islands_from_graph.py's member-set logic."""
    members = list(group.get("members", []))
    anchor = group.get("anchor", "")
    if anchor and anchor.startswith("obj_") and anchor not in members:
        members.append(anchor)
    return [m for m in members if m.startswith("obj_")]


def union_masks(mask_paths: list[Path], target_hw: tuple[int, int]) -> np.ndarray:
    """Return uint8 mask (h, w) with 255 where any input mask is non-zero."""
    h, w = target_hw
    out = np.zeros((h, w), dtype=np.uint8)
    for p in mask_paths:
        if not p.exists():
            print(f"  warn: missing {p}", file=sys.stderr)
            continue
        m = np.array(Image.open(p).convert("L"))
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))
        out = np.maximum(out, (m > 0).astype(np.uint8) * 255)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--out_subdir", default="relation_groups")
    ap.add_argument("--filename", default="masked.png",
                    help="Filename written into each group's folder")
    ap.add_argument("--alpha", action="store_true",
                    help="Save as RGBA with mask as alpha channel (default: RGB on black)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scene = Path(args.scene_dir).resolve()
    image_path = scene / "image.png"
    masks_dir = scene / "inputs" / "masks"
    graph_path = scene / "inputs" / "relation_graph.json"
    out_root = scene / args.out_subdir

    for p in (image_path, masks_dir, graph_path):
        if not p.exists():
            print(f"missing required input: {p}", file=sys.stderr)
            return 1
    if not out_root.exists():
        print(f"out_subdir does not exist (run build_group_islands_from_graph.py first): {out_root}",
              file=sys.stderr)
        return 1

    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    graph = json.loads(graph_path.read_text())

    n_made = 0
    n_skipped = 0
    for grp in graph.get("groups", []):
        gid = grp["group_id"]
        member_ids = collect_member_ids(grp)
        if not member_ids:
            print(f"[{gid}] no real obj_* members, skipping")
            continue

        out_dir = out_root / gid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.filename
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        mask_paths = [masks_dir / f"{m.replace('obj_', '')}.png" for m in member_ids]
        union = union_masks(mask_paths, (h, w))

        if args.alpha:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = image
            rgba[..., 3] = union
            Image.fromarray(rgba, "RGBA").save(out_path)
        else:
            keep = union > 0
            masked = np.zeros_like(image)
            masked[keep] = image[keep]
            Image.fromarray(masked, "RGB").save(out_path)

        coverage = float((union > 0).mean()) * 100
        print(f"[{gid}] members={member_ids}  coverage={coverage:.1f}%  -> {out_path.name}")
        n_made += 1

    print(f"\ndone: made={n_made} skipped={n_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
