"""
Stage 1 pre-evaluator deterministic over-segmentation detector.

Computes pixel-level pairwise overlap between every per-object mask under
`<scene_dir>/masks/<id>.png` and writes `<scene_dir>/overlap_pairs.json`.

Purpose: when GroundedSAM segments the same physical object twice (e.g. a desk
captured both as a full mask and as a top-edge strip), the two masks can have
near-total pixel overlap. The vision-only evaluator may visually miss such
pairs among 30+ per-mask PNGs. This script enumerates every (i, j) pair
deterministically and flags any pair where `intersection / smaller_area >= 0.5`
as a must-merge candidate that the evaluator is then REQUIRED to include in
`merge_plan.json::merge_groups`.

NOT class-based dedup (which is forbidden by the skill contract): geometry-only
pre-filter. The output is hint data for the evaluator — final authorship of
`merge_plan.json` still belongs to the evaluator agent.

Output schema:
{
  "must_merge_pairs": [
    {"ids": [15, 35], "keep_id": 15, "absorb_id": 35,
     "intersection_px": 8242, "smaller_area_px": 8251,
     "overlap_in_smaller": 0.999}
    , ...
  ],
  "review_pairs": [  // 0.2 <= overlap < 0.5 — sandwich risk
    {"ids": [...], "overlap_in_smaller": 0.31, ...}
  ],
  "thresholds": {"must_merge": 0.5, "review": 0.2},
  "n_masks": 33
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MUST_MERGE_THRESHOLD = 0.5
REVIEW_THRESHOLD = 0.2


def _load_masks(masks_dir: Path) -> dict[int, np.ndarray]:
    """Return {id: bool ndarray} for every per-object PNG (1-indexed, gaps OK)."""
    masks: dict[int, np.ndarray] = {}
    for p in sorted(masks_dir.iterdir()):
        if not (p.suffix == ".png" and p.stem.isdigit()):
            continue  # skip mask.png (label map) and any non-digit-stemmed files
        mid = int(p.stem)
        if mid == 0:
            continue  # 0 reserved for background
        arr = np.array(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks[mid] = arr > 0
    return masks


def compute_overlap_pairs(scene_dir: str | Path) -> dict:
    scene_dir = Path(scene_dir)
    masks_dir = scene_dir / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks dir not found: {masks_dir}")

    masks = _load_masks(masks_dir)
    if not masks:
        raise RuntimeError(f"no per-object masks found in {masks_dir}")

    areas = {mid: int(m.sum()) for mid, m in masks.items()}
    ids_sorted = sorted(masks.keys())

    must_merge_pairs: list[dict] = []
    review_pairs: list[dict] = []

    for i, mid_a in enumerate(ids_sorted):
        m_a = masks[mid_a]
        area_a = areas[mid_a]
        if area_a == 0:
            continue
        for mid_b in ids_sorted[i + 1 :]:
            m_b = masks[mid_b]
            area_b = areas[mid_b]
            if area_b == 0:
                continue
            inter = int(np.logical_and(m_a, m_b).sum())
            if inter == 0:
                continue
            smaller_area = min(area_a, area_b)
            ratio = inter / smaller_area
            if ratio < REVIEW_THRESHOLD:
                continue
            smaller_id = mid_a if area_a < area_b else mid_b
            larger_id = mid_b if smaller_id == mid_a else mid_a
            entry = {
                "ids": [mid_a, mid_b],
                "keep_id": larger_id,
                "absorb_id": smaller_id,
                "intersection_px": inter,
                "smaller_area_px": smaller_area,
                "overlap_in_smaller": round(ratio, 4),
            }
            if ratio >= MUST_MERGE_THRESHOLD:
                must_merge_pairs.append(entry)
            else:
                review_pairs.append(entry)

    return {
        "must_merge_pairs": must_merge_pairs,
        "review_pairs": review_pairs,
        "thresholds": {"must_merge": MUST_MERGE_THRESHOLD, "review": REVIEW_THRESHOLD},
        "n_masks": len(masks),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 deterministic pairwise mask-overlap pre-filter"
    )
    ap.add_argument("--scene_dir", required=True, type=Path)
    args = ap.parse_args()

    result = compute_overlap_pairs(args.scene_dir)
    out_path = args.scene_dir / "overlap_pairs.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    n_must = len(result["must_merge_pairs"])
    n_review = len(result["review_pairs"])
    print(
        f"[compute_overlap_pairs] n_masks={result['n_masks']}  "
        f"must_merge={n_must}  review={n_review}  "
        f"-> {out_path}"
    )
    if n_must:
        for p in result["must_merge_pairs"]:
            print(
                f"  must_merge: keep={p['keep_id']} absorb={p['absorb_id']} "
                f"overlap_in_smaller={p['overlap_in_smaller']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
