"""
Stage 1 pre-evaluator small-mask candidate detector.

Surfaces masks that are geometrically suspicious by their size and adjacency
profile so the vision-only evaluator does not miss them among 30+ per-mask
PNGs. Three categories are produced and the evaluator then VISUALLY confirms
before acting:

  delete_candidates       — very tiny AND no significant neighbor (likely
                            GroundedSAM noise / fragment of nothing).
                            Evaluator should visually verify before adding to
                            `delete_ids` — a real small object (printer button,
                            remote control) may legitimately be tiny.

  merge_into_candidates   — small AND clearly adjacent to a much larger mask
                            (≥ DOMINANT_NEIGHBOR_RATIO×). Likely a fragment of
                            the larger object that GroundedSAM split off.
                            Evaluator should visually verify before merging.

  review_candidates       — small but ambiguous (multiple medium-sized
                            neighbors, no single dominant one). Evaluator
                            decides case-by-case.

This is a HINT generator, NOT a deterministic mandate. The pixel-overlap
pre-filter (compute_overlap_pairs.py) IS deterministic for its narrow case
(pixel-confirmed duplicate masks), but small-mask decisions require visual
judgment that geometry alone cannot make.

Output schema:
{
  "delete_candidates": [
    {"id": 18, "area_px": 200, "area_ratio": 0.00020, "reason": "..."}
  ],
  "merge_into_candidates": [
    {"small_id": 22, "small_area_px": 900, "small_area_ratio": 0.0009,
     "large_id": 12, "large_area_px": 24000, "size_ratio": 0.0375,
     "dilated_touch_px": 1500, "reason": "..."}
  ],
  "review_candidates": [
    {"id": 7, "area_px": 1200, "area_ratio": 0.0012,
     "top_neighbors": [{"id": 12, "area_px": 24000, "dilated_touch_px": 200},
                       {"id": 9, "area_px": 2500, "dilated_touch_px": 150}],
     "reason": "..."}
  ],
  "thresholds": {...},
  "n_masks": 33
}

Threshold rationale (tunable):
- SMALL_RATIO         = 0.001  (< 0.1% of image area → "small")
- ISOLATED_RATIO      = 0.0005 (< 0.05% AND no neighbor → likely delete)
- DILATE_PX           = 5      (adjacency check radius)
- MIN_TOUCH_PX        = 200    (dilated overlap < 200 px ≈ not adjacent)
- DOMINANT_RATIO      = 5      (neighbor must be ≥ 5× larger to "dominate")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """4-connected binary dilation, iterated `iterations` times. numpy-only."""
    out = mask.copy()
    for _ in range(iterations):
        up = np.zeros_like(out); up[:-1] = out[1:]
        dn = np.zeros_like(out); dn[1:] = out[:-1]
        lt = np.zeros_like(out); lt[:, :-1] = out[:, 1:]
        rt = np.zeros_like(out); rt[:, 1:] = out[:, :-1]
        out = out | up | dn | lt | rt
    return out

SMALL_RATIO = 0.001       # area / image_area
ISOLATED_RATIO = 0.0005   # very small AND isolated → delete candidate
DILATE_PX = 5
MIN_TOUCH_PX = 200
DOMINANT_RATIO = 5.0


def _load_masks(masks_dir: Path) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for p in sorted(masks_dir.iterdir()):
        if not (p.suffix == ".png" and p.stem.isdigit()):
            continue
        mid = int(p.stem)
        if mid == 0:
            continue
        arr = np.array(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks[mid] = arr > 0
    return masks


def compute_small_mask_candidates(scene_dir: str | Path) -> dict:
    scene_dir = Path(scene_dir)
    masks_dir = scene_dir / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks dir not found: {masks_dir}")

    masks = _load_masks(masks_dir)
    if not masks:
        raise RuntimeError(f"no per-object masks in {masks_dir}")

    # image area
    any_mask = next(iter(masks.values()))
    img_area = any_mask.size

    areas = {mid: int(m.sum()) for mid, m in masks.items()}
    ids_sorted = sorted(masks.keys())

    # Pre-compute dilated masks for adjacency (numpy-only 4-connected dilation)
    dilated: dict[int, np.ndarray] = {
        mid: _dilate(m, iterations=DILATE_PX) for mid, m in masks.items()
    }

    small_ids = [mid for mid in ids_sorted if areas[mid] / img_area < SMALL_RATIO and areas[mid] > 0]

    delete_candidates: list[dict] = []
    merge_into_candidates: list[dict] = []
    review_candidates: list[dict] = []

    for sid in small_ids:
        s_area = areas[sid]
        s_ratio = s_area / img_area
        s_mask = masks[sid]
        s_dil = dilated[sid]

        # Find neighbors (other ids whose dilated mask touches this one's dilated mask)
        neighbors = []  # list of (nid, n_area, touch_px)
        for nid in ids_sorted:
            if nid == sid:
                continue
            n_area = areas[nid]
            if n_area == 0:
                continue
            touch = int(np.logical_and(s_dil, dilated[nid]).sum())
            if touch >= MIN_TOUCH_PX:
                neighbors.append((nid, n_area, touch))
        neighbors.sort(key=lambda t: -t[2])  # sort by touch desc

        # Find a dominant neighbor: significantly larger
        dominant = None
        for nid, n_area, touch in neighbors:
            if n_area >= DOMINANT_RATIO * s_area:
                dominant = (nid, n_area, touch)
                break

        # Classify
        if not neighbors and s_ratio < ISOLATED_RATIO:
            delete_candidates.append({
                "id": sid,
                "area_px": s_area,
                "area_ratio": round(s_ratio, 6),
                "reason": (
                    f"tiny isolated mask (area_ratio < {ISOLATED_RATIO}, "
                    f"no neighbor within {DILATE_PX}px dilation with touch ≥ {MIN_TOUCH_PX}px)"
                ),
            })
        elif dominant is not None and (
            # secondary neighbors are much smaller than the dominant one
            len(neighbors) == 1
            or neighbors[1][1] < dominant[1] * 0.3
        ):
            nid, n_area, touch = dominant
            merge_into_candidates.append({
                "small_id": sid,
                "small_area_px": s_area,
                "small_area_ratio": round(s_ratio, 6),
                "large_id": nid,
                "large_area_px": n_area,
                "size_ratio": round(s_area / n_area, 4),
                "dilated_touch_px": touch,
                "reason": (
                    f"small mask adjacent to dominant neighbor "
                    f"({n_area // s_area}× larger, touch_px={touch})"
                ),
            })
        else:
            review_candidates.append({
                "id": sid,
                "area_px": s_area,
                "area_ratio": round(s_ratio, 6),
                "top_neighbors": [
                    {"id": nid, "area_px": n_area, "dilated_touch_px": touch}
                    for nid, n_area, touch in neighbors[:3]
                ],
                "reason": "small but ambiguous neighbor profile — evaluator decides",
            })

    return {
        "delete_candidates": delete_candidates,
        "merge_into_candidates": merge_into_candidates,
        "review_candidates": review_candidates,
        "thresholds": {
            "small_ratio": SMALL_RATIO,
            "isolated_ratio": ISOLATED_RATIO,
            "dilate_px": DILATE_PX,
            "min_touch_px": MIN_TOUCH_PX,
            "dominant_ratio": DOMINANT_RATIO,
        },
        "n_masks": len(masks),
        "image_area_px": img_area,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 deterministic small-mask candidate detector"
    )
    ap.add_argument("--scene_dir", required=True, type=Path)
    args = ap.parse_args()

    result = compute_small_mask_candidates(args.scene_dir)
    out_path = args.scene_dir / "small_mask_candidates.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    n_del = len(result["delete_candidates"])
    n_merge = len(result["merge_into_candidates"])
    n_review = len(result["review_candidates"])
    print(
        f"[compute_small_mask_candidates] n_masks={result['n_masks']}  "
        f"delete={n_del}  merge_into={n_merge}  review={n_review}  "
        f"-> {out_path}"
    )
    for c in result["delete_candidates"]:
        print(f"  delete: id={c['id']} area={c['area_px']} ({c['area_ratio']*100:.3f}%)")
    for c in result["merge_into_candidates"]:
        print(
            f"  merge_into: small={c['small_id']} ({c['small_area_px']}) "
            f"→ large={c['large_id']} ({c['large_area_px']}, touch={c['dilated_touch_px']})"
        )
    for c in result["review_candidates"]:
        print(f"  review: id={c['id']} area={c['area_px']} neighbors={len(c['top_neighbors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
