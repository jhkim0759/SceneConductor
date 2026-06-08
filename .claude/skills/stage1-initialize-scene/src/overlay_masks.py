#!/usr/bin/env python3
"""
overlay_masks.py — Draw all masks on top of the scene image with per-id
color, centroid dot, and numbered label. Optionally highlights merge_groups
and mesh_groups from a plan JSON using matching color families.

Output: <scene_dir>/verification_overlay.png

Usage:
    python overlay_masks.py --scene_dir /path/to/scene
    python overlay_masks.py --scene_dir /path/to/scene \
        --plan /path/to/merge_plan.json \
        --out verification_overlay_pass2.png

Use case: after a merge_masks.py pass, generate an overlay so the
Mask-Evaluator (or a human) can visually verify the current mask layout and
spot remaining issues — especially masks "sandwiched" between a merge group.

Public API (importable by other scripts):
    render_overlay(scene_dir, out_path, plan=None, alpha=0.45) -> Path
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── importable helpers ────────────────────────────────────────────────────────

def _distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """Evenly-spaced HSV colors, converted to RGB 0-255."""
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _mask_centroid(mask_png: Path) -> tuple[int, int] | None:
    arr = np.array(Image.open(mask_png).convert("L")) > 0
    if arr.sum() == 0:
        return None
    rows, cols = np.where(arr)
    return int(cols.mean()), int(rows.mean())


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable TrueType font, falling back to PIL default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, 14)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_plan(plan_path: Path | None) -> dict:
    if plan_path is None or not plan_path.exists():
        return {}
    with open(plan_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_class_map(scene_dir: Path) -> dict[int, str]:
    p = scene_dir / "object_class.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    data = raw.get("objects", raw)
    return {int(k): str(v) for k, v in data.items() if str(k).isdigit()}


def render_overlay(
    scene_dir: Path,
    out_path: Path,
    plan: Optional[dict] = None,
    alpha: float = 0.45,
) -> Path:
    """
    Render a mask overlay PNG to out_path and return its resolved path.

    Parameters
    ----------
    scene_dir : Path
        Scene directory containing image.png and masks/.
    out_path : Path
        Absolute path for the output PNG (does not need to be inside scene_dir).
    plan : dict | None
        Optional merge plan dict (already loaded). When provided, merge_groups
        and mesh_groups are highlighted with matching color families.
    alpha : float
        Mask fill opacity (0–1, default 0.45).

    Returns
    -------
    Path
        Resolved path of the written PNG.
    """
    scene_dir = Path(scene_dir).resolve()
    out_path = Path(out_path).resolve()

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        raise FileNotFoundError(f"image.png not found in {scene_dir}")

    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]

    masks_dir = scene_dir / "masks"
    mask_files = sorted(
        [f for f in masks_dir.glob("*.png") if f.stem.isdigit() and f.name != "mask.png"],
        key=lambda f: int(f.stem),
    )
    if not mask_files:
        raise FileNotFoundError(f"No numbered masks found in {masks_dir}")

    n = len(mask_files)
    palette = _distinct_colors(n)
    class_map = _load_class_map(scene_dir)
    if plan is None:
        plan = {}

    # Color overrides from plan:
    # masks in same merge_group → same color family (slight brightness variation)
    # masks in same mesh_group → one color, add marker
    color_by_id: dict[int, tuple[int, int, int]] = {
        int(f.stem): palette[i] for i, f in enumerate(mask_files)
    }
    merge_markers: dict[int, str] = {}  # id -> "M1", "M2", ...
    mesh_markers: dict[int, str] = {}   # id -> "canonical·group_name", etc.

    for gi, mg in enumerate(plan.get("merge_groups", []), start=1):
        base_color = palette[(gi * 7) % n] if n > 0 else (255, 0, 0)
        keep_id = int(mg.get("keep_id"))
        absorbed = [int(x) for x in mg.get("absorb_ids", [])]
        marker = f"M{gi}"
        color_by_id[keep_id] = base_color
        merge_markers[keep_id] = f"{marker}★"
        for j, aid in enumerate(absorbed, start=1):
            # slightly darker variants for absorbed ids
            r, g, b = base_color
            factor = 0.65 + 0.1 * (j % 3)
            color_by_id[aid] = (int(r * factor), int(g * factor), int(b * factor))
            merge_markers[aid] = marker

    for group_name, ginfo in plan.get("mesh_groups", {}).items():
        cid = int(ginfo.get("canonical_id"))
        iids = [int(x) for x in ginfo.get("instance_ids", [cid])]
        for iid in iids:
            tag = "★canonical" if iid == cid else f"↳{cid}"
            mesh_markers[iid] = f"{tag}·{group_name}"

    # Compose overlay
    overlay = image.astype(np.float32)
    for f in mask_files:
        mid = int(f.stem)
        m = np.array(Image.open(f).convert("L")) > 0
        color = np.array(color_by_id[mid], dtype=np.float32)
        for c in range(3):
            overlay[..., c][m] = (1 - alpha) * overlay[..., c][m] + alpha * color[c]

    out_img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out_img)

    font = _load_font()

    # Draw labels at centroids
    for f in mask_files:
        mid = int(f.stem)
        c = _mask_centroid(f)
        if c is None:
            continue
        x, y = c
        cls = class_map.get(mid, "?")
        merge_tag = merge_markers.get(mid, "")
        mesh_tag = mesh_markers.get(mid, "")
        label_parts = [f"#{mid}", cls]
        if merge_tag:
            label_parts.append(merge_tag)
        if mesh_tag:
            label_parts.append(mesh_tag)
        label = " | ".join(label_parts)

        # Centroid dot
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255),
                     outline=(0, 0, 0))
        # Label with backdrop for readability
        bbox = draw.textbbox((x + 6, y - 8), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x + 6, y - 8), label, fill=(255, 255, 255), font=font)

    # Header strip
    header = f"Scene: {scene_dir.name}   Masks: {n}"
    if plan:
        mg_count = len(plan.get("merge_groups", []))
        msh_count = len(plan.get("mesh_groups", {}))
        header += f"   merge_groups: {mg_count}   mesh_groups: {msh_count}"
    draw.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    draw.text((6, 4), header, fill=(255, 255, 255), font=font)

    out_img.save(out_path)
    return out_path


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=None,
                    help="Optional merge_plan.json to highlight merge_groups/mesh_groups")
    ap.add_argument("--out", type=str, default="verification_overlay.png",
                    help="Output filename (inside scene_dir)")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="Mask fill opacity (0-1)")
    args = ap.parse_args()

    scene_dir = args.scene_dir.resolve()
    if not scene_dir.is_dir():
        print(f"[overlay_masks] ERROR: {scene_dir} not found", file=sys.stderr)
        sys.exit(1)

    plan = _load_plan(args.plan)
    out_path = scene_dir / args.out

    written = render_overlay(scene_dir, out_path, plan=plan, alpha=args.alpha)

    # Determine image dimensions for the log line
    image = np.array(Image.open(scene_dir / "image.png").convert("RGB"))
    H, W = image.shape[:2]
    masks_dir = scene_dir / "masks"
    n = len([f for f in masks_dir.glob("*.png") if f.stem.isdigit() and f.name != "mask.png"])

    log_suffix = ""
    if args.plan:
        log_suffix = f"   Plan: {Path(args.plan).name}"
        log_suffix += f"   merge_groups: {len(plan.get('merge_groups', []))}"
        log_suffix += f"   mesh_groups: {len(plan.get('mesh_groups', {}))}"

    print(f"[overlay_masks] Wrote {written} ({n} masks, {W}x{H}){log_suffix}")


if __name__ == "__main__":
    main()
