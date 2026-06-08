#!/usr/bin/env python3
"""Step 3.6 — Enrich <scene_dir>/mask_attribute.json with shape descriptors
and per-mask Qwen-VL class-consistency check.

Runs after Step 3 (init_attributes) and Step 3.5 (make_annotated_mask), before
phase eval (Mask-Evaluator). Each object's entry gains two sub-objects:

  "shape": {
    "aspect_ratio":   float,           # max(w,h) / max(min(w,h), 1)
    "compactness":    float,           # mask_area_px / bbox_area_px (∈ [0,1])
    "is_thin_strip":  bool             # aspect_ratio > 5.0 OR compactness < 0.25
  },
  "vlm_check": {
    "content":          str,           # plain noun phrase of what is actually in the crop
    "matches_class":    bool,          # does the crop reasonably depict the assigned class?
    "confidence":       float,         # 0.0–1.0
    "suspected_actual": str            # best category for the crop content
  }

The Mask-Evaluator (phase eval) reads these and can delete masks where
matches_class=false + visually evidently a structural fragment, without
needing class whitelists.

NO class whitelists are baked in here — the only authority is the VLM's
visual comparison between the crop and its assigned class label.

Idempotent: re-running overwrites existing shape / vlm_check blocks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Allow importing the Qwen loader from the stage3 sub-skill (shared model code).
THIS_DIR = Path(__file__).resolve().parent
SHARED_QWEN_DIR = THIS_DIR.parent.parent / "stage3-sub-scene-analyze-prepare" / "src"
sys.path.insert(0, str(SHARED_QWEN_DIR))
from generate_object_state_json import load_qwen, run_qwen  # noqa: E402


CROP_PAD_FRAC = 0.15          # pad bbox by this fraction of its dim before cropping
CROP_MIN_PX = 96              # minimum crop side (upscale if smaller)
CROP_MAX_PX = 768             # max crop side (downscale if larger; bandwidth control)
FULL_MAX_PX = 768             # max side for the full-image context view (bandwidth control)
OUTLINE_THICKNESS_FULL = 3    # outline thickness in pixels on the downscaled full view


# ────────────────────────────────────────────────────────────────────────────
# Shape descriptors
# ────────────────────────────────────────────────────────────────────────────

def _compute_shape(mask_arr: np.ndarray) -> dict[str, Any]:
    binary = mask_arr > 0
    area = int(binary.sum())
    if area == 0:
        return {
            "aspect_ratio": 0.0,
            "compactness": 0.0,
            "is_thin_strip": False,
        }
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    h = max(rmax - rmin + 1, 1)
    w = max(cmax - cmin + 1, 1)
    long_side = max(h, w)
    short_side = max(min(h, w), 1)
    aspect_ratio = float(long_side / short_side)
    bbox_area = h * w
    compactness = float(area / max(bbox_area, 1))
    is_thin_strip = bool(aspect_ratio > 5.0 or compactness < 0.25)
    return {
        "aspect_ratio": round(aspect_ratio, 3),
        "compactness": round(compactness, 3),
        "is_thin_strip": is_thin_strip,
    }


# ────────────────────────────────────────────────────────────────────────────
# Crop with mask outline overlay
# ────────────────────────────────────────────────────────────────────────────

def _make_crop_with_outline(
    image: Image.Image,
    mask_arr: np.ndarray,
    out_path: Path,
) -> bool:
    """Crop image to mask bbox (padded), draw the mask outline in red. Returns
    True on success; False if the mask is empty / degenerate."""
    H, W = mask_arr.shape
    binary = mask_arr > 0
    if not binary.any():
        return False
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    pad_x = int(w * CROP_PAD_FRAC)
    pad_y = int(h * CROP_PAD_FRAC)
    x0 = max(0, cmin - pad_x)
    y0 = max(0, rmin - pad_y)
    x1 = min(W, cmax + pad_x + 1)
    y1 = min(H, rmax + pad_y + 1)
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")
    crop_mask = mask_arr[y0:y1, x0:x1] > 0

    # Draw outline by detecting pixels where binary neighbour differs.
    from PIL import ImageDraw
    arr = np.array(crop, dtype=np.uint8)
    edge = np.zeros_like(crop_mask, dtype=bool)
    edge[:-1, :] |= crop_mask[:-1, :] != crop_mask[1:, :]
    edge[1:, :] |= crop_mask[:-1, :] != crop_mask[1:, :]
    edge[:, :-1] |= crop_mask[:, :-1] != crop_mask[:, 1:]
    edge[:, 1:] |= crop_mask[:, :-1] != crop_mask[:, 1:]
    # Thicken outline by 1px so it survives downscale.
    arr[edge] = (255, 32, 32)
    crop = Image.fromarray(arr)

    # Resize for bandwidth + ensure minimum legibility.
    cw, ch = crop.size
    long = max(cw, ch)
    if long > CROP_MAX_PX:
        scale = CROP_MAX_PX / long
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
    elif long < CROP_MIN_PX:
        scale = CROP_MIN_PX / long
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

    crop.save(out_path)
    return True


def _make_full_with_outline(
    image: Image.Image,
    mask_arr: np.ndarray,
    out_path: Path,
) -> bool:
    """Save a downscaled copy of the full image with the mask's outline drawn
    in red. Gives the VLM scene-level context so it can judge plausibility of
    the class given the region's position in the room."""
    H, W = mask_arr.shape
    if not (mask_arr > 0).any():
        return False
    rgb = np.array(image.convert("RGB"), dtype=np.uint8).copy()
    binary = mask_arr > 0
    edge = np.zeros_like(binary, dtype=bool)
    edge[:-1, :] |= binary[:-1, :] != binary[1:, :]
    edge[1:, :] |= binary[:-1, :] != binary[1:, :]
    edge[:, :-1] |= binary[:, :-1] != binary[:, 1:]
    edge[:, 1:] |= binary[:, :-1] != binary[:, 1:]
    # Thicken outline so it survives downscale.
    if OUTLINE_THICKNESS_FULL > 1:
        from scipy.ndimage import binary_dilation  # noqa: F401
        try:
            edge = binary_dilation(edge, iterations=OUTLINE_THICKNESS_FULL - 1)
        except Exception:
            # Fallback: manual 1px dilate per extra step.
            for _ in range(OUTLINE_THICKNESS_FULL - 1):
                e2 = edge.copy()
                e2[1:, :] |= edge[:-1, :]
                e2[:-1, :] |= edge[1:, :]
                e2[:, 1:] |= edge[:, :-1]
                e2[:, :-1] |= edge[:, 1:]
                edge = e2
    rgb[edge] = (255, 32, 32)
    out = Image.fromarray(rgb)
    iw, ih = out.size
    long = max(iw, ih)
    if long > FULL_MAX_PX:
        scale = FULL_MAX_PX / long
        out = out.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
    out.save(out_path)
    return True


# ────────────────────────────────────────────────────────────────────────────
# Qwen prompt + response parsing
# ────────────────────────────────────────────────────────────────────────────

VLM_SYSTEM_PROMPT = (
    "You are inspecting ONE candidate region inside a wider indoor photograph. "
    "An automatic segmentation pipeline assigned the region a class label. You "
    "will see TWO images: (1) the full source image with the region outlined in "
    "red — use this for scene-level context (room layout, where the region "
    "sits relative to floor / walls / ceiling, and what other objects exist), "
    "and (2) a close-up crop of the same region — use this for detail. "
    "Your job: report what is actually inside the red outline AND judge whether "
    "the assigned class is plausible given BOTH the visual content AND the "
    "region's POSITION in the room. A 'counter' or 'table' silhouette that "
    "sits along the floor base (not at standing height) is likely a "
    "floor-edge / carpet line / baseboard, not a real counter. A 'fluorescent "
    "light' silhouette that does not lie against the ceiling is likely a "
    "wall poster or panel. Use indoor common sense. Be concrete and literal. "
    "Do not speculate about parts outside the outline."
)


def _position_bucket(c: float) -> str:
    """Map a normalized coordinate (0..1) into a coarse spatial bucket."""
    if c < 0.33:
        return "left/top"  # caller annotates axis
    if c < 0.66:
        return "middle"
    return "right/bottom"


def _build_user_prompt(assigned_class: str, bbox: list[float], shape: dict) -> str:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    # Coarse horizontal / vertical zones for human-readable position cues.
    hzone = "left" if cx < 0.33 else ("center" if cx < 0.66 else "right")
    vzone = "upper" if cy < 0.33 else ("middle" if cy < 0.66 else "lower")
    floor_warn = ""
    if y1 > 0.85:
        floor_warn = (
            " The region's bottom edge sits in the lower-image floor band "
            f"(y1={y1:.2f} > 0.85); be especially careful about labels that "
            "should NOT be at floor level (counter, table, shelf, wall fixture)."
        )
    ceiling_warn = ""
    if y0 < 0.15:
        ceiling_warn = (
            " The region's top edge sits in the upper-image ceiling band "
            f"(y0={y0:.2f} < 0.15); be especially careful about labels that "
            "should NOT be at ceiling level (floor strip, chair, table)."
        )
    thin = "yes" if shape.get("is_thin_strip") else "no"
    aspect = shape.get("aspect_ratio", 0.0)
    compact = shape.get("compactness", 0.0)
    return (
        f"The region inside the red outline was assigned class: '{assigned_class}'.\n"
        f"Region position in the source image (normalized 0..1, top-left origin):\n"
        f"  bbox = [{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}]\n"
        f"  center ≈ ({cx:.2f}, {cy:.2f})  → {vzone}-{hzone} of the image\n"
        f"  shape: aspect_ratio={aspect:.2f}, compactness={compact:.2f}, "
        f"thin_strip={thin}.{floor_warn}{ceiling_warn}\n"
        "Use BOTH the full image (for position context) AND the crop (for "
        "detail) to decide.\n"
        "Answer in EXACTLY four lines, no extra text:\n"
        "CONTENT: <one short noun phrase describing what is actually inside the red outline>\n"
        "MATCHES: yes or no — whether the assigned class is plausible given "
        "both visual content AND position (e.g. 'counter at floor base' → no)\n"
        "CONFIDENCE: a number from 0.0 to 1.0\n"
        "ACTUAL: <one short noun phrase for the true category; if MATCHES=yes, "
        f"repeat '{assigned_class}'>"
    )


_LINE_RE = re.compile(r"^\s*([A-Z_]+)\s*:\s*(.+?)\s*$")


def _parse_vlm_response(text: str, assigned_class: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).upper(), m.group(2).strip()
        fields[key] = val

    content = fields.get("CONTENT", "")
    matches_raw = fields.get("MATCHES", "").lower()
    matches = matches_raw.startswith("y")
    conf_raw = fields.get("CONFIDENCE", "0.0")
    try:
        # Tolerate "0.85", ".85", "0.85.", "0.85 (high)" etc.
        conf_match = re.search(r"[01](?:\.\d+)?|\.\d+", conf_raw)
        confidence = float(conf_match.group(0)) if conf_match else 0.0
    except (ValueError, AttributeError):
        confidence = 0.0
    actual = fields.get("ACTUAL", "")
    if not actual:
        actual = assigned_class if matches else content or "unknown"
    return {
        "content": content,
        "matches_class": matches,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "suspected_actual": actual,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3.6 — enrich mask_attribute.json with shape + Qwen-VL "
                    "class-consistency check, before phase eval."
    )
    ap.add_argument("--scene_dir", required=True, type=Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B",
                    help="VLM model id or local checkpoint path "
                         "(default: Qwen/Qwen3.5-27B).")
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="VLM gen budget per crop (default 128 — answer is 4 lines).")
    ap.add_argument("--keep_crops", action="store_true",
                    help="Persist per-mask crops under <scene_dir>/.enrich_crops/ "
                         "for debugging. Default: cleaned up.")
    args = ap.parse_args()

    scene_dir: Path = args.scene_dir.resolve()
    image_path = scene_dir / "image.png"
    masks_dir = scene_dir / "masks"
    attr_path = scene_dir / "mask_attribute.json"
    for required in (image_path, masks_dir, attr_path):
        if not required.exists():
            raise SystemExit(f"[enrich] missing required input: {required}")

    image = Image.open(image_path).convert("RGB")
    with open(attr_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    objects: dict[str, Any] = data.get("objects", {})
    if not objects:
        raise SystemExit(f"[enrich] no objects in {attr_path}")

    # Load Qwen-VL.
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    print(f"[enrich] loading VLM: {args.model}", flush=True)
    torch_mod, processor, model, model_device = load_qwen(
        args.model, local_files_only=True, allow_cpu=False,
    )

    if args.keep_crops:
        crops_dir = scene_dir / ".enrich_crops"
        crops_dir.mkdir(exist_ok=True)
        tmpctx = None
    else:
        tmpctx = tempfile.TemporaryDirectory(prefix="enrich_crops_")
        crops_dir = Path(tmpctx.name)

    n_total = len(objects)
    n_ok = 0
    n_skip = 0
    n_thin = 0
    n_mismatch = 0

    try:
        # Sort by integer id so logs are ordered.
        for mid in sorted(objects.keys(), key=lambda s: int(s) if s.isdigit() else 0):
            obj = objects[mid]
            assigned_class = obj.get("class", "unknown")
            mask_path = masks_dir / f"{mid}.png"
            if not mask_path.exists():
                print(f"[enrich] obj {mid}: mask not found, skipping", flush=True)
                n_skip += 1
                continue

            mask_arr = np.array(Image.open(mask_path).convert("L"))
            shape = _compute_shape(mask_arr)
            obj["shape"] = shape
            if shape["is_thin_strip"]:
                n_thin += 1

            crop_path = crops_dir / f"crop_{mid}.png"
            full_path = crops_dir / f"full_{mid}.png"
            ok_crop = _make_crop_with_outline(image, mask_arr, crop_path)
            ok_full = _make_full_with_outline(image, mask_arr, full_path)
            if not (ok_crop and ok_full):
                print(f"[enrich] obj {mid}: empty mask, vlm_check skipped", flush=True)
                obj["vlm_check"] = {
                    "content": "",
                    "matches_class": False,
                    "confidence": 0.0,
                    "suspected_actual": "empty_mask",
                }
                n_skip += 1
                continue

            user_prompt = _build_user_prompt(
                assigned_class,
                bbox=obj.get("bbox_xy", [0.0, 0.0, 0.0, 0.0]),
                shape=shape,
            )
            result = run_qwen(
                torch_module=torch_mod,
                processor=processor,
                model=model,
                model_device=model_device,
                user_prompt=user_prompt,
                image_paths=[full_path, crop_path],
                system_prompt=VLM_SYSTEM_PROMPT,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                do_sample=False,
            )
            vlm = _parse_vlm_response(result["text"], assigned_class)
            obj["vlm_check"] = vlm
            if not vlm["matches_class"]:
                n_mismatch += 1

            n_ok += 1
            print(
                f"[enrich] obj {mid:>3} class='{assigned_class}' "
                f"aspect={shape['aspect_ratio']:.2f} "
                f"compact={shape['compactness']:.2f} "
                f"thin={'Y' if shape['is_thin_strip'] else 'n'} "
                f"matches={'Y' if vlm['matches_class'] else 'N'} "
                f"conf={vlm['confidence']:.2f} "
                f"actual='{vlm['suspected_actual']}'",
                flush=True,
            )
    finally:
        if tmpctx is not None:
            tmpctx.cleanup()

    data["history"].append({
        "step": "enrich_mask_attributes",
        "n_objects": n_total,
        "n_enriched": n_ok,
        "n_thin_strip": n_thin,
        "n_class_mismatch": n_mismatch,
        "model": args.model,
        "timestamp": _now_iso(),
    })
    with open(attr_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print(
        f"[enrich] done — enriched {n_ok}/{n_total} "
        f"({n_thin} thin strips, {n_mismatch} class mismatches, {n_skip} skipped)",
        flush=True,
    )


if __name__ == "__main__":
    main()
